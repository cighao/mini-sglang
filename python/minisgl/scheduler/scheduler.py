from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # 按执行后端的要求对 batch 做 padding。主要是让 decode batch 在需要时
        # 对齐到已 capture 的 CUDA Graph batch size。
        self.engine.graph_runner.pad_batch(batch)
        # 为这轮请求里新增进入 device 侧的 token 分配 KV page，并把对应的
        # 物理位置写回 `page_table`。
        self.cache_manager.allocate_paged(batch.reqs)
        # 为本轮真正参与前向计算的 token 生成位置编号，主要用于位置编码。
        batch.positions = _make_positions(batch, self.device)
        # 构造本轮输入读取索引 `(table_idxs, positions)`，后续用它从
        # `token_pool` 中取出这轮 forward 的 input ids。
        input_mapping = _make_input_tuple(batch, self.device)
        # 构造本轮输出写回索引 `(table_idxs, write_positions)`，后续把
        # 采样出的 next token 写回各请求在 `token_pool` 中的下一个位置。
        write_mapping = _make_write_tuple(batch, self.device)
        # 根据 `input_mapping` 到 `page_table` 中查每个输入 token 对应的
        # KV Cache 物理位置，并按这轮输入 token 的顺序拼成一个连续张量。
        #
        # 这里 `page_table` 存的是：
        #
        #   page_table[table_idx, pos] = physical_kv_index
        #
        # 而 `input_mapping` 等价于一组二维索引 `(table_idxs, positions)`，
        # 表示“这轮 forward 需要读取哪些请求、哪些逻辑位置上的 token”。
        # 因此这一句逻辑上等价于：
        #
        #   batch.out_loc = page_table[table_idxs, positions]
        #
        # 它得到的 `batch.out_loc[i]` 表示：本轮第 i 个输入 token 在
        # KV Cache 中对应的物理位置。后续 attention / kernel 会据此
        # 读写这批 token 的 KV。
        #
        # 例子：
        # - 如果 `input_mapping` 等价于 `([7, 7, 3], [3, 4, 0])`
        # - 且 `page_table[7,3] = 100`, `page_table[7,4] = 101`, `page_table[3,0] = 40`
        # - 那么这一句得到的 `batch.out_loc` 就等价于 `[100, 101, 40]`
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    # 为本轮真正要参与前向计算的 token 构造位置编号，主要用于位置编码。
    # 对单个请求来说，只需要为 `[cached_len, device_len)` 这一段生成位置：
    # - `cached_len` 之前的前缀已经在 KV Cache 中，不需要重新计算
    # - `[cached_len, device_len)` 是本轮新增要计算的 token
    #
    # 这里使用 `padded_reqs` 而不是 `reqs`，是为了在启用 CUDA Graph 时
    # 让位置张量与实际执行使用的 padded batch shape 保持一致。
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        # 为这个请求生成连续的位置区间：
        #   [cached_len, cached_len + 1, ..., device_len - 1]
        # 并顺序写入整批位置张量中，逻辑上等价于：
        #
        #   indices_host[offset : offset + length] =
        #       torch.arange(req.cached_len, req.device_len)
        #
        # 例子：
        # - 请求 A: cached_len = 3, device_len = 5 -> 写入 [3, 4]
        # - 请求 B: cached_len = 0, device_len = 4 -> 写入 [0, 1, 2, 3]
        #
        # 如果 A 先写、B 后写，那么最终整批 positions 会是：
        #
        #   [3, 4, 0, 1, 2, 3]
        #
        # 后续这些位置会与 input_mapping 取出的输入 token 一一对应，
        # 用于位置编码 / RoPE。
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    # pinned host memory -> device，便于后续异步搬运。
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # 为 `token_pool` 构造本轮输入的二维索引 `(table_idxs, positions)`。
    # 后续会用它直接取出：
    #
    #   token_pool[table_idxs, positions]
    #
    # 其中：
    # - `table_idxs[i]` 表示第 i 个输入 token 属于哪个请求行
    # - `positions[i]` 表示该 token 在该请求中的逻辑位置
    #
    # 例子：
    # - 如果某轮有两个请求，分别对应：
    #   table_idx=7, positions=[3, 4]
    #   table_idx=3, positions=[0, 1, 2, 3]
    # - 那么返回的二维索引等价于：
    #   ([7, 7, 3, 3, 3, 3], [3, 4, 0, 1, 2, 3])
    # - 后续取到的输入就是：
    #   token_pool[7,3], token_pool[7,4], token_pool[3,0], ...
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        # 这个请求本轮有 `length` 个输入 token 要参与前向，它们都来自
        # 同一个请求行 `req.table_idx`，所以把这一段全部填成相同的 table_idx。
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    # 返回 `(table_idxs, positions)`，与 `batch.positions` 一一配对。
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # 为本轮生成出的 next token 构造写回 `token_pool` 的二维索引
    # `(table_idxs, write_positions)`。
    # 后续会用它直接写入：
    #
    #   token_pool[table_idxs, write_positions] = next_tokens_gpu
    #
    # 对单个真实请求来说，写回位置通常是 `req.device_len`，即当前已存在
    # token 之后的下一个槽位。
    #
    # 例子：
    # - 如果两个请求分别对应：
    #   table_idx=7, device_len=5
    #   table_idx=3, device_len=2
    # - 那么返回的二维索引等价于：
    #   ([7, 3], [5, 2])
    # - 后续写回效果就是：
    #   token_pool[7,5] = next_token_of_req_7
    #   token_pool[3,2] = next_token_of_req_3
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    # 只有真实请求会产生输出 token，因此这里使用 `batch.reqs`，
    # 不包含 padded 用的 dummy request。
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
