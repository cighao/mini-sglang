from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # 为这个新接纳的请求分配一个运行时槽位。后续这个请求会占用
        # `token_pool[table_idx]` 和 `page_table[table_idx]` 这一整行。
        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: 设置已经命中 cache 的前缀部分
            # 取出该请求对应行的前 `cached_len` 个位置：
            # - `device_ids` 记录这段前缀的 token id
            # - `page_entry` 记录这段前缀对应的 KV Cache 物理位置
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            # 把请求输入中已经命中的前缀 token 拷到 `token_pool`。
            # 这样后续调度 / 组 batch 时，这个请求在逻辑上已经拥有这段前缀。
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            # 把 prefix cache 命中的物理 KV 位置写入 `page_table`。
            # 这样 attention / kernel 后续可以直接复用已有 KV，而不用重算
            # 前 `cached_len` 个 token。
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        # 还剩多少输入 token 需要继续处理。前 `cached_len` 个 token
        # 已经命中 cache，因此本轮只需要处理后面的部分。
        remain_len = pending_req.input_len - cached_len
        # 本轮最多只能消耗 `token_budget` 个 token；如果剩余输入更短，
        # 就一次处理完。
        chunk_size = min(self.token_budget, remain_len)
        # 如果本轮处理不完剩余输入，就返回 `ChunkedReq`，后续轮次再继续；
        # 否则直接返回完整的 `Req`。
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        # 扣减本轮 prefill token 预算。
        self.token_budget -= chunk_size
        # 为这个请求后续仍需处理的输入部分以及预期输出预留 cache 空间。
        # 这里策略偏保守，避免本轮接纳过多请求导致后续空间不足。
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        # 这轮新推进的 token 范围是 [cached_len, cached_len + chunk_size)。
        # 这里只把 token id 拷到 `token_pool`，真正的 KV page 分配由
        # 后续 scheduler 调用 `cache_manager.allocate_paged()` 完成。
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        # 返回“当前这轮可执行范围”的请求对象：
        # - `input_ids[: cached_len + chunk_size]` 表示本轮结束后，设备侧
        #   已经可见到的位置
        # - `cached_len` 之前的部分可直接复用已有 KV
        # - `[cached_len, cached_len + chunk_size)` 是本轮新增处理的部分
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        # 本轮 prefill 已经没有可用 token 预算，不能再接纳更多请求。
        if self.token_budget <= 0:
            return None

        # 如果这个请求上一轮已经作为 `ChunkedReq` 运行过，说明它已经拿到过
        # `table_idx` / `cache_handle`，本轮只需要沿用已有状态继续推进。
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # 否则这是一个新请求：先尝试做准入检查并分配初始资源
        # （如 cache handle、table_idx、已命中前缀的 page_table/token_pool）。
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        # 资源不足，当前请求无法加入本轮 prefill batch。
        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        # 当前没有待进入 prefill 的请求。
        if len(self.pending_list) == 0:
            return None

        # 为这一轮 prefill 创建一个临时的“加请求器”：
        # - `token_budget` 限制这轮最多处理多少个 token
        # - `reserved_size` 先为正在 decode 的请求预留空间，避免 prefill
        #   把 cache 资源全部占满
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        # 按 pending 队列顺序尝试装入本轮 batch。一旦某个请求无法加入，
        # 就停止继续看后面的请求。
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                # 默认先清空旧的 chunk 状态；如果本轮仍然没处理完，会在下面
                # 重新设置回去。
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    # 这个请求本轮只处理了一部分，需要把最新的 chunk 状态
                    # 挂回 `pending_req`，并在本轮结束后放回 pending 队列前部。
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        # 虽然有 pending 请求，但这轮一个都没能加入 batch。
        if len(reqs) == 0:
            return None
        # 下一轮的 pending 队列由两部分组成：
        # 1. 本轮没处理完的 chunked 请求（优先续跑）
        # 2. 原 pending 队列中这轮尚未消费到的剩余请求
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
