from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.version = 4 if is_sm100_supported() else 3

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        # 把当前 batch 整理成 FlashAttention 所需的 metadata：
        # - `cu_seqlens_q / cu_seqlens_k` 描述变长 Q/K 的分段边界
        # - `cache_seqlens` 描述每个请求当前可见的总长度
        # - `page_table` 描述每个请求逻辑位置到物理 KV page 的映射
        #
        # 这里使用 `batch.padded_reqs`，是为了与实际执行时的 padded batch
        # shape 保持一致，尤其是 decode + CUDA Graph 的场景。
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        # `seqlens_q` 是本轮真正新增参与计算的 query 长度；
        # `seqlens_k` 是每个请求当前可见的总长度；
        # `cached_lens` 用来区分 decode / 普通 prefill / 部分 cache hit prefill。
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        # `cu_seqlens_q` 是 query 侧的 cumulative sequence lengths（前缀和），
        # 用来描述每个请求的 query 在扁平化数组中的起止范围。
        #
        # 例子：
        # - 如果每个请求的 query 长度分别是 [2, 3, 1]
        # - 那么对应的前缀和就是 [0, 2, 5, 6]
        # - 表示：
        #   请求 0 的 query 范围是 [0, 2)
        #   请求 1 的 query 范围是 [2, 5)
        #   请求 2 的 query 范围是 [5, 6)
        if max_seqlen_q == 1:
            # decode：每个请求这轮只计算 1 个 query，所以 q 的前缀和就是
            # [0, 1, 2, ..., padded_size]。
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # 普通 prefill 且没有 cache hit：这时每个请求 q 长度等于 k 长度，
            # 因此可以直接复用 `cu_seqlens_k`。
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            # 部分前缀命中 cache 的 prefill：这时 q 长度是 `extend_len`，
            # k 长度是 `device_len`，两者不同，需要单独构造 `cu_seqlens_q`。
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        # 全局 `page_table` 是按 token 粒度存储的：
        #
        #   page_table[table_idx, pos] = physical_token_index
        #
        # 也就是说，同一页中的每个 token 都各有一个条目。例如 `page_size = 4`
        # 时，一页可能在某一行表现为：
        #
        #   [20, 21, 22, 23]
        #
        # 这里的 20/21/22/23 是这一页 4 个 token 对应的物理位置。
        #
        # 但 FlashAttention 这里想要的是 page 粒度映射，而不是把一页里的每个
        # token 位置都传进去。所以这里分三步做：
        #
        # 1. 先取出 batch 中每个请求自己对应的那一行：
        #
        #      page_table[req.table_idx, ...]
        #
        # 2. 再按 `page_size` 做步长切片：
        #
        #      page_table[req.table_idx, : max_seqlen_k : self.page_size]
        #
        #    这等价于只取逻辑位置：
        #
        #      0, page_size, 2*page_size, ...
        #
        #    也就是每一页的起始 token 位置。
        #
        # 3. 最后把 batch 里每个请求取出的这一行 `stack` 成一个新的二维表，
        #    作为当前 batch 专用的局部 `page_table`。
        #
        # 例子：
        # - 如果 `page_size = 4`
        # - 且某个请求这一行前 12 个位置是：
        #   `[20,21,22,23, 40,41,42,43, 60,61,62,63]`
        # - 那么：
        #   `page_table[req.table_idx, :10:4]`
        #   取到的是 `[20, 40, 60]`
        # - 含义是：只保留第 0 页、第 1 页、第 2 页的起始 token 位置
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            # 把每页起始的 token index 转成 page index，供 FlashAttention
            # 按 page 粒度索引。
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: support FA4 on blackwell
    )
