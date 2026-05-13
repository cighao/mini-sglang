from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            # 对于单个请求，只需要为“这轮新增进入 device 侧的部分”分配 page。
            # - `cached_len` 之前的 token 已经有可复用的 KV 映射
            # - `[cached_len, device_len)` 是这轮新增需要落到 KV Cache 的部分
            #
            # 由于底层按 page 管理，所以先把 token 边界换算成 page 边界。
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                # 记录这个请求需要新增多少页，以及后续应当把这些页写回
                # `page_table` 的哪一行、哪个 page 区间。
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            # 为整批请求一次性分配所需 page：
            # - `_allocate` 返回 page 起始位置
            # - `_page_to_token` 再把每个 page 展开成具体 token 位置
            #   （例如 page_size=4 时，page 起点 20 会展开为 [20, 21, 22, 23]）
            allocated = self._page_to_token(self._allocate(needed_pages))
            # 把新分配的物理 KV 位置批量写回各请求对应的 `page_table` 行中，
            # 这样后续 attention / kernel 就知道这些新增 token 的 KV 应该
            # 去哪里读写。
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    # `allocated` 是本轮新分配好的物理 KV 位置，已经按 token 粒度展开。
    # 这里需要把它们按照 `allocation_info`，批量写回各请求对应的
    # `page_table[table_idx, pos]`。
    needed_tokens = len(allocated)
    # 先在 host 端构造两组一维索引：
    # - `table_idx_host[i]` 表示第 i 个分配结果属于哪一行（哪个请求）
    # - `positions_host[i]` 表示写入该行的哪个逻辑 token 位置
    #
    # 使用 pinned memory 便于后续异步搬运到 device。
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        # `allocation_info` 按 page 记录区间，这里先换算成 token 区间。
        # 例如 page_size = 4, first_page = 1, last_page = 3 时，
        # 对应的逻辑位置区间就是 [4, 12)，也就是 token 位置 4~11。
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        # 这一段新分配的位置都属于同一个请求行 `table_idx`。
        table_idx_host[offset : offset + length].fill_(table_idx)
        # 这一段在该请求行里写入连续的逻辑 token 位置。
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    # 展开的目标写入位置数量必须与 `allocated` 的长度完全一致。
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    # 把 host 端构造好的二维高级索引搬到 device。
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    # 批量散写：
    # page_table[table_idxs[i], offsets[i]] = allocated[i]
    page_table[table_idxs, offsets] = allocated
