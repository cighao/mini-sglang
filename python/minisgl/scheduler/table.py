import torch


class TableManager:
    """管理活跃请求对应的表行。

    `table_idx` 是运行时槽位编号。
    每个活跃请求都会占用 `page_table` 和 `token_pool` 中的一行。

    - `page_table[table_idx, pos]` 表示该请求第 `pos` 个 token
      对应的 KV Cache 物理位置。
    - `token_pool[table_idx, pos]` 表示该请求第 `pos` 个 token 的
      token id。

    `_free_slots` 是可用运行时槽位池。初始化时所有行都空闲，因此：

        _free_slots = [0, 1, ..., max_running_reqs - 1]

    例子：
        如果请求槽位 `table_idx = 3` 的前 4 个 token 使用的 KV Cache
        物理位置是 `[20, 21, 22, 23]`，那么：

            page_table[3, 0:4] = [20, 21, 22, 23]

        这表示请求槽位 3 的第 0/1/2/3 个 token 需要到物理位置
        20/21/22/23 读写对应的 KV 状态。
    """

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        # 可用请求槽位（`table_idx`）池。请求运行结束后，对应槽位会在
        # `free` 中被回收，供后续请求复用。
        self._free_slots = list(range(max_running_reqs))
        # 按行存储“逻辑 token 位置 -> 物理 KV Cache 位置”的映射。
        # 每个活跃请求通过 `table_idx` 占用其中一行。
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # 与 `page_table` 配套，按请求行存储对应位置上的 token id。
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        # 为新接纳的运行中请求分配一个空闲运行时槽位。
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        # 请求离开运行集合后，回收对应的运行时槽位。
        self._free_slots.append(slot)
