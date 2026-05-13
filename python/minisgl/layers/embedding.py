from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import indexing

        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算当前 batch 每个请求用于采样的 logits。
        # 这个函数先在 prefill 阶段从 `x` 中取出每个请求最后一个位置的 hidden state，
        # 再与本 rank 持有的词表分片做线性映射；如果启用了 tensor parallel，
        # 最后把各 rank 的局部 logits 拼回完整词表维度。
        #
        # 例子：prefill 时两个请求新增 token 数分别为 3 和 2，`x` 会包含这 5 个位置；
        # 这里会只取每个请求的最后一个位置，最终输出 2 行 logits。
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            # prefill 只需要每个请求最后一个位置的 logits 用于下一 token 采样。
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        # `x` 是最后一个位置的 hidden state，形状可看作 `[bs, hidden_size]`；
        # `module.weight` 是词表权重表，形状可看作 `[vocab_size_tp, hidden_size]`。
        # 这里做一次线性映射，相当于让每个 hidden state 和当前 rank 负责的每个词向量分别打分，
        # 得到该词表分片上的 logits，而不是概率；后续还需要在完整词表维度上做 softmax 才是概率分布。
        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            # `all_gather` 后形状可看作 `[tp_size, vocab_size_tp]`，
            # 这里直接展平成 `[1, tp_size * vocab_size_tp]`，把各 rank 的词表分片首尾拼起来。
            # 末尾再裁到 `self.num_embeddings`，是因为按 TP 分片时可能做了向上取整，尾部会有补齐出来的无效位置。
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        # `all_gather` 后形状可看作 `[bs * tp_size, vocab_size_tp]`；
        # 这里先改成 `[tp_size, bs, vocab_size_tp]`，再转成 `[bs, tp_size, vocab_size_tp]`，
        # 让同一个样本在不同 rank 上的词表分片排到一起，最后展平为 `[bs, tp_size * vocab_size_tp]`，
        # 也就是每个样本对应一行完整词表 logits。末尾同样需要裁掉向上取整补出的无效位置。
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
