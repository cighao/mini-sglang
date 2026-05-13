from __future__ import annotations

#
# 本文件里的三个核心类，分别处在三个不同抽象层次：
#
# 1. `Qwen3DecoderLayer`
#    表示“一层 decoder block”。
#    它只关心单层内部怎么计算：RMSNorm -> Attention -> RMSNorm -> MLP。
#
# 2. `Qwen3Model`
#    表示“Transformer 主干”。
#    它负责把 token embedding、多层 `Qwen3DecoderLayer` 和最后的 norm
#    组装起来，输出的是 hidden states，而不是最终 logits。
#
# 3. `Qwen3ForCausalLM`
#    表示“完整的自回归语言模型”。
#    它在 `Qwen3Model` 外面再包一层 `lm_head`，把 hidden states 投影到词表
#    维度，得到 logits，供后续采样使用。
#
# 三者关系可以概括为：
#
#     Qwen3ForCausalLM
#         └── Qwen3Model
#                 └── N x Qwen3DecoderLayer
#
# 一次 forward 的调用链大致是：
#
#     Engine.forward_batch(batch)
#       -> with ctx.forward_batch(batch):
#            self.model.forward()              # 这里的 self.model 是 Qwen3ForCausalLM
#              -> Qwen3ForCausalLM.forward()
#                   -> input_ids = get_global_ctx().batch.input_ids
#                   -> hidden_states = Qwen3Model.forward(input_ids)
#                        -> embed_tokens(input_ids)
#                        -> for layer in layers:
#                             x, residual = layer.forward(x, residual)
#                        -> final norm
#                   -> logits = lm_head(hidden_states)
#
# 之所以拆成这三层，而不是写成一个大类，主要是为了：
# - 把“单层实现”“主干网络”“任务头”分开
# - 让 backbone 可以和不同任务头解耦
# - 让 dense / MoE 这类变体可以复用同一套组织方式，只替换局部模块
#
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    #
    # `Qwen3DecoderLayer` 表示“一层 Qwen3 decoder block”。
    #
    # 这一层内部包含四个主要子模块：
    # - `input_layernorm`: attention 前的 RMSNorm
    # - `self_attn`: 自注意力
    # - `post_attention_layernorm`: MLP 前的 RMSNorm
    # - `mlp`: 前馈网络
    #
    # 从结构上看，它对应标准 Transformer 的一层：
    #
    #     x
    #       -> norm
    #       -> attention
    #       -> norm
    #       -> mlp
    #
    # 但这里的实现不是最常见的
    # `x = x + Attn(Norm(x)); x = x + MLP(Norm(x))`
    # 这种直写形式，而是显式传递 `residual`，把“残差相加 + RMSNorm”
    # 融合进 `RMSNormFused.forward()` 里，以减少中间 tensor 和 kernel launch。
    #
    # 因此，这个 layer 的 forward 会同时维护两路状态：
    # - `x`: 当前真正参与 attention / mlp 计算的 hidden states
    # - `residual`: 残差分支
    #
    # 它们并不是彼此独立的两份结果，而是共同表示当前 block 状态。
    # 真正的 residual add 通常不会在本层末尾手写完成，而是在下一次
    # `RMSNormFused.forward(x, residual)` 时被融合处理。
    #
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 第一次 norm：
        # - 若 `residual is None`，返回 `norm(x)`，并把原始 `x` 保存为 residual
        # - 否则融合执行 `residual + x + rmsnorm`
        x, residual = self.input_layernorm.forward(x, residual)

        # 对 norm 后的 hidden states 做自注意力。
        x = self.self_attn.forward(x)

        # 第二次 norm：
        # 把 attention 输出并入 residual，再做进入 MLP 前的 norm。
        x, residual = self.post_attention_layernorm.forward(x, residual)

        # 前馈网络。
        x = self.mlp.forward(x)

        # 返回 `(x, residual)`，而不是立即手写做残差相加。
        # 下一层开头的 `input_layernorm.forward(x, residual)` 会继续接管这件事。
        return x, residual


#
# Qwen3Model 只表示 Transformer 主干本体：
# - token embedding
# - 多层 decoder
# - 最后的 norm
#
# 它的输出是 hidden states，而不是最终用于采样的 logits。
# 这里故意不把 lm_head 放进来，是为了把“主干网络”和“具体任务头”
# 分开：同一个主干理论上可以复用到不同任务，而 `ForCausalLM`
# 这层才表示“用于下一个 token 预测的完整语言模型”。
class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


#
# Qwen3ForCausalLM 在主干 `Qwen3Model` 外再包一层 `lm_head`：
# - `self.model` 负责把 input_ids 编码成 hidden states
# - `self.lm_head` 负责把 hidden states 投影到词表维度，得到 logits
#
# 推理时 engine 真正需要的是 logits 来做采样，因此外部实际使用的是
# `Qwen3ForCausalLM`，而不是只输出中间特征的 `Qwen3Model`。
class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        # `lm_head` 是最后的读出层：把主干输出的 hidden states 投影到
        # 整个词表维度，得到每个 token 的 logits，用于后续采样。
        # 可以把它理解成“拿当前语义状态去给词表里的所有候选词打分”。
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3ForCausalLM"]
