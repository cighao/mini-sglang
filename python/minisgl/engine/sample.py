from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    # 把 logits 转成概率分布后，按当前 batch 的采样配置为每个请求选出
    # 一个 next token。这里的分支只决定“先用哪种约束过滤候选集合，再采样”。
    import flashinfer.sampling as sampling

    # - top-k：限制“最多看前多少个 token”
    # - top-p：限制“只看累计概率达到多少的那一批 token”

    # 因为`logits` 不是概率，所以这里先结合 temperature 做 softmax，得到采样概率。
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        # 没有限制候选集合时，直接从完整概率分布中采样。
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        # 只启用 top-k：先保留前 k 个候选，再从中采样。
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        # 只启用 top-p：先保留累计概率达到 p 的那批候选，再从中采样。
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    # 两个约束同时启用：候选集合需要同时满足 top-k 和 top-p。
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        # 把 batch 中每个请求的采样参数整理成一组批量参数，供后续 sampler
        # 在 GPU 上统一采样。这里不执行采样，只做参数准备和规范化。
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            # 整批都是 greedy decoding 时，不需要准备 temperature / top-k / top-p。
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        # 逐请求整理采样参数：
        # - greedy 请求会被映射成一个极小 temperature
        # - 非法或未启用的 top-k / top-p 会被替换成“无约束”的等价形式
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        # 只有当这批请求里至少有一个真的启用了对应约束时，才构造该参数张量。
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
