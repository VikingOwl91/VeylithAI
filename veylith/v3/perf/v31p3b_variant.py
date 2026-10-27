"""v3.1-PERF-3B experimental variant: single BF16 cast at the RMSNorm output.

Applied as instance-level method overrides by the prototype harness. The
production modules and files are not modified. This variant is only valid for
BF16 autocast training; the production path is unchanged.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from architecture.model import MoE, RMSNorm


def rmsnorm_forward_single_cast(self, x):
    """Same FP32 normalization, one BF16 cast of the final result."""
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    return norm.to(torch.bfloat16)


def moe_forward_fp32_accum(self, x):
    """Exact MoE.forward copy with the dense accumulator fixed to FP32."""
    B, T, C = x.shape
    flat = x.reshape(B * T, C)
    logits = self.gate(flat).float()
    top_logits, top_idx = logits.topk(self.cfg.n_experts_per_tok, dim=-1)
    top_weights = F.softmax(top_logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    assign = F.one_hot(top_idx, self.cfg.n_experts).float()
    aux = self.cfg.n_experts * (assign.mean(dim=(0, 1)) * probs.mean(0)).sum()

    if self.shared is not None:
        out = self.shared(flat)
    else:
        out = flat.new_zeros(flat.shape, dtype=torch.float32)
    if self.cfg.moe_impl == "dense":
        weights = (assign * top_weights.unsqueeze(-1)).sum(1)
        for expert_index, expert in enumerate(self.experts):
            contribution = (weights[:, expert_index:expert_index + 1]
                            * expert(flat)).to(out.dtype)
            out = out + contribution
    else:
        for expert_index, expert in enumerate(self.experts):
            mask = (top_idx == expert_index).any(dim=-1)
            if not bool(mask.any()):
                continue
            token_weights = (top_weights * (top_idx == expert_index)).sum(dim=-1)[mask]
            contribution = (token_weights.unsqueeze(-1)
                            * expert(flat[mask])).to(out.dtype)
            out = out.index_add(
                0, mask.nonzero(as_tuple=False).squeeze(1), contribution)
    return out.view(B, T, C), aux, top_idx


def apply_variant(model):
    patched = {"rmsnorm": 0, "moe": 0}
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.forward = types.MethodType(rmsnorm_forward_single_cast, module)
            patched["rmsnorm"] += 1
        elif isinstance(module, MoE):
            module.forward = types.MethodType(moe_forward_fp32_accum, module)
            patched["moe"] += 1
    return patched
