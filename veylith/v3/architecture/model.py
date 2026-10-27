"""Reference v3.1-A decoder-only MoE model (CPU validation build).

No MTP. No training loop. No flash-attention dependency: the reference
forward is a readable manual attention path. A GPU training implementation
should switch the attention kernel to SDPA/flash and keep the same masks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(seq_len: int, head_dim: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """GQA + RoPE. Every `full_attn_every`-th layer is full; the rest use a
    sliding window. The mask is causal in every layer."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.full = (layer_idx + 1) % cfg.full_attn_every == 0
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        cos, sin = rope_tables(cfg.block_size, self.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        q = self.q_proj(x).view(B, T, self.cfg.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.cfg.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.cfg.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        i = torch.arange(T, device=x.device)[:, None]
        j = torch.arange(T, device=x.device)[None, :]
        allowed = j <= i
        if not self.full:
            allowed = allowed & (i - j < self.cfg.sliding_window)
        if self.cfg.attention_impl == "sdpa":
            # bool mask: True means the key participates. The same allowed
            # matrix as the manual path, so the semantics cannot drift.
            try:
                attended = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=allowed[None, None, :, :],
                    dropout_p=self.cfg.dropout if self.training else 0.0,
                    is_causal=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    "SDPA does not support this mask/backend combination; "
                    "use attention_impl='manual' or change the backend") from exc
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~allowed[None, None, :, :],
                                  torch.finfo(att.dtype).min)
            att = self.attn_drop(F.softmax(att, dim=-1))
            attended = att @ v
        out = attended.transpose(1, 2).reshape(B, T, self.cfg.n_head * self.head_dim)
        return self.resid_drop(self.o_proj(out))


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MoE(nn.Module):
    """Routed experts with optional shared expert and Switch-style aux loss.

    ``sparse`` evaluates only the top-k experts per token (the algorithmic
    active count). ``dense`` evaluates every expert and combines them with the
    top-k weights; it is the safe fallback for the unresolved ROCm sparse
    fault, at E/k times the expert FLOPs.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(cfg.n_embd, cfg.expert_ffn_dim, cfg.dropout)
            for _ in range(cfg.n_experts))
        self.shared = (SwiGLU(cfg.n_embd, cfg.shared_ffn_dim, cfg.dropout)
                       if cfg.shared_expert else None)

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(B * T, C)
        logits = self.gate(flat).float()
        top_logits, top_idx = logits.topk(self.cfg.n_experts_per_tok, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        assign = F.one_hot(top_idx, self.cfg.n_experts).float()
        # Switch-style load balancing over assignments, as in v2.
        aux = self.cfg.n_experts * (assign.mean(dim=(0, 1)) * probs.mean(0)).sum()

        out = self.shared(flat) if self.shared is not None else flat.new_zeros(flat.shape)
        if self.cfg.moe_impl == "dense":
            weights = (assign * top_weights.unsqueeze(-1)).sum(1)
            for expert_index, expert in enumerate(self.experts):
                out = out + weights[:, expert_index:expert_index + 1] * expert(flat)
        else:
            for expert_index, expert in enumerate(self.experts):
                mask = (top_idx == expert_index).any(dim=-1)
                if not bool(mask.any()):
                    continue
                token_weights = (top_weights * (top_idx == expert_index)).sum(dim=-1)[mask]
                out = out.index_add(
                    0, mask.nonzero(as_tuple=False).squeeze(1),
                    token_weights.unsqueeze(-1) * expert(flat[mask]))
        return out.view(B, T, C), aux, top_idx


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg, layer_idx)
        self.ln2 = RMSNorm(cfg.n_embd)
        if layer_idx < cfg.first_k_dense:
            self.mlp = SwiGLU(cfg.n_embd, int(cfg.n_embd * 8 / 3), cfg.dropout)
            self.is_moe = False
        else:
            self.mlp = MoE(cfg)
            self.is_moe = True

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        mlp_in = self.ln2(x)
        if self.is_moe:
            mlp_out, aux, idx = self.mlp(mlp_in)
            return x + mlp_out, aux, idx
        return x + self.mlp(mlp_in), x.new_zeros(()), None


class GPT(nn.Module):
    """Decoder-only model. Position t predicts token t+1; no MTP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, index) for index in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if module.weight.device.type == "meta":
                return
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None, return_metrics=False, return_router=False):
        if idx.dim() != 2:
            raise ValueError("idx must have shape (batch, sequence)")
        _, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.tok_emb(idx)
        aux_total = x.new_zeros(())
        router = []
        for block in self.blocks:
            x, aux, top_idx = block(x)
            aux_total = aux_total + aux
            if top_idx is not None:
                router.append(top_idx)
        logits = self.head(self.ln_f(x))
        loss = None
        metrics = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx")
            main_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = main_ce + self.cfg.moe_aux_weight * aux_total
            metrics = {"main_ce": main_ce, "aux_loss": aux_total,
                       "composite_loss": loss}
        elif return_metrics:
            metrics = {"main_ce": None, "aux_loss": aux_total,
                       "composite_loss": None}
        if return_metrics and return_router:
            return logits, loss, metrics, router
        if return_metrics:
            return logits, loss, metrics
        if return_router:
            return logits, loss, router
        return logits, loss
