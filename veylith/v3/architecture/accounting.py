"""Exact analytical parameter accounting and analytical memory/compute budgets.

Three counts are kept separate:

1. ``total``: all trainable parameters (shared weights counted once).
2. ``active_unique``: unique parameters touched by one token in a sparse
   forward pass (top-k experts, shared expert, one embedding/head, norms,
   attention, router).
3. ``active_dense``: the same with the dense-combined MoE path, where every
   expert runs for every token.

Training FLOPs use ``6 * tokens * active_params`` plus an explicit attention
term. They are not wall-clock predictions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig
from .model import GPT, RMSNorm


def analytical_counts(cfg: ModelConfig) -> dict:
    cfg.validate()
    head_dim = cfg.n_embd // cfg.n_head
    embedding = cfg.vocab_size * cfg.n_embd
    head = 0 if cfg.tie_embeddings else embedding
    final_norm = cfg.n_embd
    norms = 2 * cfg.n_embd * cfg.n_layer + final_norm
    attention = cfg.n_layer * (2 * cfg.n_embd ** 2
                               + 2 * cfg.n_embd * cfg.n_kv_head * head_dim)
    dense_ffn_dim = int(cfg.n_embd * 8 / 3)
    dense_ffn = cfg.first_k_dense * 3 * cfg.n_embd * dense_ffn_dim
    n_moe = cfg.n_layer - cfg.first_k_dense
    router = n_moe * cfg.n_embd * cfg.n_experts
    expert = 3 * cfg.n_embd * cfg.expert_ffn_dim
    routed = n_moe * cfg.n_experts * expert
    shared = n_moe * 3 * cfg.n_embd * cfg.shared_ffn_dim if cfg.shared_expert else 0
    total = (embedding + head + attention + norms + dense_ffn + router
             + routed + shared)
    active_base = embedding + head + attention + norms + dense_ffn + router
    active_unique = active_base + n_moe * (
        cfg.n_experts_per_tok * expert
        + (3 * cfg.n_embd * cfg.shared_ffn_dim if cfg.shared_expert else 0))
    active_dense = active_base + n_moe * (
        cfg.n_experts * expert
        + (3 * cfg.n_embd * cfg.shared_ffn_dim if cfg.shared_expert else 0))
    if active_unique != total and cfg.moe_impl == "dense":
        # the config flag is an execution mode, not a different architecture
        pass
    return {
        "embedding": embedding,
        "output_head": head,
        "attention": attention,
        "norms": norms,
        "dense_ffn": dense_ffn,
        "router": router,
        "routed_experts": routed,
        "shared_experts": shared,
        "total": total,
        "active_unique": active_unique,
        "active_sparse": active_unique,
        "active_dense": active_dense,
        "active_ratio": active_unique / total,
    }


def count_model_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def active_parameter_count(model: GPT, idx: torch.Tensor) -> int:
    """Unique parameters touched by one forward pass, measured with hooks."""
    touched: set[int] = set()
    handles = []

    def record(module, inputs, output):
        for parameter in module.parameters():
            touched.add(id(parameter))

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding, RMSNorm)):
            handles.append(module.register_forward_hook(record))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(idx)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return sum(parameter.numel() for parameter in model.parameters()
               if id(parameter) in touched)


def full_layers(cfg: ModelConfig) -> int:
    return sum(1 for index in range(cfg.n_layer)
               if (index + 1) % cfg.full_attn_every == 0)


def attention_flops(cfg: ModelConfig, tokens: int) -> int:
    """Training attention FLOPs (QK + AV), full and sliding-window layers."""
    full = full_layers(cfg)
    windowed = cfg.n_layer - full
    per_token = (12 * cfg.block_size * cfg.n_embd * full
                 + 12 * min(cfg.block_size, cfg.sliding_window) * cfg.n_embd * windowed)
    return tokens * per_token


def training_flops(cfg: ModelConfig, tokens: int, dense_combined: bool = False) -> dict:
    counts = analytical_counts(cfg)
    active = counts["active_dense"] if dense_combined else counts["active_sparse"]
    six_pd = 6 * tokens * active
    attention = attention_flops(cfg, tokens)
    return {
        "tokens": tokens,
        "active_params": active,
        "six_pd_flops": six_pd,
        "attention_flops": attention,
        "total_flops": six_pd + attention,
        "petaflop_days_at_50_tflops": (six_pd + attention) / (50e12 * 86400),
    }


def wall_clock_estimate(cfg: ModelConfig, tokens: int,
                        anchor_tokens_per_s: float = 101_105.0,
                        anchor_active: int = 25_602_304,
                        block_factor: float = 1.3) -> dict:
    """Rough extrapolation from the P1-1 profile. Not a wall-clock prediction."""
    active = analytical_counts(cfg)["active_sparse"]
    tokens_per_s = anchor_tokens_per_s / (active / anchor_active) / block_factor
    seconds = tokens / tokens_per_s
    return {"tokens": tokens, "tokens_per_s": tokens_per_s,
            "seconds": seconds, "hours": seconds / 3600}


def memory_estimate(cfg: ModelConfig, microbatch: int, seq_len: int | None = None,
                    activation_checkpointing: bool = True,
                    dense_combined: bool = False, flash_attention: bool = True,
                    reserved_gb: float = 1.9, margin: float = 0.12) -> dict:
    """Analytical VRAM upper bound in GB. Not a measured value.

    Assumptions:
    - mixed precision: bf16 weights and grads, fp32 master weights and Adam
      moments = 16 bytes per parameter;
    - activation checkpointing stores layer inputs and recomputes the rest;
      without it, every layer keeps its transient activations;
    - flash attention means no T x T attention matrix is materialized;
    - ``reserved_gb`` covers the desktop/ROCm context observed on the box
      (about 1.9 GB) and a minimum workspace; ``margin`` is fragmentation and
      allocator headroom.
    """
    seq_len = seq_len or cfg.block_size
    if seq_len > cfg.block_size:
        raise ValueError("seq_len exceeds block_size")
    counts = analytical_counts(cfg)
    state_bytes = counts["total"] * 16
    state_gb = state_bytes / 1e9

    batch_tokens = microbatch * seq_len
    layer_input = batch_tokens * cfg.n_embd * 2
    transient_layer = batch_tokens * cfg.n_embd * 2 * 10
    routed = cfg.n_experts if dense_combined else cfg.n_experts_per_tok
    expert_transient = batch_tokens * routed * cfg.expert_ffn_dim * 2 * 4
    shared_transient = (batch_tokens * cfg.shared_ffn_dim * 2 * 4
                        if cfg.shared_expert else 0)
    router = batch_tokens * cfg.n_experts * 4
    if flash_attention:
        attention_buffer = 0
    else:
        full = full_layers(cfg)
        windowed = cfg.n_layer - full
        attention_buffer = microbatch * cfg.n_head * 2 * (
            full * seq_len * seq_len
            + windowed * seq_len * min(seq_len, cfg.sliding_window))
    if activation_checkpointing:
        stored = cfg.n_layer * layer_input
        working = transient_layer + expert_transient + shared_transient + router
    else:
        stored = 0
        working = cfg.n_layer * (layer_input + transient_layer) + expert_transient + shared_transient + router
    activation_gb = (stored + working + attention_buffer) / 1e9
    data_gb = batch_tokens * 8 / 1e9
    total_gb = (state_gb + activation_gb + data_gb + reserved_gb) * (1 + margin)
    fits_16gb = total_gb <= 16.0
    return {
        "microbatch": microbatch,
        "seq_len": seq_len,
        "activation_checkpointing": activation_checkpointing,
        "dense_combined": dense_combined,
        "flash_attention": flash_attention,
        "state_gb": round(state_gb, 3),
        "activation_gb": round(activation_gb, 3),
        "data_gb": round(data_gb, 3),
        "reserved_gb": reserved_gb,
        "total_gb": round(total_gb, 3),
        "fits_16gb": fits_16gb,
    }
