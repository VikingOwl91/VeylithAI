"""v3.1-A architecture candidates (vocabulary 8192, context 512).

Four configurations are defined:

A  primary: 8 layers, width 512, 8 experts top-2, no shared expert
B  depth variant: 10 layers, first 2 dense, 8 experts top-2
C  fallback: 12 layers, width 384, 8 experts top-2 plus one shared expert
D  routing variant: 8 layers, width 512, top-1 plus one shared expert

MTP is intentionally absent. See docs/V3.1-ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    name: str = "v3.1-a"
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 2
    n_embd: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0
    sliding_window: int = 256
    full_attn_every: int = 4
    n_experts: int = 8
    n_experts_per_tok: int = 2
    first_k_dense: int = 1
    expert_ffn_dim: int = 960
    shared_expert: bool = False
    shared_ffn_dim: int = 0
    tie_embeddings: bool = False
    moe_aux_weight: float = 0.01
    moe_impl: str = "sparse"  # sparse | dense

    def validate(self) -> None:
        problems = []
        if self.vocab_size <= 0:
            problems.append("vocab_size must be positive")
        if self.block_size < 2:
            problems.append("block_size must be at least 2")
        if self.n_layer <= 0:
            problems.append("n_layer must be positive")
        if self.n_head <= 0 or self.n_embd % self.n_head != 0:
            problems.append("n_embd must be divisible by n_head")
        if self.n_kv_head <= 0 or self.n_head % self.n_kv_head != 0:
            problems.append("n_head must be divisible by n_kv_head")
        head_dim = self.n_embd // self.n_head if self.n_head else 0
        if head_dim % 2 != 0:
            problems.append("head_dim must be even for RoPE")
        if self.n_experts <= 0:
            problems.append("n_experts must be positive")
        if not 1 <= self.n_experts_per_tok <= self.n_experts:
            problems.append("n_experts_per_tok must be in [1, n_experts]")
        if not 0 <= self.first_k_dense <= self.n_layer:
            problems.append("first_k_dense must be in [0, n_layer]")
        if self.expert_ffn_dim <= 0:
            problems.append("expert_ffn_dim must be positive")
        if self.shared_expert and self.shared_ffn_dim <= 0:
            problems.append("shared_ffn_dim must be positive when shared_expert is on")
        if self.shared_expert and self.first_k_dense >= self.n_layer:
            problems.append("at least one layer must be MoE")
        if not 0 <= self.dropout < 1:
            problems.append("dropout must be in [0, 1)")
        if self.moe_impl not in ("sparse", "dense"):
            problems.append("moe_impl must be sparse or dense")
        if self.block_size > 4096:
            problems.append("block_size above 4096 is not supported")
        if problems:
            raise ValueError("; ".join(problems))

    def to_dict(self) -> dict:
        return asdict(self)


CANDIDATES = {
    "A": ModelConfig(
        name="v3.1-a-primary", n_layer=8, n_embd=512, n_head=8, n_kv_head=2,
        n_experts=8, n_experts_per_tok=2, first_k_dense=1,
        expert_ffn_dim=960, shared_expert=False, shared_ffn_dim=0,
        tie_embeddings=False, block_size=512, sliding_window=256),
    "B": ModelConfig(
        name="v3.1-b-depth", n_layer=10, n_embd=512, n_head=8, n_kv_head=2,
        n_experts=8, n_experts_per_tok=2, first_k_dense=2,
        expert_ffn_dim=800, shared_expert=False, shared_ffn_dim=0,
        tie_embeddings=False, block_size=512, sliding_window=256),
    "C": ModelConfig(
        name="v3.1-c-fallback", n_layer=12, n_embd=384, n_head=8, n_kv_head=2,
        n_experts=8, n_experts_per_tok=2, first_k_dense=1,
        expert_ffn_dim=736, shared_expert=True, shared_ffn_dim=736,
        tie_embeddings=False, block_size=512, sliding_window=256),
    "D": ModelConfig(
        name="v3.1-d-top1", n_layer=8, n_embd=512, n_head=8, n_kv_head=2,
        n_experts=8, n_experts_per_tok=1, first_k_dense=1,
        expert_ffn_dim=896, shared_expert=True, shared_ffn_dim=896,
        tie_embeddings=False, block_size=512, sliding_window=256),
}


def get_candidate(name: str) -> ModelConfig:
    key = name.upper()
    if key not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; choose from {sorted(CANDIDATES)}")
    return ModelConfig(**CANDIDATES[key].to_dict())


def tiny_config(**overrides) -> ModelConfig:
    base = dict(
        name="tiny", vocab_size=64, block_size=16, n_layer=2, n_head=2,
        n_kv_head=1, n_embd=32, dropout=0.0, sliding_window=8,
        full_attn_every=4, n_experts=4, n_experts_per_tok=2,
        first_k_dense=1, expert_ffn_dim=32, shared_expert=True,
        shared_ffn_dim=32)
    base.update(overrides)
    return ModelConfig(**base)
