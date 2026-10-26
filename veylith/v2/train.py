#!/usr/bin/env python3
"""veylith v2: a miniature 2026 frontier-style transformer, from scratch.

Upgrades over v1 (frozen in ../v1):
  - RoPE rotary position embeddings (no learned position table)
  - RMSNorm instead of LayerNorm
  - GQA: 8 query heads, 2 KV heads
  - sliding-window attention, full attention every 4th layer
  - SwiGLU MLPs at 8/3 x hidden (same params as v1's 4x GELU)
  - MoE: 8 routed experts, top-2 per token, plus 1 shared expert,
    trained with a load-balancing auxiliary loss
  - multi-token prediction: position t also predicts token t+2

Same recipe otherwise: single file, PyTorch only (+ sentencepiece for
--tokenizer bpe), bf16 autocast, early stopping, AMD/NVIDIA/Apple/CPU.

Usage:
    python train.py                                           # char tokenizer, full run
    python train.py --tokenizer bpe --bpe_vocab_size 1000     # custom BPE tokenizer
    python train.py --n_layer 4 --n_embd 64 --max_iters 60    # quick smoke test
    python train.py --device cpu                              # force CPU
    python train.py --save-every 500 --resume                  # crash-safe, resumable

Checkpoints: `--checkpoint` holds the best-validation weights for inference.
`--state-path` (default `<checkpoint>.state.pt`) holds the full training
state (weights, optimizer, step, RNG) for `--resume`.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import sentencepiece as spm
except ImportError:
    spm = None

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join("data", "tinyshakespeare.txt")


@dataclass
class Config:
    n_layer: int = 6
    n_head: int = 8            # query heads
    n_kv_head: int = 2         # KV heads (GQA: n_head must divide by n_kv_head)
    n_embd: int = 256
    block_size: int = 256
    sliding_window: int = 128
    full_attn_every: int = 4  # every 4th layer uses full causal attention
    dropout: float = 0.1
    mlp_ratio: float = 8 / 3   # SwiGLU hidden width = mlp_ratio * n_embd
    n_experts: int = 8
    n_experts_per_tok: int = 2
    first_k_dense: int = 1     # first k layers use a dense MLP, the rest MoE
    n_mtp: int = 1             # extra future tokens predicted per position
    mtp_weight: float = 0.5
    moe_aux_weight: float = 0.01
    vocab_size: int = 0  # filled in after tokenizing


def pick_device(device="auto"):
    """`auto` keeps the historical pick; an explicit device is validated.

    An explicitly requested device never falls back silently: if it is not
    available, this raises ValueError naming the device.
    """
    if device == "auto":
        if torch.cuda.is_available():  # NVIDIA (CUDA) and AMD (ROCm) both land here
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA/ROCm is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps was requested but MPS is not available")
    return device


# ------------------------------------------------------------------- data

def download_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"downloading dataset -> {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return open(DATA_PATH, encoding="utf-8").read()


def train_bpe_tokenizer(text_path, model_path, vocab_size):
    """Train a BPE tokenizer on the corpus once; cached on disk."""
    if spm is None:
        raise ImportError("sentencepiece is required for --tokenizer bpe (pip install sentencepiece)")
    if os.path.exists(model_path):
        return
    print(f"training BPE tokenizer (vocab {vocab_size}, byte_fallback) -> {model_path}")
    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=model_path[: -len(".model")],
        model_type="bpe",
        vocab_size=vocab_size,
        byte_fallback=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=True,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
    )


def tokenizer_from_info(info):
    """Rebuild (encode, decode) from checkpoint-serializable tokenizer info."""
    if info["type"] == "bpe":
        sp = spm.SentencePieceProcessor(model_file=info["model_path"])
        return (lambda s: sp.encode(s, out_type=int)), (lambda ids: sp.decode(ids))
    stoi, itos = info["stoi"], info["itos"]
    return (lambda s: [stoi[c] for c in s]), (lambda ids: "".join(itos[i] for i in ids))


def load_data(tokenizer, bpe_vocab_size):
    """Return (train_ids, val_ids, encode, decode, vocab_size, tokenizer_info)."""
    text = download_data()
    if tokenizer == "bpe":
        model_path = os.path.join("data", f"tinyshakespeare_bpe{bpe_vocab_size}.model")
        train_bpe_tokenizer(DATA_PATH, model_path, bpe_vocab_size)
        sp = spm.SentencePieceProcessor(model_file=model_path)
        encode = lambda s: sp.encode(s, out_type=int)
        decode = lambda ids: sp.decode(ids)
        vocab_size = sp.get_piece_size()
        info = {"type": "bpe", "model_path": model_path}
        cache_path = model_path + ".ids.pt"
    else:
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: "".join(itos[i] for i in ids)
        vocab_size = len(stoi)
        info = {"type": "char", "stoi": stoi, "itos": itos}
        cache_path = DATA_PATH + ".ids.pt"

    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=True)
    else:
        data = torch.tensor(encode(text), dtype=torch.long)
        torch.save(data, cache_path)
    print(f"tokenizer: {tokenizer} | vocab {vocab_size} | {len(data):,} tokens "
          f"({len(text) / len(data):.2f} chars/token)")
    n = int(0.9 * len(data))
    return data[:n], data[n:], encode, decode, vocab_size, info


def get_batch(split_data, block_size, batch_size, device, generator=None):
    """Random (x, y) windows; y is x shifted by one token.

    `generator` pins the window choice without touching the global RNG
    (used by the fixed evaluation mode).
    """
    ix = torch.randint(len(split_data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# --------------------------------------------------------------- validation

def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_context_and_mtp(block_size, n_mtp):
    """The t+2 target needs two positions in a window."""
    _require(block_size >= 1, f"block_size must be >= 1, got {block_size}")
    _require(block_size >= 2 or n_mtp < 1,
             f"block_size must be >= 2 when n_mtp >= 1 (the t+2 target needs two "
             f"positions), got block_size {block_size} with n_mtp {n_mtp}")


def validate_config(cfg):
    """Reject model configs that cannot train sanely; called by GPT.__init__."""
    _check_context_and_mtp(cfg.block_size, cfg.n_mtp)
    _require(cfg.n_layer >= 1, f"n_layer must be >= 1, got {cfg.n_layer}")
    _require(cfg.n_embd >= 1, f"n_embd must be >= 1, got {cfg.n_embd}")
    _require(cfg.n_head >= 1, f"n_head must be >= 1, got {cfg.n_head}")
    _require(cfg.n_kv_head >= 1, f"n_kv_head must be >= 1, got {cfg.n_kv_head}")
    _require(cfg.sliding_window >= 1,
             f"sliding_window must be >= 1, got {cfg.sliding_window}")
    _require(cfg.full_attn_every >= 1,
             f"full_attn_every must be >= 1, got {cfg.full_attn_every}")
    _require(cfg.n_experts >= 1, f"n_experts must be >= 1, got {cfg.n_experts}")
    _require(1 <= cfg.n_experts_per_tok <= cfg.n_experts,
             f"n_experts_per_tok must be between 1 and n_experts ({cfg.n_experts}), "
             f"got {cfg.n_experts_per_tok}")
    _require(cfg.first_k_dense >= 0,
             f"first_k_dense must be >= 0, got {cfg.first_k_dense}")
    _require(cfg.n_mtp >= 0, f"n_mtp must be >= 0, got {cfg.n_mtp}")
    _require(cfg.mlp_ratio > 0, f"mlp_ratio must be > 0, got {cfg.mlp_ratio}")
    _require(cfg.vocab_size >= 1, f"vocab_size must be >= 1, got {cfg.vocab_size}")


def validate_args(args):
    """Reject training args that would fail later or silently produce garbage.

    Messages use the exact CLI spelling, which is mixed: the original flags use
    underscores (--eval_iters), the later ones dashes (--save-every).
    """
    _check_context_and_mtp(args.block_size, args.n_mtp)
    _require(args.max_iters >= 1, f"--max_iters must be >= 1, got {args.max_iters}")
    _require(args.batch_size >= 1, f"--batch_size must be >= 1, got {args.batch_size}")
    _require(args.eval_iters >= 1, f"--eval_iters must be >= 1, got {args.eval_iters}")
    _require(args.eval_interval >= 1,
             f"--eval_interval must be >= 1, got {args.eval_interval}")
    _require(args.save_every is None or args.save_every >= 0,
             f"--save-every must be >= 0 (0 disables it), got {args.save_every}")
    _require(args.warmup_iters >= 0,
             f"--warmup_iters must be >= 0, got {args.warmup_iters}")
    _require(args.lr > 0, f"--lr must be > 0, got {args.lr}")
    _require(args.stop_after is None or args.stop_after >= 0,
             f"--stop-after must be >= 0, got {args.stop_after}")
    _require(bool(args.checkpoint), "--checkpoint must not be empty")


# ------------------------------------------------------------------ model

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_table(seq_len, head_dim, theta=10000.0):
    """cos/sin tables for rotary position embeddings, shape (T, head_dim/2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, heads, T, head_dim); cos/sin: (T, head_dim/2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """GQA with RoPE. Sliding-window layers attend only to the last
    `sliding_window` positions; every `full_attn_every`-th layer is full."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert (cfg.n_embd // cfg.n_head) % 2 == 0, "head_dim (n_embd // n_head) must be even for RoPE"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.full = (layer_idx + 1) % cfg.full_attn_every == 0
        self.sliding_window = cfg.sliding_window
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        cos, sin = rope_table(cfg.block_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        k = k.repeat_interleave(self.n_rep, dim=1)  # GQA: share KV heads across Q groups
        v = v.repeat_interleave(self.n_rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        i = torch.arange(T, device=x.device)[:, None]
        j = torch.arange(T, device=x.device)[None, :]
        allowed = j <= i  # causal
        if not self.full:
            allowed = allowed & (i - j < self.sliding_window)
        att = att.masked_fill(~allowed[None, None, :, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        x = (att @ v).transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.resid_drop(self.o_proj(x))


class SwiGLU(nn.Module):
    """gate = silu(W1 x), up = W2 x, out = W3 (gate * up)."""

    def __init__(self, n_embd, hidden_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MoE(nn.Module):
    """Mixture of routed experts plus one always-on shared expert.

    Dense-combined: routing is still top-k, but every expert computes on
    every token and the router weights combine the results. That is a
    simplification of true sparse routing (only the top-k experts run per
    token): it costs E/k times the expert FLOPs, but needs no
    variable-size indexing (nonzero/index_select/index_add), whose
    gather/scatter + sync pattern faults intermittently on ROCm.
    Gradients reach experts and router through the same dense path.

    Returns (output, aux_loss). The aux loss is the classic Switch-style
    load-balancing penalty; the frontier models in ../reference use
    aux-loss-free balancing instead (topk_method "noaux_tc" in their
    configs), which is harder to read.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_tok
        expert_dim = int(cfg.n_embd * cfg.mlp_ratio)
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(cfg.n_embd, expert_dim, cfg.dropout) for _ in range(cfg.n_experts))
        self.shared = SwiGLU(cfg.n_embd, expert_dim, cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        router_logits = self.gate(x_flat).float()              # (N, E) fp32: stable topk
        topk_weights, topk_idx = router_logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)         # (N, k), grad to router
        assign = (topk_idx.unsqueeze(-1)
                  == torch.arange(self.n_experts, device=x.device)).float()  # (N, k, E)
        w = (assign * topk_weights.unsqueeze(-1)).sum(1)       # (N, E) dense routing weights
        out = self.shared(x_flat)                              # shared expert, always on
        for e, expert in enumerate(self.experts):
            out = out + (w[:, e:e + 1] * expert(x_flat)).to(out.dtype)
        # load balancing: E * sum_e (routed_fraction_e * mean_router_prob_e)
        routed_frac = assign.mean(dim=(0, 1))                 # (E,)
        probs = F.softmax(router_logits, dim=-1)               # (N, E)
        mean_prob = probs.mean(0)                              # (E,)
        aux = self.n_experts * (routed_frac * mean_prob).sum()
        return out.view(B, T, C), aux


class Block(nn.Module):
    """One transformer block. The first `first_k_dense` layers use a dense
    MLP; the rest are MoE."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln2 = RMSNorm(cfg.n_embd)
        if layer_idx < cfg.first_k_dense:
            self.mlp = SwiGLU(cfg.n_embd, int(cfg.n_embd * cfg.mlp_ratio), cfg.dropout)
            self.is_moe = False
        else:
            self.mlp = MoE(cfg)
            self.is_moe = True

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        mlp_in = self.ln2(x)
        if self.is_moe:
            mlp_out, aux = self.mlp(mlp_in)
            return x + mlp_out, aux
        return x + self.mlp(mlp_in), x.new_zeros(())


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        validate_config(cfg)
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, return_metrics=False):
        B, T = idx.shape
        x = self.tok_emb(idx)  # RoPE encodes positions, no position table
        aux_total = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux
        x = self.ln_f(x)
        logits = self.head(x)  # position t predicts token t+1
        loss = None
        metrics = None
        if targets is not None:
            main_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = main_ce
            mtp_ce = None
            if self.cfg.n_mtp >= 1:
                # multi-token prediction: position t also predicts token t+2
                mtp_logits = logits[:, :-1]
                mtp_targets = targets[:, 1:]
                mtp_ce = F.cross_entropy(mtp_logits.reshape(-1, logits.size(-1)),
                                         mtp_targets.reshape(-1))
                loss = loss + self.cfg.mtp_weight * mtp_ce
            loss = loss + self.cfg.moe_aux_weight * aux_total
            metrics = {
                "main_ce": main_ce,
                "mtp_ce": mtp_ce,
                "aux_loss": aux_total,
                "composite_loss": loss,
            }
        if return_metrics:
            return logits, loss, metrics
        return logits, loss


def param_report(model, cfg):
    """(total params, active-per-token params). MoE capacity vs compute."""
    total = sum(p.numel() for p in model.parameters())
    active = model.tok_emb.weight.numel() + model.head.weight.numel()
    for block in model.blocks:
        active += sum(p.numel() for p in block.attn.parameters())
        if block.is_moe:
            active += sum(p.numel() for p in block.mlp.shared.parameters())
            active += cfg.n_experts * sum(  # dense-combined: all experts compute
                p.numel() for p in block.mlp.experts[0].parameters())
        else:
            active += sum(p.numel() for p in block.mlp.parameters())
    return total, active


# ------------------------------------------------------------ checkpoints

STATE_FORMAT = 2


def atomic_save(obj, path):
    """Write obj to path via a temp file in the same directory, then rename.

    The target only ever holds a complete checkpoint: a crash mid-write
    leaves the previous file untouched and removes the temp file.
    """
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".tmp"
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return target


def default_state_path(checkpoint):
    """Resume state next to the output checkpoint: ckpt.pt -> ckpt.state.pt."""
    root, _ext = os.path.splitext(os.path.abspath(checkpoint))
    return root + ".state.pt"


def _fingerprint_payload(cfg, args, tok_info):
    """What a resume must match. Deliberately excluded: max_iters (extending a
    run is allowed, but it does change the LR schedule), eval_interval,
    save_every, seed, paths and device (the RNG state is restored)."""
    return {
        "config": asdict(cfg),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_iters": args.warmup_iters,
        "tokenizer_type": (tok_info or {}).get("type"),
    }


def config_fingerprint(cfg, args, tok_info):
    payload = json.dumps(_fingerprint_payload(cfg, args, tok_info), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _eval_meta(args):
    """Evaluation settings, stored for traceability."""
    return {
        "mode": args.eval_mode,
        "seed": args.eval_seed if args.eval_mode == "fixed" else None,
        "iters": args.eval_iters,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
    }


def _eval_compat(args):
    """The evaluation and selection settings a resume must match."""
    return {
        "mode": args.eval_mode,
        "seed": args.eval_seed if args.eval_mode == "fixed" else None,
        "metric": args.select_metric,
    }


def _stored_eval_compat(state):
    """Same view for a state file; P0-1 states fall back to the old defaults."""
    evaluation = state.get("evaluation") or {}
    saved = state.get("args") or {}
    mode = evaluation.get("mode", saved.get("eval_mode", "legacy"))
    seed = evaluation.get("seed", saved.get("eval_seed"))
    metric = (state.get("selection") or {}).get("metric",
                                                saved.get("select_metric", "composite"))
    return {"mode": mode, "seed": seed if mode == "fixed" else None, "metric": metric}


def capture_rng_state():
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_train_state(path, *, model, optimizer, step, best_val, best_state,
                     cfg, tok_info, args, best_metrics=None):
    atomic_save({
        "format": STATE_FORMAT,
        "kind": "train_state",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "best_model": best_state,
        "best_metrics": best_metrics,
        "evaluation": _eval_meta(args),
        "selection": {"metric": args.select_metric, "value": best_val},
        "config": asdict(cfg),
        "tokenizer": tok_info,
        "args": vars(args),
        "rng": capture_rng_state(),
        "fingerprint": config_fingerprint(cfg, args, tok_info),
        "fingerprint_payload": _fingerprint_payload(cfg, args, tok_info),
    }, path)
    return path


def load_train_state(path, cfg, args, tok_info):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("kind") != "train_state" or state.get("format") != STATE_FORMAT:
        raise ValueError(
            f"{path} is not a resumable training state "
            f"(format {state.get('format')!r}, kind {state.get('kind')!r}). "
            f"Use --checkpoint for the inference artifact and --state-path for resume.")
    want = _fingerprint_payload(cfg, args, tok_info)
    got = state.get("fingerprint_payload")
    if got != want or state.get("fingerprint") != config_fingerprint(cfg, args, tok_info):
        diffs = []
        if isinstance(got, dict):
            for key in sorted(set(want) | set(got)):
                a, b = got.get(key), want.get(key)
                if a != b:
                    if key == "config" and isinstance(a, dict) and isinstance(b, dict):
                        for name in sorted(set(a) | set(b)):
                            if a.get(name) != b.get(name):
                                diffs.append(f"{name}: {a.get(name)!r} in checkpoint, "
                                             f"{b.get(name)!r} now")
                    else:
                        diffs.append(f"{key}: {a!r} in checkpoint, {b!r} now")
        detail = "; ".join(diffs) if diffs else "fingerprint mismatch"
        raise ValueError(f"resume config mismatch ({detail}). A run can only be "
                         f"resumed with the config it was trained with.")
    got_eval, want_eval = _stored_eval_compat(state), _eval_compat(args)
    if got_eval != want_eval:
        diffs = [f"{key}: {got_eval[key]!r} in checkpoint, {want_eval[key]!r} now"
                 for key in ("mode", "seed", "metric") if got_eval[key] != want_eval[key]]
        raise ValueError(f"resume evaluation mismatch ({'; '.join(diffs)}). Evaluation "
                         f"mode, eval seed and selection metric must not change mid-run.")
    return state


# --------------------------------------------------------------- training

def lr_at(step, max_iters, warmup_iters, base_lr):
    """Linear warmup, then cosine decay down to 10% of base lr."""
    if step < warmup_iters:
        return base_lr * (step + 1) / warmup_iters
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


SCORE_KEYS = {"composite": "composite_loss", "main_ce": "main_ce"}
METRIC_KEYS = ("main_ce", "mtp_ce", "aux_loss", "composite_loss", "main_perplexity")


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device, autocast):
    """Mean loss components per split.

    `legacy` draws fresh random windows and consumes the global RNG, which
    is the historical behaviour. `fixed` re-seeds a private generator, so
    every evaluation uses the same windows and the training RNG is not
    touched.
    """
    generator = None
    if args.eval_mode == "fixed":
        generator = torch.Generator().manual_seed(args.eval_seed)
    model.eval()
    result = {}
    for split, split_data in (("train", train_data), ("val", val_data)):
        sums = {key: 0.0 for key in ("main_ce", "mtp_ce", "aux_loss", "composite_loss")}
        mtp_seen = 0
        for _ in range(args.eval_iters):
            x, y = get_batch(split_data, args.block_size, args.batch_size, device,
                             generator=generator)
            with autocast:
                _, loss, metrics = model(x, y, return_metrics=True)
            sums["main_ce"] += metrics["main_ce"].item()
            sums["aux_loss"] += metrics["aux_loss"].item()
            sums["composite_loss"] += loss.item()
            if metrics["mtp_ce"] is not None:
                sums["mtp_ce"] += metrics["mtp_ce"].item()
                mtp_seen += 1
        n = args.eval_iters
        split_result = {
            "main_ce": sums["main_ce"] / n,
            "mtp_ce": sums["mtp_ce"] / mtp_seen if mtp_seen else float("nan"),
            "aux_loss": sums["aux_loss"] / n,
            "composite_loss": sums["composite_loss"] / n,
        }
        split_result["main_perplexity"] = math.exp(split_result["main_ce"])
        result[split] = split_result
    model.train()
    return result


def run_training(model, optimizer, train_data, val_data, args, device, autocast,
                 tok_info, start_step=0, best_val=float("inf"), best_state=None,
                 best_metrics=None, stats=None):
    """Train from start_step to args.max_iters (or args.stop_after).

    Writes a resumable state every `save_every` iterations and at an early
    stop. `best_val` is the best value of the configured selection metric.
    Returns (best_val, best_state, per-iteration losses); `stats`, if given,
    receives the full validation metrics of the selected checkpoint.
    """
    save_every = args.eval_interval if args.save_every is None else max(0, args.save_every)
    state_path = args.state_path or default_state_path(args.checkpoint)
    last_step = args.max_iters if args.stop_after is None else min(args.stop_after, args.max_iters)
    stops_early = last_step < args.max_iters
    final_step = None if stops_early else args.max_iters - 1

    losses = []
    t0 = time.time()
    for step in range(start_step, last_step):
        lr = lr_at(step, args.max_iters, args.warmup_iters, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0 or step == final_step:
            elapsed = time.time() - t0
            mem = f" | mem {torch.cuda.memory_allocated() / 1e9:.1f}GB" if device == "cuda" else ""
            print(f"iter {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"{(step - start_step) / max(elapsed, 1e-9):.1f} it/s{mem}")

        if step % args.eval_interval == 0 or step == final_step:
            evaluated = estimate_loss(model, train_data, val_data, args, device, autocast)
            train_m, val_m = evaluated["train"], evaluated["val"]
            print(f"  eval | train loss {train_m['composite_loss']:.4f} | "
                  f"val loss {val_m['composite_loss']:.4f}")
            print(f"  comp | train main {train_m['main_ce']:.4f} mtp {train_m['mtp_ce']:.4f} "
                  f"aux {train_m['aux_loss']:.4f} | "
                  f"val main {val_m['main_ce']:.4f} (ppl {val_m['main_perplexity']:.1f}) "
                  f"mtp {val_m['mtp_ce']:.4f} aux {val_m['aux_loss']:.4f}")
            score = val_m[SCORE_KEYS[args.select_metric]]
            if score < best_val:
                best_val = score
                best_metrics = dict(val_m)
                # early stopping: keep the best-validation weights, not the last
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"  best   -> {args.select_metric} {score:.4f} (iter {step})")

        due = save_every > 0 and (step + 1) % save_every == 0 and step + 1 < args.max_iters
        if due or (stops_early and step == last_step - 1):
            save_train_state(state_path, model=model, optimizer=optimizer, step=step + 1,
                             best_val=best_val, best_state=best_state, cfg=model.cfg,
                             tok_info=tok_info, args=args, best_metrics=best_metrics)
            print(f"  state  -> {state_path} (iter {step + 1})")

    if stats is not None:
        stats["best_metrics"] = best_metrics
    return best_val, best_state, losses


@torch.no_grad()
def generate(model, encode, decode, device, prompt="\n", max_new_tokens=500,
             temperature=0.8, top_k=40):
    if not prompt:
        raise ValueError("prompt must not be empty")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")
    try:
        prompt_ids = encode(prompt)
    except KeyError as exc:
        unknown = exc.args[0] if exc.args else "?"
        raise ValueError(
            f"prompt {prompt!r} contains {unknown!r}, which is not in this "
            f"checkpoint's character vocabulary. A char tokenizer has a fixed "
            f"set of characters; use a prompt the model was trained on, or a "
            f"BPE checkpoint (byte fallback encodes anything)."
        ) from exc
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
    model.train()
    return decode(idx[0].tolist())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="veylith v2: train a miniature frontier-style GPT from scratch")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--bpe_vocab_size", type=int, default=1000)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_kv_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--sliding_window", type=int, default=128)
    p.add_argument("--full_attn_every", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--n_experts_per_tok", type=int, default=2)
    p.add_argument("--first_k_dense", type=int, default=1)
    p.add_argument("--n_mtp", type=int, default=1)
    p.add_argument("--mtp_weight", type=float, default=0.5)
    p.add_argument("--moe_aux_weight", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt")
    p.add_argument("--eval-mode", choices=["legacy", "fixed"], default="legacy",
                   help="legacy: fresh random windows per evaluation (historical default); "
                        "fixed: the same seeded windows every time, training RNG untouched")
    p.add_argument("--eval-seed", type=int, default=12345,
                   help="window seed for --eval-mode fixed")
    p.add_argument("--select-metric", choices=["composite", "main_ce"], default="composite",
                   help="criterion for best checkpoint and early stopping "
                        "(default: composite, the historical behaviour)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="auto keeps the historical pick: CUDA/ROCm, then MPS, then CPU")
    p.add_argument("--save-every", type=int, default=None,
                   help="write a resumable state every N iterations "
                        "(default: eval_interval, 0 disables)")
    p.add_argument("--state-path", type=str, default=None,
                   help="resume state path (default: next to --checkpoint, <name>.state.pt)")
    p.add_argument("--resume", nargs="?", const="auto", default=None, metavar="PATH",
                   help="resume from a training state; bare --resume uses the default path")
    p.add_argument("--stop-after", type=int, default=None,
                   help="stop after N iterations and save state; --max_iters still sets the LR schedule")
    return p.parse_args(argv)


def main():
    args = parse_args()
    try:
        validate_args(args)
        device = pick_device(args.device)
    except ValueError as exc:
        raise SystemExit(str(exc))

    torch.manual_seed(args.seed)
    suffix = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    print(f"device: {device}{suffix}")

    train_data, val_data, encode, decode, vocab_size, tok_info = load_data(
        args.tokenizer, args.bpe_vocab_size)

    cfg = Config(n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head,
                 n_embd=args.n_embd, block_size=args.block_size,
                 sliding_window=args.sliding_window, full_attn_every=args.full_attn_every,
                 dropout=args.dropout, n_experts=args.n_experts,
                 n_experts_per_tok=args.n_experts_per_tok,
                 first_k_dense=args.first_k_dense, n_mtp=args.n_mtp,
                 mtp_weight=args.mtp_weight, moe_aux_weight=args.moe_aux_weight,
                 vocab_size=vocab_size)
    model = GPT(cfg).to(device)
    total, active = param_report(model, cfg)
    n_moe = sum(1 for b in model.blocks if b.is_moe)
    print(f"model: {total / 1e6:.2f}M total / ~{active / 1e6:.2f}M compute per token "
          f"({n_moe} MoE layers x {cfg.n_experts} experts, dense-combined, top-{cfg.n_experts_per_tok} routing, "
          f"GQA {cfg.n_head}Q/{cfg.n_kv_head}KV, SWA {cfg.sliding_window}, "
          f"full every {cfg.full_attn_every}, RoPE, SwiGLU, MTP x{cfg.n_mtp})")

    if args.eval_mode == "fixed":
        eval_desc = (f"fixed windows seed {args.eval_seed}, {args.eval_iters} batches of "
                     f"{args.batch_size} x {args.block_size} per split")
    else:
        eval_desc = f"legacy random windows, {args.eval_iters} batches per split"
    print(f"eval: {eval_desc} | selection: {args.select_metric}")

    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        print("mixed precision: bfloat16")
    else:
        autocast = contextlib.nullcontext()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step, best_val, best_state, best_metrics = 0, float("inf"), None, None
    if args.resume:
        resume_path = (default_state_path(args.checkpoint)
                       if args.resume == "auto" else args.resume)
        if not os.path.exists(resume_path):
            raise SystemExit(f"no training state at {resume_path}; "
                             f"start a run with --save-every first")
        state = load_train_state(resume_path, cfg, args, tok_info)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_val = float(state["best_val"])
        best_state = state.get("best_model")
        best_metrics = state.get("best_metrics")
        restore_rng_state(state["rng"])
        print(f"resumed {resume_path} at iteration {start_step} "
              f"(best {args.select_metric} {best_val:.4f})")
        if start_step >= args.max_iters:
            print(f"--max_iters {args.max_iters} already reached, nothing to do")
            return

    stats = {}
    best_val, best_state, losses = run_training(
        model, optimizer, train_data, val_data, args, device, autocast, tok_info,
        start_step=start_step, best_val=best_val, best_state=best_state,
        best_metrics=best_metrics, stats=stats)
    best_metrics = stats.get("best_metrics")

    if start_step + len(losses) < args.max_iters:
        state_path = args.state_path or default_state_path(args.checkpoint)
        print(f"\nstopped after {start_step + len(losses)} iterations, "
              f"resume with --resume {state_path}")
        return

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss = best_metrics["composite_loss"] if best_metrics else best_val
    atomic_save({
        "format": STATE_FORMAT,
        "kind": "model",
        "model": model.state_dict(),
        "config": asdict(cfg),  # plain dict: pickles without a __main__ class reference
        "tokenizer": tok_info,
        "val_loss": val_loss,  # composite val loss at the selected checkpoint
        "metrics": best_metrics,
        "selection": {"metric": args.select_metric, "value": best_val},
        "evaluation": _eval_meta(args),
        "step": start_step + len(losses),
        "args": vars(args),
    }, args.checkpoint)
    print(f"\nsaved checkpoint -> {args.checkpoint} "
          f"(composite val loss {val_loss:.4f}, selected by {args.select_metric})")

    print("\n--- sample generation (prompt: 'ROMEO:') ---")
    try:
        print(generate(model, encode, decode, device, prompt="ROMEO:", max_new_tokens=300))
    except ValueError as exc:
        print(f"sample generation skipped: {exc}")


if __name__ == "__main__":
    main()
