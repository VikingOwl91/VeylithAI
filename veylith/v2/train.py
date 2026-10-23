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
"""

import argparse
import contextlib
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


def pick_device():
    if torch.cuda.is_available():  # NVIDIA (CUDA) and AMD (ROCm) both land here
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def get_batch(split_data, block_size, batch_size, device):
    """Random (x, y) windows; y is x shifted by one token."""
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


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
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        assert (cfg.n_embd // cfg.n_head) % 2 == 0  # RoPE needs an even head_dim
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
        router_logits = self.gate(x_flat)                        # (N, E)
        topk_weights, topk_idx = router_logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)         # (N, k)
        out = self.shared(x_flat)                              # shared expert, always on
        for e, expert in enumerate(self.experts):
            w_e = ((topk_idx == e).float() * topk_weights).sum(-1)   # (N,)
            mask = w_e > 0
            if mask.any():
                contrib = torch.zeros_like(out)
                # autocast promotes the router softmax to fp32; cast back to match
                contrib[mask] = (w_e[mask].unsqueeze(1) * expert(x_flat[mask])).to(out.dtype)
                out = out + contrib
        # load balancing: E * sum_e (routed_fraction_e * mean_router_prob_e)
        probs = F.softmax(router_logits, dim=-1)               # (N, E)
        routed_frac = F.one_hot(topk_idx, self.n_experts).float().mean(dim=(0, 1))
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)  # RoPE encodes positions, no position table
        aux_total = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux
        x = self.ln_f(x)
        logits = self.head(x)  # position t predicts token t+1
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if self.cfg.n_mtp >= 1:
                # multi-token prediction: position t also predicts token t+2
                mtp_logits = logits[:, :-1]
                mtp_targets = targets[:, 1:]
                mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, logits.size(-1)),
                                           mtp_targets.reshape(-1))
                loss = loss + self.cfg.mtp_weight * mtp_loss
            loss = loss + self.cfg.moe_aux_weight * aux_total
        return logits, loss


def param_report(model, cfg):
    """(total params, active-per-token params). MoE capacity vs compute."""
    total = sum(p.numel() for p in model.parameters())
    active = model.tok_emb.weight.numel() + model.head.weight.numel()
    for block in model.blocks:
        active += sum(p.numel() for p in block.attn.parameters())
        if block.is_moe:
            active += sum(p.numel() for p in block.mlp.shared.parameters())
            active += cfg.n_experts_per_tok * sum(
                p.numel() for p in block.mlp.experts[0].parameters())
        else:
            active += sum(p.numel() for p in block.mlp.parameters())
    return total, active


# --------------------------------------------------------------- training

def lr_at(step, max_iters, warmup_iters, base_lr):
    """Linear warmup, then cosine decay down to 10% of base lr."""
    if step < warmup_iters:
        return base_lr * (step + 1) / warmup_iters
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device, autocast):
    model.eval()
    losses = {}
    for split, split_data in (("train", train_data), ("val", val_data)):
        total = torch.zeros(args.eval_iters)
        for i in range(args.eval_iters):
            x, y = get_batch(split_data, args.block_size, args.batch_size, device)
            with autocast:
                _, loss = model(x, y)
            total[i] = loss.item()
        losses[split] = total.mean().item()
    model.train()
    return losses


@torch.no_grad()
def generate(model, encode, decode, device, prompt="\n", max_new_tokens=500,
             temperature=0.8, top_k=40):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
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


def main():
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
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
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
    print(f"model: {total / 1e6:.2f}M total / ~{active / 1e6:.2f}M active per token "
          f"({n_moe} MoE layers x {cfg.n_experts} experts, top-{cfg.n_experts_per_tok}, "
          f"GQA {cfg.n_head}Q/{cfg.n_kv_head}KV, SWA {cfg.sliding_window}, "
          f"full every {cfg.full_attn_every}, RoPE, SwiGLU, MTP x{cfg.n_mtp})")

    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        print("mixed precision: bfloat16")
    else:
        autocast = contextlib.nullcontext()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    t0 = time.time()
    for step in range(args.max_iters):
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

        if step % 100 == 0 or step == args.max_iters - 1:
            elapsed = time.time() - t0
            print(f"iter {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"{step / max(elapsed, 1e-9):.1f} it/s")

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, args, device, autocast)
            print(f"  eval | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                # early stopping: keep the best-validation weights, not the last
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": asdict(cfg),  # plain dict: pickles without a __main__ class reference
        "tokenizer": tok_info,
        "val_loss": best_val,
        "args": vars(args),
    }, args.checkpoint)
    print(f"\nsaved checkpoint -> {args.checkpoint} (best val loss {best_val:.4f})")

    print("\n--- sample generation (prompt: 'ROMEO:') ---")
    print(generate(model, encode, decode, device, prompt="ROMEO:", max_new_tokens=300))


if __name__ == "__main__":
    main()
