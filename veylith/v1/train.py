#!/usr/bin/env python3
"""Train a small GPT-style language model completely from scratch.

Single file, only needs PyTorch (plus sentencepiece for --tokenizer bpe).
Runs on AMD GPUs (ROCm), NVIDIA GPUs (CUDA), Apple Silicon (mps), or CPU.
On AMD/NVIDIA the GPU shows up as the 'cuda' device, so pick_device() needs
no AMD-specific code.

Usage:
    python train.py                                           # char tokenizer, full run
    python train.py --tokenizer bpe --bpe_vocab_size 1000     # custom BPE tokenizer
    python train.py --n_layer 2 --n_embd 64 --max_iters 60    # quick smoke test
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
    n_head: int = 8
    n_embd: int = 256
    block_size: int = 256
    dropout: float = 0.1
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

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask, written out by hand."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        x = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(x))


class Block(nn.Module):
    """One transformer block: attention + MLP, each with a residual connection."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_layer)
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


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
    p = argparse.ArgumentParser(description="train a tiny GPT from scratch")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                   help="char: 1 token per character | bpe: train a subword BPE on the corpus")
    p.add_argument("--bpe_vocab_size", type=int, default=1000)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
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

    cfg = Config(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                 block_size=args.block_size, dropout=args.dropout, vocab_size=vocab_size)
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M parameters")

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
