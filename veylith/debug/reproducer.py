#!/usr/bin/env python3
"""Minimal reproducer for the ROCm memory access fault in veylith v2 sparse MoE.

Reuses the exact failing model code recovered from git history
(sparse_moe_v2_train.py, commit bfafa70) and runs a configurable
training loop on random data (no dataset download needed). Knobs are
CLI flags so the test matrix can drive it. Prints one line per 25
iters; a GPU fault kills the process, so the last printed iter tells
you how far it got.

Usage:
    python reproducer.py                          # sparse, bf16, 250 iters
    python reproducer.py --dtype fp32 --iters 100
    python reproducer.py --impl dense             # control: current v2 dense MoE
    python reproducer.py --device cpu             # control: CPU
"""

import argparse
import contextlib
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v2"))

import sparse_moe_v2_train as sparse_impl  # recovered failing code (bfafa70)
import train as dense_impl                 # current v2 (dense-combined MoE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--iters", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_kv_head", type=int, default=2)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--n_experts_per_tok", type=int, default=2)
    p.add_argument("--first_k_dense", type=int, default=1)
    p.add_argument("--n_mtp", type=int, default=1)
    p.add_argument("--sliding_window", type=int, default=128)
    p.add_argument("--full_attn_every", type=int, default=4)
    p.add_argument("--no_optimizer", action="store_true",
                   help="zero_grad + backward only; model stays at init "
                        "(deterministic fault mode)")
    p.add_argument("--no_backward", action="store_true",
                   help="forward only, skip loss.backward()")
    p.add_argument("--grad_filter", type=str, default="all",
                   help="comma-separated param groups to require grad: "
                        "all, experts, gate, shared, attn, emb_head, ln, none")
    p.add_argument("--trace_moe", action="store_true",
                   help="log per-expert routed token counts every iter")
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--eval_interval", type=int, default=125)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--vocab_size", type=int, default=1000)
    p.add_argument("--dataset_tokens", type=int, default=200000)
    args = p.parse_args()

    impl = sparse_impl if args.impl == "sparse" else dense_impl
    torch.manual_seed(args.seed)
    device = args.device

    cfg = impl.Config(
        n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head,
        n_embd=args.n_embd, block_size=args.block_size,
        sliding_window=args.sliding_window, full_attn_every=args.full_attn_every,
        dropout=0.0, n_experts=args.n_experts,
        n_experts_per_tok=args.n_experts_per_tok,
        first_k_dense=args.first_k_dense, n_mtp=args.n_mtp,
        vocab_size=args.vocab_size)
    model = impl.GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"impl={args.impl} device={device} dtype={args.dtype} "
          f"layers={args.n_layer} embd={args.n_embd} experts={args.n_experts} "
          f"topk={args.n_experts_per_tok} batch={args.batch_size} "
          f"block={args.block_size} params={n_params / 1e6:.2f}M", flush=True)

    if args.trace_moe:
        for li, block in enumerate(model.blocks):
            if getattr(block, "is_moe", False):
                for ei, expert in enumerate(block.mlp.experts):
                    def make_hook(li, ei):
                        def hook(module, inp, out):
                            print(f"    moe L{li} E{ei}: M={out.shape[0]}", flush=True)
                        return hook
                    expert.register_forward_hook(make_hook(li, ei))

    if args.grad_filter != "all":
        groups = set(args.grad_filter.split(","))
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        if "none" not in groups:
            def in_group(name):
                if "experts" in groups and ".mlp.experts." in name:
                    return True
                if "gate" in groups and ".mlp.gate." in name:
                    return True
                if "shared" in groups and ".mlp.shared." in name:
                    return True
                if "attn" in groups and ".attn." in name:
                    return True
                if "emb_head" in groups and (name.startswith("tok_emb.")
                                             or name.startswith("head.")):
                    return True
                if "ln" in groups and (".ln1." in name or ".ln2." in name
                                       or name.startswith("ln_f.")):
                    return True
                return False
            for name, p in model.named_parameters():
                if in_group(name):
                    p.requires_grad_(True)
        n_trainable = sum(p.requires_grad for p in model.parameters())
        print(f"grad_filter={args.grad_filter}: {n_trainable} trainable params", flush=True)

    data = torch.randint(0, args.vocab_size, (args.dataset_tokens,))
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    if args.dtype == "bf16" and device == "cuda":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(args.iters):
        x, y = impl.get_batch(train_data, args.block_size, args.batch_size, device)
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        if not args.no_backward and loss.requires_grad:
            loss.backward()
        if not args.no_optimizer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if args.trace_moe:
            print(f"iter {step:5d}", flush=True)
        if step % 25 == 0 or step == args.iters - 1:
            mem = f" mem={torch.cuda.memory_allocated() / 1e9:.2f}GB" if device == "cuda" else ""
            print(f"iter {step:5d} | loss {loss.item():.4f}{mem}", flush=True)

        if not args.no_eval and (step % args.eval_interval == 0 or step == args.iters - 1):
            with torch.no_grad():
                losses = []
                for _ in range(10):
                    vx, vy = impl.get_batch(val_data, args.block_size, args.batch_size, device)
                    with autocast:
                        _, vloss = model(vx, vy)
                    losses.append(vloss.item())
            print(f"  eval | val loss {sum(losses) / len(losses):.4f}", flush=True)

    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
