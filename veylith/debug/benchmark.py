#!/usr/bin/env python3
"""Phase D: sparse vs dense MoE benchmark on the RX 9070 XT.

Measures forward latency, backward latency, full training-step latency,
tokens/sec, peak GPU memory (allocated/reserved), and CPU RSS, for the
recovered sparse MoE model and the current dense-combined MoE model,
under identical conditions (same shapes, dtype, optimizer, workload).

Timing uses CUDA events with a warmup phase; steady-state measurement
excludes initialization. Reports median and spread over 3 blocks of 50
iterations each.

Usage: python3 benchmark.py --impl sparse|dense
"""

import argparse
import contextlib
import resource
import time

import torch

import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/../v2")

import sparse_moe_v2_train as sparse_impl
import train as dense_impl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["sparse", "dense"], required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--n_experts_per_tok", type=int, default=2)
    p.add_argument("--n_kv_head", type=int, default=2)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()

    impl = sparse_impl if args.impl == "sparse" else dense_impl
    torch.manual_seed(1337)
    device = "cuda"

    cfg = impl.Config(n_layer=args.n_layer, n_head=8, n_kv_head=args.n_kv_head,
                      n_embd=args.n_embd, block_size=args.block_size,
                      sliding_window=128, full_attn_every=4, dropout=0.0,
                      n_experts=args.n_experts, n_experts_per_tok=args.n_experts_per_tok,
                      first_k_dense=1, n_mtp=1, vocab_size=1000)
    model = impl.GPT(cfg).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    data = torch.randint(0, 1000, (200000,))
    train_data = data[: int(0.9 * len(data))]

    if args.dtype == "bf16":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    torch.cuda.reset_peak_memory_stats()

    # warmup (not measured)
    for _ in range(args.warmup):
        x, y = impl.get_batch(train_data, args.block_size, args.batch_size, device)
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()

    fwd_ms, bwd_ms, step_ms, toks = [], [], [], []
    for b in range(args.blocks):
        for _ in range(args.iters):
            x, y = impl.get_batch(train_data, args.block_size, args.batch_size, device)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)

            e0.record()
            with autocast:
                _, loss = model(x, y)
            e1.record()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            e2.record()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            e3.record()
            torch.cuda.synchronize()
            fwd_ms.append(e0.elapsed_time(e1))
            bwd_ms.append(e1.elapsed_time(e2))
            step_ms.append(e0.elapsed_time(e3))
        toks.append(args.batch_size * args.block_size / (sorted(step_ms)[len(step_ms) // 2] / 1000))

    def stats(v):
        s = sorted(v)
        med = s[len(s) // 2]
        return med, s[0], s[-1]

    fm, flo, fhi = stats(fwd_ms)
    bm, blo, bhi = stats(bwd_ms)
    sm, slo, shi = stats(step_ms)
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"impl={args.impl} dtype={args.dtype} params={n_params / 1e6:.2f}M "
          f"batch={args.batch_size} block={args.block_size}")
    print(f"forward  ms: median={fm:.2f} [{flo:.2f}..{fhi:.2f}]")
    print(f"backward ms: median={bm:.2f} [{blo:.2f}..{bhi:.2f}]")
    print(f"full step ms: median={sm:.2f} [{slo:.2f}..{shi:.2f}]")
    print(f"tokens/sec (median step): {args.batch_size * args.block_size / (sm / 1000):,.0f}")
    print(f"peak GPU allocated: {peak_alloc:.2f} GB | reserved: {peak_reserved:.2f} GB")
    print(f"CPU RSS max: {rss1:.0f} MB (start {rss0:.0f} MB)")
    print(f"blocks measured: {args.blocks} x {args.iters} iters")


if __name__ == "__main__":
    main()
