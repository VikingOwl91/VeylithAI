#!/usr/bin/env python3
"""Capture the CUDA allocation history around the fault.

Runs the experts-only reproducer (the minimal faulting configuration)
with torch.cuda.memory history recording, dumping the full history
(including allocating-op stack traces) to /tmp/opencode/memhist.pickle
after every iteration. On fault, the last dump plus the fault address
from the log identify the allocation (and its op) just below the fault.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v2"))

import sparse_moe_v2_train as sparse_impl

SNAP = "/tmp/opencode/memhist.pickle"


def main():
    torch.manual_seed(1337)
    device = "cuda"
    cfg = sparse_impl.Config(
        n_layer=6, n_head=8, n_kv_head=2, n_embd=256, block_size=256,
        sliding_window=128, full_attn_every=4, dropout=0.0,
        n_experts=8, n_experts_per_tok=2, first_k_dense=1, n_mtp=1,
        vocab_size=1000)
    model = sparse_impl.GPT(cfg).to(device)
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if ".mlp.experts." in name:
            p.requires_grad_(True)

    data = torch.randint(0, 1000, (200000,))
    train_data = data[: int(0.9 * len(data))]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    torch.cuda.memory._record_memory_history()
    for step in range(100):
        x, y = sparse_impl.get_batch(train_data, 256, 64, device)
        with autocast:
            _, loss = model(x, y)
        loss.backward()
        torch.cuda.memory._dump_snapshot(SNAP)
        print(f"iter {step} ok, allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB",
              flush=True)
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
