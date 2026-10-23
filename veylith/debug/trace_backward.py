#!/usr/bin/env python3
"""Backward-pass op-level trace of the deterministic sparse-MoE fault.

The forward pass is clean (forward-only runs complete 3/3). This script
traces the backward: a traced GPT.forward prints after each loss term,
then full backward hooks on every named module print as each module's
backward completes. The last printed line before the fault localizes it.

Usage:  TRACE_FROM=45 CUDA_LAUNCH_BLOCKING=1 python3 trace_backward.py
"""

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v2"))

import sparse_moe_v2_train as sparse_impl

TRACE_FROM = int(os.environ.get("TRACE_FROM", "45"))
TRACING = {"on": False}


def traced_gpt_forward(self, idx, targets=None):
    """Recovered GPT.forward (bfafa70), op for op, plus prints and syncs."""
    B, T = idx.shape
    x = self.tok_emb(idx)
    aux_total = x.new_zeros(())
    for block in self.blocks:
        x, aux = block(x)
        aux_total = aux_total + aux
    x = self.ln_f(x)
    logits = self.head(x)
    print(f"  [fwd] head {tuple(logits.shape)} {logits.dtype}", flush=True)
    torch.cuda.synchronize()
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        print(f"  [fwd] main CE {float(loss):.4f}", flush=True)
        torch.cuda.synchronize()
        if self.cfg.n_mtp >= 1:
            mtp_logits = logits[:, :-1]
            mtp_targets = targets[:, 1:]
            mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, logits.size(-1)),
                                       mtp_targets.reshape(-1))
            print(f"  [fwd] mtp CE {float(mtp_loss):.4f}", flush=True)
            torch.cuda.synchronize()
            loss = loss + self.cfg.mtp_weight * mtp_loss
            print("  [fwd] mtp add ok", flush=True)
            torch.cuda.synchronize()
        loss = loss + self.cfg.moe_aux_weight * aux_total
        print("  [fwd] aux add ok", flush=True)
        torch.cuda.synchronize()
    print("  [fwd] done", flush=True)
    torch.cuda.synchronize()
    return logits, loss


def main():
    torch.manual_seed(1337)
    device = "cuda"
    cfg = sparse_impl.Config(
        n_layer=6, n_head=8, n_kv_head=2, n_embd=256, block_size=256,
        sliding_window=128, full_attn_every=4, dropout=0.0,
        n_experts=8, n_experts_per_tok=2, first_k_dense=1, n_mtp=1,
        vocab_size=1000)
    model = sparse_impl.GPT(cfg).to(device)
    sparse_impl.GPT.forward = traced_gpt_forward

    gf = os.environ.get("GRAD_FILTER", "all")
    if gf != "all":
        groups = set(gf.split(","))
        for _, p in model.named_parameters():
            p.requires_grad_(False)

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
        print(f"GRAD_FILTER={gf}: {sum(p.requires_grad for p in model.parameters())} "
              f"trainable params", flush=True)

    hooked = 0
    for name, mod in model.named_modules():
        if not name:
            continue

        def make_bhook(name):
            def bhook(module, grad_input, grad_output):
                if TRACING["on"]:
                    print(f"  [bwd] {name}", flush=True)
            return bhook

        try:
            mod.register_full_backward_hook(make_bhook(name))
            hooked += 1
        except Exception:
            pass
    print(f"backward hooks registered on {hooked} modules", flush=True)

    data = torch.randint(0, 1000, (200000,))
    train_data = data[: int(0.9 * len(data))]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    for step in range(60):
        x, y = sparse_impl.get_batch(train_data, 256, 64, device)
        if step >= TRACE_FROM:
            TRACING["on"] = True
            print(f"=== iter {step} ===", flush=True)
        with autocast:
            _, loss = model(x, y)
        if not TRACING["on"]:
            loss.backward()
            continue
        print("  [bwd] start", flush=True)
        torch.cuda.synchronize()
        loss.backward()
        print(f"=== iter {step} done ===", flush=True)
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
