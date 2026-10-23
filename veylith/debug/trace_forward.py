#!/usr/bin/env python3
"""Op-level trace of the deterministic sparse-MoE GPU fault.

Replicates `reproducer.py --no_optimizer` exactly (same seed, data, model
init, batch order), then runs the faulting iteration's forward with
per-op prints and CUDA_LAUNCH_BLOCKING=1 (proven not to prevent the
fault), so the last printed op is the one whose kernel faults.

MoE and attention internals are traced via monkeypatched forwards that
are op-for-op identical to the recovered code; all other modules are
covered by forward hooks. Tracing starts at iter TRACE_FROM (default 45;
the fault occurs during iter 47's forward).

Usage:  TRACE_FROM=45 python3 trace_forward.py
"""

import math
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


def traced_moe_forward(self, x):
    """Recovered MoE forward (bfafa70), op for op, plus prints and syncs."""
    B, T, C = x.shape
    x_flat = x.reshape(B * T, C)
    print(f"    [moe] enter x_flat {tuple(x_flat.shape)} {x_flat.dtype}", flush=True)
    torch.cuda.synchronize()
    router_logits = self.gate(x_flat).float()
    print(f"    [moe] gate {tuple(router_logits.shape)}", flush=True)
    torch.cuda.synchronize()
    topk_weights, topk_idx = router_logits.topk(self.top_k, dim=-1)
    print(f"    [moe] topk {tuple(topk_idx.shape)}", flush=True)
    torch.cuda.synchronize()
    topk_weights = F.softmax(topk_weights, dim=-1)
    print("    [moe] softmax ok", flush=True)
    torch.cuda.synchronize()
    out = self.shared(x_flat)
    print(f"    [moe] shared {tuple(out.shape)} {out.dtype}", flush=True)
    torch.cuda.synchronize()
    for e, expert in enumerate(self.experts):
        w_e = ((topk_idx == e).float() * topk_weights).sum(-1)
        sel = (w_e > 0).nonzero(as_tuple=True)[0]
        print(f"    [moe] expert {e}: nonzero M={sel.numel()}", flush=True)
        torch.cuda.synchronize()
        if sel.numel() == 0:
            continue
        xs = x_flat.index_select(0, sel)
        print(f"    [moe] expert {e}: index_select {tuple(xs.shape)}", flush=True)
        torch.cuda.synchronize()
        expert_out = expert(xs)
        print(f"    [moe] expert {e}: fwd {tuple(expert_out.shape)}", flush=True)
        torch.cuda.synchronize()
        delta = (w_e[sel].unsqueeze(1) * expert_out).to(out.dtype)
        out = out.index_add(0, sel, delta)
        print(f"    [moe] expert {e}: index_add ok", flush=True)
        torch.cuda.synchronize()
    probs = F.softmax(router_logits, dim=-1)
    counts = torch.zeros(self.n_experts, device=x_flat.device)
    counts.scatter_add_(0, topk_idx.reshape(-1),
                        torch.ones(topk_idx.numel(), device=x_flat.device))
    routed_frac = counts / topk_idx.numel()
    mean_prob = probs.mean(0)
    aux = self.n_experts * (routed_frac * mean_prob).sum()
    print(f"    [moe] aux {float(aux):.4f}", flush=True)
    torch.cuda.synchronize()
    return out.view(B, T, C), aux


def traced_attn_forward(self, x):
    """Recovered attention forward (bfafa70), op for op, plus prints+syncs."""
    B, T, C = x.shape
    print(f"    [attn] enter {tuple(x.shape)} full={self.full}", flush=True)
    torch.cuda.synchronize()
    q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
    print("    [attn] qkv ok", flush=True)
    torch.cuda.synchronize()
    q = sparse_impl.apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
    k = sparse_impl.apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
    print("    [attn] rope ok", flush=True)
    torch.cuda.synchronize()
    k = k.repeat_interleave(self.n_rep, dim=1)
    v = v.repeat_interleave(self.n_rep, dim=1)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    print(f"    [attn] qk {tuple(att.shape)}", flush=True)
    torch.cuda.synchronize()
    i = torch.arange(T, device=x.device)[:, None]
    j = torch.arange(T, device=x.device)[None, :]
    allowed = j <= i
    if not self.full:
        allowed = allowed & (i - j < self.sliding_window)
    att = att.masked_fill(~allowed[None, None, :, :], float("-inf"))
    att = F.softmax(att, dim=-1)
    print("    [attn] softmax ok", flush=True)
    torch.cuda.synchronize()
    att = self.attn_drop(att)
    x2 = (att @ v).transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
    print("    [attn] av ok", flush=True)
    torch.cuda.synchronize()
    return self.resid_drop(self.o_proj(x2))


def main():
    torch.manual_seed(1337)
    device = "cuda"
    cfg = sparse_impl.Config(
        n_layer=6, n_head=8, n_kv_head=2, n_embd=256, block_size=256,
        sliding_window=128, full_attn_every=4, dropout=0.0,
        n_experts=8, n_experts_per_tok=2, first_k_dense=1, n_mtp=1,
        vocab_size=1000)
    model = sparse_impl.GPT(cfg).to(device)

    sparse_impl.MoE.forward = traced_moe_forward
    sparse_impl.CausalSelfAttention.forward = traced_attn_forward
    for name, mod in model.named_modules():
        if name and not isinstance(mod, (sparse_impl.MoE, sparse_impl.CausalSelfAttention)):
            def make_hook(name):
                def hook(module, inp, out):
                    if not TRACING["on"]:
                        return
                    shape = tuple(out.shape) if torch.is_tensor(out) else type(out).__name__
                    print(f"  hook {name}: {shape}", flush=True)
                return hook
            mod.register_forward_hook(make_hook(name))

    data = torch.randint(0, 1000, (200000,))
    train_data = data[: int(0.9 * len(data))]

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for step in range(60):
        x, y = sparse_impl.get_batch(train_data, 256, 64, device)
        print(f"=== iter {step} ===", flush=True)
        if step >= TRACE_FROM:
            TRACING["on"] = True
        with autocast:
            _, loss = model(x, y)
        loss.backward()
        print(f"=== iter {step} done loss={loss.item():.4f} ===", flush=True)
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
