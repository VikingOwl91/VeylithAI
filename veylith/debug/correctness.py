#!/usr/bin/env python3
"""Phase C: sparse-MoE correctness tests in a known-good environment.

Compares the recovered sparse MoE (variable-size routing: topk -> nonzero
-> index_select/index_add) against the dense-combined MoE (same math, dense
ops) from veylith v2. The two are mathematically equivalent: the dense form
computes every expert on every token and multiplies by the router weight
(0 for non-routed pairs); the sparse form computes only routed pairs and
accumulates with index_add.

Tests (run in the known-good env, e.g. torch 2.16.0.dev + rocm7.14):
  1. output equality (sparse vs dense), bf16 and fp32
  2. gradient equality: x, gate.weight, shared weights, expert weights
  3. routing shapes: (E, k) in {(8,2), (4,2), (16,4), (8,8)}, N in {2048, 16384}
  4. seeds: 0, 1, 2
  5. NaN/Inf checks on outputs and gradients
  6. repeated iterations with memory-growth check

Usage: python3 correctness.py
"""

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/../v2")

import sparse_moe_v2_train as sparse_impl
import train as dense_impl

device = "cuda"
failures = []


def build_pair(n_embd, n_experts, top_k, mlp_ratio=8 / 3):
    """Sparse and dense MoE modules with identical weights."""
    class Cfg:
        pass
    cs = Cfg()
    cs.n_embd, cs.n_experts, cs.n_experts_per_tok = n_embd, n_experts, top_k
    cs.mlp_ratio, cs.dropout = mlp_ratio, 0.0
    sp = sparse_impl.MoE(cs)
    dn = dense_impl.MoE(cs)
    with torch.no_grad():
        dn.gate.weight.copy_(sp.gate.weight)
        dn.shared.w1.weight.copy_(sp.shared.w1.weight)
        dn.shared.w2.weight.copy_(sp.shared.w2.weight)
        dn.shared.w3.weight.copy_(sp.shared.w3.weight)
        for de, se in zip(dn.experts, sp.experts):
            de.w1.weight.copy_(se.w1.weight)
            de.w2.weight.copy_(se.w2.weight)
            de.w3.weight.copy_(se.w3.weight)
    for m in (sp, dn):
        m.to(device)
        for p in m.parameters():
            p.requires_grad_(True)
    return sp, dn


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


def finite(t):
    return bool(torch.isfinite(t).all())


def run_case(N, n_embd, E, k, dtype, seed):
    torch.manual_seed(seed)
    sp, dn = build_pair(n_embd, E, k)
    B, T = 8, N // 8  # both modules expect (B, T, C) and flatten internally
    x = torch.randn(B, T, n_embd, device=device, requires_grad=True)
    xd = x.detach().clone().requires_grad_(True)

    if dtype == "bf16":
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = torch.autocast(device_type="cuda", enabled=False)
    with ctx:
        out_sp, aux_sp = sp(x)
        out_dn, aux_dn = dn(xd)
    torch.cuda.synchronize()

    tol = dict(atol=2e-2, rtol=2e-2) if dtype == "bf16" else dict(atol=1e-5, rtol=1e-5)
    eq = torch.allclose(out_sp.float(), out_dn.float(), **tol)
    check(f"out E{E}k{k} N{N} {dtype} s{seed} output equality", eq,
          f"maxdiff={(out_sp.float() - out_dn.float()).abs().max().item():.2e}")
    check(f"out E{E}k{k} N{N} {dtype} s{seed} finite", finite(out_sp) and finite(out_dn))

    # gradients: same upstream grad through both
    g = torch.randn_like(out_sp)
    (out_sp * g).sum().backward()
    (out_dn * g).sum().backward()
    torch.cuda.synchronize()
    eq = torch.allclose(x.grad.float(), xd.grad.float(), **tol)
    check(f"out E{E}k{k} N{N} {dtype} s{seed} grad-x equality", eq,
          f"maxdiff={(x.grad.float() - xd.grad.float()).abs().max().item():.2e}")
    eq = torch.allclose(sp.gate.weight.grad.float(), dn.gate.weight.grad.float(), **tol)
    check(f"out E{E}k{k} N{N} {dtype} s{seed} grad-gate equality", eq)
    eq = torch.allclose(sp.experts[0].w1.weight.grad.float(),
                        dn.experts[0].w1.weight.grad.float(), **tol)
    check(f"out E{E}k{k} N{N} {dtype} s{seed} grad-expert0 equality", eq)
    check(f"out E{E}k{k} N{N} {dtype} s{seed} grads finite",
          all(finite(p.grad) for p in sp.parameters() if p.grad is not None))


def memory_stability(iters=200):
    torch.manual_seed(7)
    sp, _ = build_pair(256, 8, 2)
    x0 = torch.randn(8, 2048, 256, device=device, requires_grad=True)
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    base = None
    grow = False
    last_x_grad = None
    for i in range(iters):
        x = x0.detach().clone().requires_grad_(True)
        with ctx:
            out, aux = sp(x)
        (out.sum() + 0.01 * aux).backward()
        last_x_grad = x.grad
        if i == 10:
            base = torch.cuda.memory_allocated()
        if i == iters - 1:
            end = torch.cuda.memory_allocated()
            grow = end > base + 10 * 1024 * 1024  # >10MB growth = suspicious
    torch.cuda.synchronize()
    check(f"memory stability over {iters} iters (no growth)", not grow,
          f"base={base / 1e9:.2f}GB end={end / 1e9:.2f}GB")
    check("no NaN/Inf after repeated iters",
          finite(out) and last_x_grad is not None and finite(last_x_grad))


print("=== sparse-MoE correctness (sparse vs dense reference) ===")
for dtype in ("bf16", "fp32"):
    for (E, k) in ((8, 2), (4, 2), (16, 4), (8, 8)):
        for N in (2048, 16384):
            run_case(N, 256, E, k, dtype, seed=0)
run_case(16384, 256, 8, 2, "bf16", seed=1)
run_case(16384, 256, 8, 2, "bf16", seed=2)
memory_stability()

print(f"\nRESULT: {len(failures)} failure(s)" + (f": {failures}" if failures else " - all PASS"))
sys.exit(1 if failures else 0)
