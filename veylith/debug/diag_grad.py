#!/usr/bin/env python3
"""Diagnose the fp32 gradient mismatches from correctness.py.

For one failing case (E8 k2, N=2048, fp32, seed 0): compare sparse vs
dense-combined MoE gradients, and both against an fp64 ground-truth
reference of the same math. If sparse and dense are each ~equally close
to fp64 (differing from each other only at rounding scale), the
implementations are mathematically equivalent and the mismatch is fp32
summation-order noise. If one form is systematically farther from fp64,
there is a structural difference.
"""

import sys
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/../v2")
import sparse_moe_v2_train as sparse_impl
import train as dense_impl

device = "cuda"
N, C, E, k = 2048, 256, 8, 2


def build(dtype):
    class Cfg:
        pass
    cs = Cfg()
    cs.n_embd, cs.n_experts, cs.n_experts_per_tok = C, E, k
    cs.mlp_ratio, cs.dropout = 8 / 3, 0.0
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
    sp.to(device); dn.to(device)
    if dtype == torch.float64:
        sp = sp.to(torch.float64)
        dn = dn.to(torch.float64)
    for m in (sp, dn):
        for p in m.parameters():
            p.requires_grad_(True)
    return sp, dn


torch.manual_seed(0)
B, T = 8, N // 8
x64 = torch.randn(B, T, C, device=device, dtype=torch.float64, requires_grad=True)
sp64, dn64 = build(torch.float64)
out64, _ = sp64(x64)
g64 = torch.randn_like(out64)

# fp32 runs (same seed -> same weights? no: build() re-inits. Use fp64 weights cast down)
x32 = x64.detach().to(torch.float32).requires_grad_(True)
sp32, dn32 = build(torch.float32)
with torch.no_grad():
    sp32.gate.weight.copy_(sp64.gate.weight.to(torch.float32))
    sp32.shared.w1.weight.copy_(sp64.shared.w1.weight.to(torch.float32))
    sp32.shared.w2.weight.copy_(sp64.shared.w2.weight.to(torch.float32))
    sp32.shared.w3.weight.copy_(sp64.shared.w3.weight.to(torch.float32))
    for se, s64 in zip(sp32.experts, sp64.experts):
        se.w1.weight.copy_(s64.w1.weight.to(torch.float32))
        se.w2.weight.copy_(s64.w2.weight.to(torch.float32))
        se.w3.weight.copy_(s64.w3.weight.to(torch.float32))
    dn32.gate.weight.copy_(sp32.gate.weight)
    dn32.shared.w1.weight.copy_(sp32.shared.w1.weight)
    dn32.shared.w2.weight.copy_(sp32.shared.w2.weight)
    dn32.shared.w3.weight.copy_(sp32.shared.w3.weight)
    for de, se in zip(dn32.experts, sp32.experts):
        de.w1.weight.copy_(se.w1.weight)
        de.w2.weight.copy_(se.w2.weight)
        de.w3.weight.copy_(se.w3.weight)

g32 = g64.to(torch.float32)
out_sp, _ = sp32(x32)
out_dn, _ = dn32(x32)
(out_sp * g32).sum().backward()
(out_dn * g32).sum().backward()

# fp64 reference gradients (dense math, exact)
sp64.zero_grad()
(out64 * g64).sum().backward()

def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()

print("fp32 case: E8 k2 N2048 seed0")
print(f"out  sparse-vs-dense maxdiff: {(out_sp - out_dn).abs().max().item():.3e}")
print(f"out  sparse-vs-fp64 maxdiff: {(out_sp.double() - out64).abs().max().item():.3e}")
print(f"out  dense-vs-fp64  maxdiff: {(out_dn.double() - out64).abs().max().item():.3e}")
for name, sp_p, dn_p, ref_p in [
        ("gate.weight", sp32.gate.weight, dn32.gate.weight, sp64.gate.weight),
        ("expert0.w1", sp32.experts[0].w1.weight, dn32.experts[0].w1.weight, sp64.experts[0].w1.weight)]:
    gs, gd, gr = sp_p.grad, dn_p.grad, ref_p.grad
    print(f"grad {name}: sparse-vs-dense maxdiff {(gs - gd).abs().max().item():.3e} "
          f"(rel {rel(gs, gd):.2e}); |grad|max={gs.abs().max().item():.3e}")
    print(f"grad {name}: sparse-vs-fp64 rel {rel(gs.double(), gr):.2e} | "
          f"dense-vs-fp64 rel {rel(gd.double(), gr):.2e}")
