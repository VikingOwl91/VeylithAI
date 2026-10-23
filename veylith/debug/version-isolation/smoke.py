#!/usr/bin/env python3
"""Compatibility smoke test for a PyTorch/ROCm environment.

Checks: torch.cuda.is_available(), GPU arch (expect gfx1201 / cc 12.0),
bf16 matmul, bf16 forward+backward, fp32 gradient correctness, GPU sync,
and which ROCm/HIP libraries are loaded at runtime (to detect
system-library conflicts with /opt/rocm).

Exit code 0 = core checks passed (library loading is informational).
"""

import sys

import torch

print(f"python: {sys.version.split()[0]}")
print(f"torch: {torch.__version__} | hip: {torch.version.hip} | cuda attr: {torch.version.cuda}")

ok = True

if not torch.cuda.is_available():
    print("FAIL: torch.cuda.is_available() is False")
    sys.exit(1)
print("cuda available: True")

p = torch.cuda.get_device_properties(0)
print(f"device: {p.name} | cc: {p.major}{p.minor} | vram GB: {round(p.total_memory / 1e9, 1)}")
if (p.major, p.minor) != (12, 0):
    print(f"WARN: expected gfx1201 (cc 12.0), got cc {p.major}{p.minor}")
    ok = False

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
finite = bool(torch.isfinite(y).all())
print(f"bf16 matmul: finite={finite}")
ok = ok and finite

w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
xx = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
out = (xx @ w).sum()
out.backward()
torch.cuda.synchronize()
grad_ok = w.grad is not None and bool(torch.isfinite(w.grad).all())
print(f"bf16 fwd+bwd: grad finite={grad_ok}")
ok = ok and grad_ok

w32 = torch.randn(64, 64, device="cuda", requires_grad=True)
xx32 = torch.randn(32, 64, device="cuda")
(xx32 @ w32).sum().backward()
expected = xx32.sum(0).unsqueeze(1).expand(64, 64)
grad_correct = torch.allclose(w32.grad, expected, atol=1e-4)
print(f"fp32 grad correct: {grad_correct}")
ok = ok and grad_correct

torch.cuda.synchronize()
print("sync: ok")

maps = open("/proc/self/maps").read()
paths = sorted({line.split()[-1] for line in maps.splitlines()
                if any(k in line.lower() for k in ("roc", "hip", "amd"))
                and line.split()[-1].startswith("/")})
n_system = sum(1 for p in paths if p.startswith("/opt/rocm"))
print(f"loaded rocm/hip/amd libs: {len(paths)} "
      f"(system /opt/rocm: {n_system}, elsewhere: {len(paths) - n_system})")
for p in paths:
    print(f"  {p}")
if n_system > 0:
    print("WARN: system ROCm libraries are loaded; ROCm-version isolation may be incomplete")

print("SMOKE RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
