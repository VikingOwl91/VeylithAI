# DRAFT upstream issue report (NOT SUBMITTED)

**Status:** draft only. Do not submit without the user's approval.
**Scope (2026-10-26):** the reproduction below is the 8-expert/top-2
configuration. A later 16-expert/top-4 run still faulted on
2.16.0.dev20261006 (exit 134, same signature), so the passing builds in
the table are configuration-dependent and no general fix is claimed. The
responsible change is not identified.

**Target:** PyTorch (ROCm backend) - https://github.com/pytorch/pytorch/issues
**Title:** [ROCm] GPU memory fault in sparse-MoE training backward pass (bf16, gfx1201)

## Summary

A minimal, deterministic reproducer triggers a GPU memory access fault
("Page not present") on ROCm 7.2 / gfx1201 / PyTorch 2.14.1 (ROCm build)
when running forward+backward of a sparse Mixture-of-Experts layer that
uses variable-size routing (`topk` -> `nonzero` -> `index_select` /
`index_add`) under bf16 autocast. The fault is in the backward pass, is
bf16-specific (fp32 clean across seeds), and its manifestation is
layout-sensitive (intermittent in the original training configuration).
All indices in the reproducer are in bounds; an equivalent dense
implementation of the same math is stable. We suspect an OOB write in a
ROCm kernel on this path.

## Environment

- GPU: AMD Radeon RX 9070 XT (gfx1201, Navi 48), 17.1 GB
- ROCm userspace: 7.2.4 (`/opt/rocm/.info/version`); HIP 7.2.53211 (PyTorch build)
- PyTorch: 2.14.1 (ROCm build; `torch.version.hip` = 7.2.53211)
- Python: 3.14.8
- OS: Arch Linux, kernel 7.2.11-zen1-1.1-zen
- No `HIP*`/`HSA*`/`CUDA*` env vars set

## Reproduction

Self-contained reproducer (no dataset needed; random data; only PyTorch):

```bash
python reproducer.py --no_optimizer --grad_filter experts --iters 100
```

(`reproducer.py` is attached. It reconstructs the failing model code and
runs a frozen-model training loop; the fault is deterministic in this
mode. With `--grad_filter all` it matches the original training loop and
faults at ~10-30% per 100 iters. Full logs in the investigation repo.)

Expected: training loop completes.
Actual: process aborts with, e.g.:

```
Memory access fault by GPU node-1 (Agent handle: ...) on address 0x7f........000.
Reason: Page not present or supervisor privilege.
```

The fault address is consistently a 2MB-aligned segment boundary (the end
of a mapped segment whose last block is a 16-20MB fp32 tensor).

## What we ruled out (with evidence)

- Not an async race: persists under `CUDA_LAUNCH_BLOCKING=1`.
- Not the caching allocator: reproduces with
  `PYTORCH_CUDA_ALLOC_CONF=backend:native`.
- Not OOM/leak: memory flat at 0.3-0.6GB.
- Not GEMM shapes: full sweep of forward-path GEMM shapes (fixed and 24
  subset sizes, fp32+bf16, fwd+bwd) clean.
- Not single ops in isolation: per-op repros (topk, nonzero+index_select,
  index_add, scatter_add_, one_hot, bool index_put, bool gather, expert
  SwiGLU, full sparse-MoE layer) clean at real shapes, 300 iters each,
  fp32+bf16; a 3000-trial randomized MoE expert loop clean.
- Not our indices: all index sets audited in bounds (sel unique/sorted in
  [0, N); topk_idx in [0, E); token ids/targets in [0, vocab)).
- Not the math: a dense-combined MoE (same computation, dense ops) is
  stable, and the same sparse code runs cleanly on CPU.
- Not bf16-independent: fp32 clean across 5 seeds.

## Additional observations

- The fault needs the backward pass (forward-only is clean 3/3).
- In a backward bisect (freezing parameter groups, which does not change
  init/data/batches), the fault manifests with experts-only gradients;
  adding gate+shared grads shifts the layout and it does not manifest
  (same expert kernels) - consistent with a layout-sensitive OOB write.
- An OOB index in `index_add` (control test) reproduces the same fault
  signature, confirming the fault class.
- In a later 16-expert/top-4 run on 2.16.0.dev20261006, rocprof named a
  hipBLASLt GEMM (`Cijk_Ailk_Bjlk_...`, solution 107018) at the fault. That
  configuration still aborts on that build. The kernel is not confirmed as
  the root cause.
- rocprofv3 is available but cannot finalize after a GPU fault (hangs
  with "kernels still active"); compute-sanitizer and rocgdb are not
  installed on the repro machine, so we could not capture the exact
  faulting kernel. We would appreciate help identifying it.

## Phase 3 evidence (cross-version + narrowed window)

| torch | ROCm runtime | result |
|---|---|---|
| 2.14.1 (Arch pkg) | 7.2.4 (system) | FAULTS 4/4 |
| 2.14.1+rocm7.2 (pip) | 7.2 (venv rocm-sdk) | FAULTS 4/4 |
| 2.14.1+rocm7.14 (pip) | 7.14 / HIP 7.14.60850 (venv) | FAULTS 4/4 |
| 2.15.0.dev20260824+rocm7.14 | 7.14 (venv) | FAULTS 4/4 |
| 2.15.0.dev20260913 / 0924 / 0930 / 1001 | 7.14 (venv) | FAULTS 2/2 each (date bisect) |
| 2.15.0.dev20261002+rocm7.14 | 7.14 (venv) | PASSES 2/2 |
| 2.15.0.dev20261005+rocm7.14 | 7.14 (venv) | PASSES 4/4 (+2/2 re-verified) |
| 2.16.0.dev20261006+rocm7.14 | 7.14 (venv) | PASSES 12/12 |

So ROCm 7.14 does NOT fix it; the behavior changes within the 2.15 dev
cycle, narrowed by date bisect to **2026-10-01/02**. The responsible
commit is not identified. The 2.14 stable line is affected on both ROCm
7.2 and 7.14. Sparse-MoE correctness on 2.16.0.dev20261006 is verified
against the dense reference (see root-cause.md Phase 3 section).

Later evidence (16 experts, top-4, batch 64, block 256): the same
signature still aborts on 2.16.0.dev20261006 (exit 134, "Page not
present", 2 MiB-aligned address). The passing builds above therefore do
not mean the fault is gone in general.

## Sparse vs dense benchmark (Phase D, torch 2.16.0.dev20261006+rocm7.14)

Same shapes (batch 64 x 256, 25.60M params), bf16, 3x50 measured iters
after warmup, CUDA events, medians:

| metric | sparse | dense |
|---|---|---|
| forward ms | 51.34 | 47.09 |
| backward ms | 68.23 | 88.95 |
| full step ms | 124.10 | 140.60 |
| tokens/sec | 132,027 | 116,527 |
| peak GPU allocated | 4.70 GB | 8.02 GB |

Sparse is 12% faster per training step and uses 41% less peak memory on
2.16.0.dev20261006.

## Sparse-MoE correctness (Phase C summary)

Sparse output and gradients match the dense reference: 20 comparisons
across (E,k) in {(8,2),(4,2),(16,4),(8,8)}, N in {2048,16384}, bf16+fp32,
seeds 0/1/2. fp32 gradient deltas are ~1e-6 relative and both forms are
equidistant from an fp64 ground truth (summation-order noise). Memory
flat over 200 iters; no NaN/Inf.

## Request

Could you help identify the change landed in the 2.15 dev cycle on
2026-10-01/02 that changes this behavior (so we know whether it is a
deliberate
fix or a side effect; note it also made the sparse path FASTER than the
dense-equivalent), whether it is relevant to backport to the 2.14
line, why a 16-expert/top-4 configuration still faults on 2.16.0.dev20261006,
and the faulting kernel on this path (variable-size
`index_select`/`index_add` backward and/or expert weight-grad GEMMs under
bf16 autocast, gfx1201, ROCm 7.2/7.14)? Happy to provide more traces or
test patches.
