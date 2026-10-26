# Root-cause analysis: ROCm memory access fault in veylith v2 sparse MoE

> **Status note (2026-10-26).** This is the Phase 2/3 report. Its "fixed"
> statements and its sparse-MoE recommendation are scoped to the tested
> 8-expert/top-2 reproducer. A later 16-expert/top-4 run faulted on
> 2.16.0.dev20261006, so no general fix is established and adoption is not
> recommended. See `README.md` for the current status.

**Question:** is the root cause (a) our sparse MoE implementation, (b) the
PyTorch/ROCm integration, or (c) an underlying HIP/GPU kernel issue?

**Short answer:** (b)/(c). A GPU kernel on the sparse-MoE backward path
(bf16) writes out of bounds despite fully in-bounds indices. Our
implementation's math and indices are correct (audited line by line; the
dense-MoE rewrite of the same math is stable; a deliberate OOB-index
control reproduces the same fault signature). The fault is a
PyTorch-on-ROCm kernel issue that is **no longer reproducible in newer
PyTorch for the tested reproducer**. The version isolation study
(`version-comparison.md`, this branch) shows the fault reproduces on
torch 2.14.1 with ROCm 7.2 AND 7.14, and does not reproduce on
torch 2.15.0.dev20261005+rocm7.14 and
2.16.0.dev20261006+rocm7.14. The behavior-change window is PyTorch commits
between 2026-08-24 and 2026-10-05 in the 2.15 dev cycle. The exact kernel
of the 8-expert/top-2 reproducer is still not named (no compute-sanitizer/
rocgdb on this box), but the cross-version evidence makes the
classification firm. The 16-expert/top-4 run still faulted on
2.16.0.dev20261006. A deterministic reproducer and a draft upstream
issue are included.

## Confirmed observations

All measured, not assumed. Logs are in `logs/`, matrix in `matrix.md`.

1. **Deterministic reproducer exists.** `reproducer.py --no_optimizer
   --grad_filter experts` (frozen model, experts-only backward) faults
   8/8 at seed 1337 (iter ~26-49); the full loop faults 5/5 (iter ~0-24).
   The original training-loop fault is the same fault at ~17% per 250
   iters (1/6 measured), because GPU nondeterminism (split-K atomics in
   GEMMs) varies the trigger condition. Per-seed rates at fixed config
   vary 0-1/3.
2. **The fault is in the backward pass.** Forward-only
   (`--no_backward`) completes 3/3 over 100 iters.
3. **The fault is bf16-specific.** fp32 completes 5/5 seeds x 100 iters.
4. **The fault is sparse-MoE-specific.** The dense-combined MoE (same
   math, dense ops) completes; CPU completes (logic correct there).
5. **The fault is not an async race.** It persists under
   `CUDA_LAUNCH_BLOCKING=1` (fully serialized kernel launches).
6. **The fault is not the caching allocator.** It reproduces with
   `PYTORCH_CUDA_ALLOC_CONF=backend:native` (no caching allocator).
7. **The fault is not memory pressure.** GPU memory flat at 0.3-0.6GB
   (17GB card).
8. **The fault is not a GEMM shape bug.** Sweep of every forward-path
   GEMM shape (fixed + 24 subset sizes, fp32+bf16, fwd+bwd): clean.
9. **The fault is not any single MoE op in isolation.** Per-op repros at
   real shapes (N=16384, E=8, k=2, C=256, H=682), 300 iters each, fp32
   and bf16: topk, nonzero+index_select, index_add, scatter_add_,
   one_hot, bool index_put, bool advanced indexing, expert SwiGLU, and
   the full sparse-MoE layer (fixed input): all clean. A 3000-trial
   randomized MoE expert loop (varying routing, weight grads,
   in-bounds index_add/index_select backward): clean.
10. **The fault address is always a 2MB-aligned segment boundary.**
    Across ~12 faults, always `0x7f........000`. In the captured run
    (fault address `0x7f7f42a00000`), the address is exactly the end of
    a mapped segment whose last block is a single 16-20MB fp32 tensor.
    Signature of a write past the last allocation in a segment into
    unmapped VA.
11. **The boundary tensor is allocated by sparse-MoE index ops.**
    PyTorch's allocation history (`capture_memhist.py`) shows the
    fault-boundary block's allocating stacks involve `index_select` /
    `index_add`. (Caveat: the history's stack capture is partly garbled
    in this build, so this is corroborating, not conclusive.)
12. **The fault is layout-sensitive.** The backward bisect
    (`--grad_filter`, which does not change init/data/batches) shows the
    fault manifests with experts-only grads but not with
    experts+gate+shared grads (same expert kernels, different layout).
    Different seeds fault at different depths/iters (even 1 MoE layer,
    even iter ~1). Consistent with an OOB write whose manifestation
    depends on whether the overflowed tensor is the last allocation in
    a mapped VA chunk.
13. **Our code's indices are all in bounds.** Audited: sel from nonzero
    is unique, sorted, in [0, N); topk_idx in [0, E); token ids in
    [0, vocab); CE targets in [0, vocab); scatter counts target (E,).

## The micro-test false positive (important for honesty)

A combined index-op micro-test faulted deterministically at trial 0, at
the first `index_add`. Op-level instrumentation showed the fault there.
It looked like the root cause - until code review caught that **the
micro-test itself had an out-of-bounds index**: it used the (M, C) expert
input as the `index_add` base while `sel` holds indices up to N-1. The
real MoE accumulates into `shared(x_flat)` which is (N, C), so the real
code is in bounds. A corrected micro-test (full-size base) passes 3000
trials. So the micro-test fault was a bug in the micro-test, not the
real fault.

What this *did* establish, usefully: an OOB index in `index_add`
produces exactly this fault signature (HSAIL hardware exception at
unmapped VA), which (a) validates the OOB-write theory for the real
fault and (b) confirms the real code, whose indices are in bounds, must
be faulting inside a kernel rather than at the index level.

## Phase 3 (this branch): narrowed behavior-change window and sparse-MoE verdict

**Behavior-change window: PyTorch commits on 2026-10-01/02.** Date-based
binary search over the 40 nightly builds between 20260824 and 20261005
(each tested in a
disposable uv venv, 2 reproducer runs per build): 20260913, 20260924,
20260930 and 20261001 all fault 2/2; 20261002 passes 2/2. Boundary:
`2.15.0.dev20261001` faults, `2.15.0.dev20261002` passes. Anchors
re-verified first (testE 2/2 FAULT, testD 2/2 PASS, testC 2/2 PASS).

**Sparse-MoE correctness (Phase C, on the known-good testC env):** the
sparse implementation is mathematically equivalent to the dense-combined
reference. 20 output/gradient comparisons across (E,k) in {(8,2),(4,2),
(16,4),(8,8)}, N in {2048,16384}, bf16+fp32, seeds 0/1/2: outputs match
everywhere; bf16 gradients match; fp32 gradient checks flagged 8 cases,
but an fp64 ground-truth comparison (diag_grad.py) shows sparse and dense
are equidistant from the reference (rel ~1e-6, fp32 reduction noise; my
1e-5 rtol was too strict). Memory flat over 200 iters; no NaN/Inf.

**Sparse vs dense benchmark (Phase D, on testC, bf16, batch 64, block 256,
150 measured iters each after warmup, CUDA-event timing, medians):**

| metric | sparse | dense | delta |
|---|---|---|---|
| forward ms | 51.34 [49.13..54.28] | 47.09 [46.71..47.66] | sparse +9% (nonzero sync overhead) |
| backward ms | 68.23 [67.74..71.02] | 88.95 [88.26..89.47] | sparse -23% (weight-grad GEMMs run on routed subsets only, K~M vs K=16384) |
| full step ms | 124.10 [121.62..128.83] | 140.60 [139.37..141.40] | **sparse -12%** |
| tokens/sec | 132,027 | 116,527 | **sparse +13%** |
| peak GPU allocated | 4.70 GB | 8.02 GB | **sparse -41%** (dense saves every expert output for every token) |
| peak GPU reserved | 5.13 GB | 8.53 GB | sparse -40% |
| CPU RSS max | 2604 MB | 2075 MB | sparse +529 MB (index bookkeeping + nonzero syncs) |
| stability (3x50 iters) | +/-3% around median | +/-1% | both stable |

GPU utilization sampling failed this run (`--loop` is not a valid
rocm-smi flag); reported as not measured rather than estimated.

Interesting cross-check: on the OLD torch (2.14.1), sparse trained SLOWER
than dense (4.4-4.8 vs 5.8 it/s); on the newer torch (2.16.0.dev) the
order flips (sparse 8.1 vs dense 7.1 it/s) - consistent with the 2.15-cycle
change also improving the index kernels' performance.

**Recommendation status:** the sparse implementation is correct, and it is
stable in the runs recorded here on torch 2.15.0.dev20261005 and later; it
is UNSAFE on any torch 2.14.x build (faults 100% in the deterministic
mode). Do not adopt sparse MoE on the strength of these runs alone: a
16-expert/top-4 configuration still faulted on 2.16.0.dev20261006.
Re-verify the target configuration, not just the reproducer, on a stable
PyTorch release (>= 2.15 at the time of writing).

## Recommendation

**Application fix (already shipped):** veylith v2 uses dense-combined MoE
routing (no variable-size gather/scatter), which is stable. Keep it.
The sparse variant is preserved in git history for reference.

**Upstream:** this is a PyTorch-on-ROCm kernel issue that is no longer
reproducible for the tested configuration from the 2.15 dev cycle on. The
version isolation study (`version-comparison.md`) shows:
torch 2.14.1 faults with ROCm 7.2 AND ROCm 7.14 (so ROCm 7.14 does not
fix it), while torch 2.15.0.dev20261005+rocm7.14 and
2.16.0.dev20261006+rocm7.14 pass the deterministic reproducer 0 faults in
16 runs. The behavior-change window is PyTorch commits between 2026-08-24
and 2026-10-05 (2.15 dev cycle). This makes the upstream report stronger:
it points the maintainers at a regression window and a deterministic
reproducer. In the later 16-expert/top-4 run the kernel named at the fault
is a hipBLASLt GEMM (`Cijk_Ailk_Bjlk_...`), not an index kernel; for the
8-expert/top-2 reproducer the faulting kernel remains unnamed. The later
working hypothesis is a layout-dependent over-read in one hipBLASLt
solution, which is not hardware-verified. The draft report is in
`issue-report.md` (NOT submitted, per instructions).

## Remaining uncertainties

- The exact PyTorch commit/PR that changed the behavior: the window is
  PyTorch commits between 2026-08-24 and 2026-10-05 (2.15 dev cycle),
  narrowed by dated nightlies (testE faults on 20260824, testD passes on
  20261005). Bisecting further would mean more dated-nightly installs.
- The exact faulting kernel is not named: no compute-sanitizer/rocgdb on
  this box, rocprofv3 cannot finalize after a GPU fault (hangs with
  "kernels still active"), and PyTorch's memory-history stack capture is
  partly garbled in this build.
- Whether the bug is in PyTorch's ATen ROCm kernels or in a lower
  rocprim/rocBLAS kernel they call is undetermined; the PyTorch-side fix
  between 2.14.1 and 2.15.0.dev20261005 suggests a PyTorch-side change.
- PyTorch versions older than 2.14.1 untested (regression vs
  long-standing bug unknown).
- bf16-specificity confirmed at the model level (5 seeds); the
  dtype-dependent kernel path is unidentified.
