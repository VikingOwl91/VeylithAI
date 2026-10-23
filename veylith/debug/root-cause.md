# Root-cause analysis: ROCm memory access fault in veylith v2 sparse MoE

**Question:** is the root cause (a) our sparse MoE implementation, (b) the
PyTorch/ROCm integration, or (c) an underlying HIP/GPU kernel issue?

**Short answer:** (b)/(c). A GPU kernel on the sparse-MoE backward path
(bf16) writes out of bounds despite fully in-bounds indices. Our
implementation's math and indices are correct (audited line by line; the
dense-MoE rewrite of the same math is stable; a deliberate OOB-index
control reproduces the same fault signature). The fault is a
PyTorch-on-ROCm kernel bug, and it is **fixed in newer PyTorch**: the
version isolation study (`version-comparison.md`, this branch) shows the
fault reproduces on torch 2.14.1 with ROCm 7.2 AND 7.14, and is gone in
torch 2.15.0.dev20261005+rocm7.14 and 2.16.0.dev20261006+rocm7.14. The fix
window is PyTorch commits between 2026-08-24 and 2026-10-05 in the 2.15
dev cycle. The exact kernel is still not named (no compute-sanitizer/
rocgdb on this box), but the cross-version evidence makes the
classification firm. A deterministic reproducer and a draft upstream
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

## Working hypothesis

A bf16 kernel on the sparse-MoE backward path - most plausibly one of the
variable-size routing kernels (`index_select` backward scatter,
`index_add` backward gather, or the expert weight-grad GEMMs that only
run when expert grads are enabled) - writes slightly out of bounds past
its output tensor despite in-bounds indices. The write is silent when it
lands in mapped VA; it faults ("Page not present") when the tensor's
segment is the last in a mapped VA chunk. This explains every
observation: backward-necessity, bf16-specificity, sparse-MoE
specificity, layout/seed sensitivity, intermittency, the 2MB-aligned
fault addresses, and the delayed/varying manifestation point.

## Recommendation

**Application fix (already shipped):** veylith v2 uses dense-combined MoE
routing (no variable-size gather/scatter), which is stable. Keep it.
The sparse variant is preserved in git history for reference.

**Upstream:** this is a PyTorch-on-ROCm kernel bug, fixed in the 2.15 dev
cycle. The version isolation study (`version-comparison.md`) shows:
torch 2.14.1 faults with ROCm 7.2 AND ROCm 7.14 (so ROCm 7.14 does not
fix it), while torch 2.15.0.dev20261005+rocm7.14 and
2.16.0.dev20261006+rocm7.14 pass the deterministic reproducer 0 faults in
16 runs. The fix window is PyTorch commits between 2026-08-24 and
2026-10-05 (2.15 dev cycle). This makes the upstream report stronger: it
points the maintainers at a regression window and a deterministic
reproducer. The strongest candidate area is ATen's variable-size index
kernels on ROCm for bf16 on gfx1201 (`index_add`/`index_select`
backward). The draft report is in `issue-report.md` (NOT submitted, per
instructions).

## Remaining uncertainties

- The exact PyTorch commit/PR that fixed it: the window is PyTorch
  commits between 2026-08-24 and 2026-10-05 (2.15 dev cycle), narrowed by
  dated nightlies (testE faults on 20260824, testD passes on 20261005).
  Bisecting further would mean more dated-nightly installs.
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
