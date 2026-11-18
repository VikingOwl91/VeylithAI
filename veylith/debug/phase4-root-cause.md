# Phase 4: root-cause archaeology - the upstream fix

**Most likely root cause (high confidence):** the ROCm BLAS **workspace
caching path** in PyTorch - one cached workspace per (hipBLAS handle,
stream) pair, the ROCm default before PyTorch 2.15 - can leave a rocBLAS
handle bound to a workspace buffer that has been freed and handed back to
the caching allocator. Subsequent GEMMs on that handle then write into
freed/reused memory: an OOB write that corrupts neighbors silently and
faults ("Page not present") when it crosses into unmapped VA. Fixed by
PR #196920, which switched ROCm to eager per-operation workspaces with
explicit bind/unbind.

## Phase A: exact source revisions

Nightly wheels embed their build commit in `torch.version.git_version`;
the deleted bisect venvs' commits were recovered from the uv cache
(`archive-v0/*/torch/version.py`). Each nightly's git_version is a
"nightly release" commit on the 2.15 release branch whose message
references the main-branch commit the build was cut from:

| build | git_version (release branch) | main commit (from release msg) | fault |
|---|---|---|---|
| 2.15.0.dev20260824 (testE) | c0577575187a | - | FAULT 4/4 |
| 2.15.0.dev20260913 | 13376c2070a7 | - | FAULT 2/2 |
| 2.15.0.dev20260924 | 095e4e3aa2a1 | - | FAULT 2/2 |
| 2.15.0.dev20260930 | 89ac0527b15d | - | FAULT 2/2 |
| 2.15.0.dev20261001 | 03a234b03f20 | cut from main `01949e998f4e` | FAULT 2/2 |
| 2.15.0.dev20261002 | 0c54d0fac6d3 | cut from main `89492c102edb` | PASS 2/2 |
| 2.15.0.dev20261005 (testD) | 7bf52b3a9782 | - | PASS 4/4 |
| 2.16.0.dev20261006 (testC) | 1a66f68165eb | - | PASS 12/12 |

## The red herring (documented so nobody chases it again)

`compare/03a234b0...0c54d0fa` (the two boundary builds' git_versions)
shows exactly ONE commit: `89492c102` = "[c10d][nccl2] Wait for
nonblocking split/shrink child initialization (#199263)" - a distributed/
NCCL fix, irrelevant to a single-GPU MoE kernel fault. Explanation: the
git_version records the release-branch version-bump commit, and that
commit's message references the main-branch tip the build was cut from
(`89492c102` itself for 20261002; `01949e998f4e` for 20261001). The real
binary difference is the 82 main-branch commits between those two cuts.

## Ranked candidate changes (from the 82 main commits, Oct 1 -> Oct 2)

1. **PR #196920 (d3aeaea8e) "[ROCm] Always eagerly allocate hipBLAS(Lt)
   workspaces"** - by far the strongest. It rewrote ROCm BLAS
   handle/workspace lifecycle: before it, ROCm cached one workspace per
   (handle, stream) by default; after it, ATen allocates a workspace per
   operation from the caching allocator and explicitly unbinds it
   ("rocblas_set_stream leaves the workspace binding in place, so restore
   by unbinding explicitly"). Its own tests document our failure mode:
   "A workspace bound with rocblas_set_workspace stays bound, even after
   the buffer is freed", and "GEMMs from two streams that share one
   workspace corrupt each other's results". It also flipped the ROCm
   default from cached to eager workspaces ("was the default before
   PyTorch 2.15").
2. PR #198905 (e53195338) HSA_CO_DMACOPY_SIZE - a CI-only workaround for
   a ROCm code-object blit-kernel bug; refuted as our cause by EXP3
   (below) and because our fault signature is a data-write VM fault at
   segment boundaries, not a code-load failure.
3. f48c3d507 "Stop bundling ROCm shared libraries into Linux wheels" -
   packaging only; refuted because both the 7.2 and 7.14 venv runtimes
   fault (the lib source does not matter).
4. Everything else in the window: inductor, CI, MPS, distributed - not on
   our code path.

## Phase B: causality experiments (env-var controlled, no rebuilds)

| experiment | build | setting | result |
|---|---|---|---|
| EXP1 | testE (failing) | TORCH_CUBLAS_WORKSPACE_CACHE=0 | FAULT 3/3 |
| EXP2 | testD (passing) | TORCH_CUBLAS_WORKSPACE_CACHE=1 | **FAULT 3/3** |
| EXP3 | testE (failing) | HSA_CO_DMACOPY_SIZE=1073741824 | FAULT 3/3 |

**EXP2 is the smoking gun in reverse**: re-enabling the old cached-
workspace path on the fixed build reintroduces the fault deterministically.
The bug lives in the workspace-caching path; #196920's fix works by
making eager per-op workspaces (with explicit unbind) the default.

**EXP1 refines the story**: the failing build still faults with caching
DISABLED, because its eager path was also broken - it relied on
`cublasSetStream` resetting the workspace (true for cuBLAS, NOT for
rocBLAS), so freed workspaces stayed bound. #196920 fixed the eager path
(explicit unbind) and left the caching path as an opt-in that still
carries the bug (EXP2).

**EXP3** refutes the blit-kernel/code-object hypothesis.

## Why the old implementation (our sparse MoE) failed, and dense did not

Sparse MoE routing makes GEMM shapes vary per iteration (per-expert
subsets M change with the routing). Variable-shape GEMMs change the
workspace size requirement, forcing the cached workspace to be replaced
(evicted/reallocated) - and after replacement, the handle's binding to
the freed buffer persists. The next GEMM writes into freed-and-reused
memory: silent corruption of other tensors, and a "Page not present" VM
fault whenever the stale write crosses into an unmapped VA segment
(2MB-aligned addresses). Dense-combined MoE has fixed GEMM shapes: the
workspace is allocated once and never replaced, so the dangling binding
never arises. bf16-only: on gfx1201, bf16 GEMMs run on rocBLAS/hipBLASLt
kernels that consume workspaces (split-K / Tensile); fp32 GEMMs on these
shapes do not, so fp32 never allocates the poisoned buffer. Backward
necessity: the backward's weight-grad GEMMs are where the largest/
most variable workspace demands occur; CUDA_LAUNCH_BLOCKING does not
prevent it because the dangling binding is a lifecycle bug, not a
scheduling race - a corrupted/unmapped access faults whenever the kernel
runs, serialized or not.

## Confidence

- **Confirmed (measured):** the build-boundary dates; the one-commit
  red herring and the real 82-commit main window; EXP1/EXP2/EXP3
  results; the fault's dependence on the workspace-caching env var on
  the fixed build.
- **Strongly supported hypothesis:** PR #196920 is the fixing change
  (temporal + mechanism + env-var causality all align). Not proven to
  commit level: that would require reverting it on a fresh build or
  cherry-picking it backwards, i.e. a PyTorch rebuild, which this
  investigation deliberately avoided (time/disk cost, per instructions).
- **Unresolved:** the exact kernel instruction sequence that writes OOB;
  whether the opt-in `TORCH_CUBLAS_WORKSPACE_CACHE=1` path on current
  builds is broken for everyone or only for variable-shape workloads
  (our EXP2 shows it is broken for ours).
