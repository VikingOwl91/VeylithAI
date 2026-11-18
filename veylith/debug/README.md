# Debug: ROCm memory access fault in veylith v2 sparse MoE

The Python commands in this historical guide run on `archive/python-research`. [Browse the archived directory](https://github.com/VikingOwl91/VeylithAI/tree/archive/python-research/veylith/debug)

Investigation of the GPU "Memory access fault" (page not present) that
killed sparse-MoE training runs of veylith v2. The working dense-MoE
v2 and all of v1 are preserved on `main`; this directory and the
`debug/rocm-*` branches are the investigation sandbox.

**Status (2026-10-26): no general fix confirmed.** The evidence separates
into four tiers.

**1. Observed failing and passing runs.** The deterministic reproducer
(defaults: 8 experts, top-2) faults on torch 2.14.1 with ROCm 7.2 and
7.14, and on 2.15.0.dev nightlies through 20261001 (2/2 each on the date
bisect). The same reproducer completes on 2.15.0.dev20261002 (2/2),
2.15.0.dev20261005 (4/4) and 2.16.0.dev20261006 (12/12).

**2. The fault is not gone in general.** A sparse benchmark with 16
experts and top-4 (batch 64, block 256, bf16) aborted on
2.16.0.dev20261006 with exit 134 and the same signature
(`Memory access fault ... Page not present`, 2 MiB-aligned address). A
configuration that does not reproduce on a build is not a fixed build.

**3. Fix attribution is a hypothesis.** The behavior changes inside the
PyTorch main window 2026-10-01 to 2026-10-02 (82 commits). PR #196920
(eager hipBLAS(Lt) workspaces) is the strongest candidate: enabling the
old workspace cache on a passing build reintroduces the fault, while
disabling it on a failing build does not prevent it. No revert or
cherry-pick was built, so the commit is not proven.

**4. The mechanism is a hypothesis.** The later investigation points at a
layout-dependent over-read in hipBLASLt solution 107018 at the end of an
allocation, based on rocprof dispatch logs and a CPU model of the kernel's
address arithmetic. It is not confirmed on hardware, and this repository
does not claim a confirmed hipBLASLt bug.

**Workaround candidate, not a fix.**
`PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:4` moved the candidate
operand away from the 2 MiB boundary, and one full benchmark run passed
with an identical GEMM sequence and identical kernel solutions. One run,
no guarantee for other shapes, batches or longer runs.

**Sparse-MoE verdict.** Numerically equivalent to the dense reference
(verified), 12% faster per step and 41% less peak memory at batch 64 /
block 256 on 2.16.0.dev20261006. **Not recommended for adoption**: a
tested configuration still faults on the newest tested nightly, torch
2.14.1 is affected, and no stable PyTorch >= 2.15 release was available.
The dense-combined workaround stays shipped.

Evidence: `version-comparison.md`, `version-isolation/`, `root-cause.md`,
`phase4-scaling.md`, `issue-report.md` (draft, not submitted). The E16/K4
analysis and its read-only archive are under
`~/investigations/amdgpu-2026-10-24/`.

## Contents

- `root-cause.md` - root-cause analysis: confirmed observations,
  the micro-test false positive, working hypothesis, recommendation,
  remaining uncertainties.
- `issue-report.md` - draft upstream issue report (NOT submitted).
- `matrix.md` - reproducibility matrix (all configurations and rates).
- `environment.md` - hardware/software manifest (raw data in `raw/`).
- `version-isolation/` - cross-version study: baseline inventory, uv venv
  installs (testA rocm7.14, testB rocm7.2, testC/testD nightlies,
  testE first 2.15 nightly), smoke tests, per-env reproducer suites,
  `version-comparison.md` matrix.
- `phase4-scaling.md` - sparse vs dense scaling grid (batch, block and
  expert configurations) with the measured results and the open questions.
- `reproducer.py` - minimal deterministic reproducer (random data, no
  dataset, CLI knobs; `--no_optimizer --grad_filter experts` is the
  deterministic fault mode).
- `sparse_moe_v2_train.py` - failing sparse MoE code recovered from git
  history (commit bfafa70). `sparse_moe_v2_train_boolmask.py` is the
  earlier bool-mask variant (commit 039152a); it faulted too.
- `trace_forward.py`, `trace_backward.py` - op-level forward/backward
  tracers (per-op prints + syncs + autograd hooks).
- `capture_memhist.py` - captures PyTorch's allocation history around
  the fault (used to map the fault address to the overflowed tensor).
- `bisect.sh`, `grad_bisect.sh`, `layer_bisect.sh`, `run_matrix.sh` -
  bisect/matrix drivers; results in `*_results.txt`.
- `logs/` - preserved reproduction and trace logs (complete, unedited).

## Reproducing the fault

```bash
cd veylith/debug
python3 reproducer.py --no_optimizer --grad_filter experts --iters 100
```

Deterministic in this mode (faults ~100% at iter ~26-49). The original
training-loop configuration faults at ~17% per 250 iters (1/6 measured).
Both figures were measured on torch 2.14.1; on 2.15.0.dev20261002 and
later the same command completes.
