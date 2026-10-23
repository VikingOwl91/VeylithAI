# Debug: ROCm memory access fault in veylith v2 sparse MoE

Investigation of the GPU "Memory access fault" (page not present) that
killed sparse-MoE training runs of veylith v2. The working dense-MoE
v2 and all of v1 are preserved on `main`; this directory and the
`debug/rocm-sparse-moe-fault` branch are the investigation sandbox.

**Status: complete.** Root cause: a bf16 GPU kernel on the sparse-MoE
backward path writes out of bounds despite in-bounds indices (PyTorch/ROCm
kernel bug). Our implementation is correct; the dense-MoE application fix
is already shipped in v2. **Cross-version:** fixed in PyTorch's 2.15 dev
cycle (faults on 2.14.1 with ROCm 7.2 AND 7.14 and on 2.15.0.dev20260824;
passes on 2.15.0.dev20261005 and 2.16.0.dev20261006). See
`version-comparison.md` and `version-isolation/`.

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
