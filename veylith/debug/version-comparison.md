# Version comparison: does the fault persist across PyTorch/ROCm builds?

**Question:** is the BF16 sparse-MoE backward memory fault specific to our
PyTorch 2.14.1 / ROCm 7.2 environment, or does it also occur with newer
ROCm-enabled PyTorch builds?

**Answer:** the fault reproduces on torch 2.14.1 with BOTH ROCm 7.2 and
ROCm 7.14 (ROCm 7.14 does NOT fix it), and is fixed in newer PyTorch
builds: the first tested working version is **torch 2.15.0.dev20261005
(+rocm7.14)**. The fix is therefore in **PyTorch's 2.15 dev cycle**
(somewhere between the 2.15.0.dev builds of 2026-08-24 and 2026-10-05),
not in the ROCm runtime.

## Test matrix

All runs: `reproducer.py --no_optimizer --grad_filter experts --iters 100`
(deterministic fault mode; 3 normal runs + 1 `CUDA_LAUNCH_BLOCKING=1` run
per environment, identical seeds/shapes/dtypes). Environments are isolated
uv venvs under `~/veylith-debug-venvs/`; isolation
verified at runtime via `/proc/self/maps` in `smoke.py` (venv ROCm SDK
libs, not system `/opt/rocm`). GPU verified healthy between suites.

| env | python | torch | ROCm (runtime) | GPU | smoke | repro runs | fault rate | signature |
|---|---|---|---|---|---|---|---|---|
| baseline | 3.14.8 (system) | 2.14.1 (Arch `python-pytorch-rocm 2.14.1-1`) | 7.2.4 / HIP 7.2.53211 (system `/opt/rocm`) | gfx1201 | PASS | 4/4 FAULT | 100% | Page not present, 2MB-aligned addr, iter 25, exit 134 |
| testB | 3.14.8 | 2.14.1+rocm7.2 (pip) | 7.2 / 7.2.53211 (venv rocm-sdk, 0 system libs) | gfx1201 | PASS | 4/4 FAULT | 100% | same |
| testA | 3.14.8 | 2.14.1+rocm7.14 (pip) | 7.14 / 7.14.60850 (venv rocm-sdk 7.14.1, 1 system lib) | gfx1201 | PASS | 4/4 FAULT | 100% | same |
| testE | 3.14.8 | 2.15.0.dev20260824+rocm7.14 (pip nightly) | 7.14 / 7.14.60850 (venv) | gfx1201 | PASS | 4/4 FAULT | 100% | same |
| testD | 3.14.8 | 2.15.0.dev20261005+rocm7.14 (pip nightly) | 7.14 / 7.14.60850 (venv) | gfx1201 | PASS | 4/4 COMPLETED | 0% | - |
| testC | 3.14.8 | 2.16.0.dev20261006+rocm7.14 (pip nightly) | 7.14 / 7.14.60850 (venv) | gfx1201 | PASS | 12/12 COMPLETED | 0% | - |

testC's 12 completions: 4 from the standard suite + 5 extra frozen-mode
runs (different nothing; same mode) + 3 runs of the original
training-loop configuration (`--grad_filter all`, frozen). testB smoke:
33/33 libs from the venv (0 system); testA/testC/testD: 60/61 from the
venv (1 minor lib from `/opt/rocm`; the HIP runtime, HSA runtime, and
math libraries all come from the venv's `rocm-sdk-core` 7.14.1 packages).

## Interpretation

1. **Bug fixed in a newer version (PyTorch).** testA vs testD/testC is a
   clean ROCm-controlled comparison: same GPU, same kernel driver, same
   ROCm 7.14 runtime lineage (torch.version.hip = 7.14.60850 in both),
   same reproducer, same seeds. torch 2.14.1 faults, torch 2.15.0.dev20261005
   and 2.16.0.dev20261006 do not. The fix is in PyTorch, not ROCm.
2. **Reproducible across ROCm versions on torch 2.14.1.** ROCm 7.2
   (baseline, testB) and ROCm 7.14 (testA) both fault. ROCm 7.14 does
   not fix it.
3. **Not an Arch-package artifact.** The pip wheel of torch 2.14.1+rocm7.2
   (testB) faults identically to the Arch package (baseline).
4. **Not an isolation artifact.** testB/testA/testC/testD/testE all run
   fully isolated venvs and the faulting ones still fault.
5. **Fix window:** between `2.15.0.dev20260824` (faults 4/4) and
   `2.15.0.dev20261005` (passes 4/4), i.e. PyTorch commits landed between
   2026-08-24 and 2026-10-05 in the 2.15 dev cycle.
6. **Inconclusive/untested:** no PyTorch version older than 2.14.1 was
   tested, so it is unknown whether 2.14.1 was a regression or whether
   the bug is long-standing in the ROCm backend. No environment was
   incompatible; every smoke test passed, so there are no
   "untested/incompatible" rows.

## First tested working version

**torch 2.15.0.dev20261005+rocm7.14** (nightly). It and
torch 2.16.0.dev20261006+rocm7.14 both pass the deterministic fault
reproducer (0 faults in 4+12 runs) and all smoke checks.

## Can Veylith safely use it?

- The only currently-working builds are **nightlies**, not stable
  releases. Nightlies move daily and are not reproducible environments.
- Recommendation: keep the dense-MoE workaround (already shipped in v2)
  on the current stable environment; switch the training environment to a
  stable PyTorch >= 2.15 (with ROCm 7.14 or newer) when it is released,
  then re-verify with this reproducer and the smoke tests before
  returning to sparse MoE.
- Upgrading the system torch is a system change (Arch package), so that
  decision stays with the user; the venvs here prove the newer builds
  work on this exact GPU/driver without touching the system.

## Remaining uncertainties

- The exact PyTorch commit/PR that fixed it: nightlies are date-tagged,
  not commit-tagged; the window is 2026-08-24 to 2026-10-05. Bisecting
  further would mean installing more dated nightlies; the window is
  narrow enough for the upstream report to point at.
- The kernel-level root cause remains unnamed (no compute-sanitizer/
  rocgdb on this machine); the cross-version evidence localizes it to a
  PyTorch ROCm kernel on the sparse-MoE backward path (bf16), fixed in
  the 2.15 cycle.
- PyTorch versions older than 2.14.1 untested.
