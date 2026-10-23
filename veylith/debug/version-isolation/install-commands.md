# Version isolation: environments and install commands

All test environments are isolated `uv` virtual environments under
`~/veylith-debug-venvs/` (outside the repository and
outside the training environment; `/tmp` is a 16GB tmpfs, too small for
the ROCm SDK packages). The system ROCm installation (`/opt/rocm`, 7.2.4)
and the Arch-packaged training environment are never modified. No system
packages are upgraded or uninstalled; no existing virtual environment is
touched.

## Baseline (existing environment, not a venv)

- Python 3.14.8 (`/usr/bin/python3`)
- torch 2.14.1 from the Arch package `python-pytorch-rocm 2.14.1-1`
  (`/usr/lib/python3.14/site-packages`), running against system ROCm 7.2.4
  (`/opt/rocm`; HIP 7.2.53211). This PyTorch build does not bundle ROCm
  libraries; it loads them from `/opt/rocm/lib` (verified via
  `/proc/self/maps`).
- Run with: system `python3`.

## Test A: torch 2.14.1 + ROCm 7.14 (latest stable)

```bash
uv venv ~/veylith-debug-venvs/testA --python /usr/bin/python3
uv pip install --python ~/veylith-debug-venvs/testA/bin/python \
    torch==2.14.1 --index-url https://download.pytorch.org/whl/rocm7.14
```

Wheel: `torch-2.14.1+rocm7.14-cp314-cp314-manylinux_2_28_x86_64.whl`.
The ROCm 7.14 runtime is pulled in as `rocm-sdk-*` pip packages
(`rocm-sdk-libraries`, `rocm-sdk-device-gfx1201`, ...) installed into the
venv, so the venv is isolated from system ROCm 7.2.4. Isolation is
verified at runtime via `/proc/self/maps` in `smoke.py`. gfx1201 device
support is explicit in the index (`rocm-sdk-device-gfx1201`).

## Test B: torch 2.14.1 + ROCm 7.2 (pip control)

```bash
uv venv ~/veylith-debug-venvs/testB --python /usr/bin/python3
uv pip install --python ~/veylith-debug-venvs/testB/bin/python \
    torch==2.14.1 --index-url https://download.pytorch.org/whl/rocm7.2
```

Same torch version and ROCm 7.2 as the baseline, but installed as a pip
wheel. Controls for "Arch package vs pip wheel" and validates that the
venv itself does not change the fault behavior.

## Test C (optional): nightly torch + ROCm 7.14

```bash
uv venv ~/veylith-debug-venvs/testC --python /usr/bin/python3
uv pip install --python ~/veylith-debug-venvs/testC/bin/python \
    torch --index-url https://download.pytorch.org/whl/nightly/rocm7.14
```

Picks the newest nightly (e.g. `torch-2.16.0.dev20261006+rocm7.14`,
cp314 available). Tests a newer torch as well as ROCm 7.14.

## Smoke test (per environment)

```bash
<env-python> veylith/debug/version-isolation/smoke.py
```

Checks: `torch.cuda.is_available()`, gfx1201 (cc 12.0), bf16 matmul,
bf16 forward+backward, fp32 gradient correctness, GPU sync, and which
ROCm/HIP libraries are loaded at runtime (system `/opt/rocm` vs venv) to
detect isolation conflicts.

## Reproducer (per environment, fresh subprocess)

```bash
cd veylith/debug
<env-python> reproducer.py --no_optimizer --grad_filter experts --iters 100
CUDA_LAUNCH_BLOCKING=1 <env-python> reproducer.py --no_optimizer --grad_filter experts --iters 100
```

Identical seeds, shapes, dtypes and parameters across environments
(the reproducer is seeded). 3 normal runs + 1 launch-blocking run per
environment; full logs and exit codes in `logs/`.

`run-env-tests.sh <label> <env-python>` automates smoke + repro runs.
