# Veylith v3.1-H isolated ROCm environment

Status 2026-10-27. Installed and CPU-validated. No GPU workload, training or
sparse-MoE reproduction has been run.

The environment is at `~/veylith-v31-isolated`. It uses a uv-managed CPython
3.14.8 and a hash-locked PyTorch ROCm 7.2 wheel. Arch's system Python, system
PyTorch, `/opt/rocm` and the existing venvs are not used by the isolated
interpreter.

## What is isolated

- Managed Python interpreter and standard library, outside `/usr/bin`.
- Python packages only in the venv, with `include-system-site-packages = false`
  and `site.ENABLE_USER_SITE = False`.
- PyTorch and its compiled extensions from the wheel.
- ROCm user-space libraries bundled in the wheel, including HIP, rocBLAS,
  hipBLASLt, MIOpen, RCCL, rocSOLVER, rocSPARSE, rocFFT, rocRAND and the
  gfx1201 kernels.
- Torch's dynamic linker resolution through `RPATH=$ORIGIN`, with no
  `/opt/rocm` library in the resolved dependency graph.
- Python and ROCm environment variables through the launcher.

Still host-dependent:

- Linux kernel, amdgpu/KFD, firmware, `/dev/kfd`, `/dev/dri` and user groups.
- glibc, libstdc++, libgcc, libzstd and liblzma.
- `uv` for installation, unless a pinned standalone `uv` is kept.
- Filesystem, dataset root and checkpoint directory.

Arch Linux is not an officially supported ROCm 7.2.1 operating system. The GPU
and gfx1201 target are listed by AMD, but the supported OS list for the Radeon
RX 9000 class is Ubuntu 24.04.4, Ubuntu 22.04.5, RHEL 10.1 and RHEL 9.7. The
local CPU checks and the earlier testB smoke are experimental evidence on Arch,
not an official support statement.

## Exact versions

| component | version |
|---|---|
| Python | 3.14.8, uv-managed |
| uv | 0.13.5 |
| torch | 2.14.1+rocm7.2 |
| HIP | 7.2.53211 |
| triton-rocm | 3.8.0 |
| numpy | 2.5.4 |
| sentencepiece | 0.2.2 |
| pytest | 9.1.1 |
| filelock | 3.32.3 |
| fsspec | 2026.7.0 |
| jinja2 | 3.1.6 |
| markupsafe | 3.0.3 |
| mpmath | 1.3.0 |
| networkx | 3.6.1 |
| setuptools | 78.1.0 |
| sympy | 1.14.0 |
| typing-extensions | 4.16.0 |
| iniconfig | 2.3.1 |
| packaging | 26.3 |
| pluggy | 1.7.0 |
| pygments | 2.21.0 |

## Package sources and wheel provenance

`pyproject.toml` is the dependency source of truth. It pins the direct
dependencies and declares two explicit indexes:

- PyPI `https://pypi.org/simple` for ordinary packages.
- PyTorch `https://download.pytorch.org/whl/rocm7.2` as an explicit source for
  `torch` and `triton-rocm` only.

`uv.lock` is the canonical resolution with hashes and source annotations.
`requirements.lock` is the exported, hash-locked requirements file used for
review and portable installation. It contains both the standard cp314 and the
free-threaded cp314t torch hashes; the standard interpreter selects cp314.

PyTorch wheel:

- File: `torch-2.14.1+rocm7.2-cp314-cp314-manylinux_2_28_x86_64.whl`
- Index: `https://download.pytorch.org/whl/rocm7.2/torch/`
- SHA-256: `e378baf90ba06c988e234449787a6fd9dac60dcd869eef35cfcc9546cd3f1f14`
- Size: 6,224,879,579 bytes
- Index last modified: 2026-09-30
- Local cache: `~/veylith-wheel-cache/torch-2.14.1+rocm7.2-cp314-cp314-manylinux_2_28_x86_64.whl`

The wheel hash was checked before installation. The local cache copy is
outside Git and can be reused to reconstruct the environment if the index is
unavailable. The lockfile hashes are the installation contract.

## Creation and reproduction

The committed files are `pyproject.toml`, `uv.lock` and `requirements.lock`.
The following commands are fish-compatible and are for a later authorized
rebuild. Do not run them against an environment that is in use.

```fish
uv python install 3.14.8
uv venv --managed-python --python 3.14.8 ~/veylith-v31-isolated
uv lock --project veylith/v3/env --python ~/veylith-v31-isolated/bin/python
uv export --project veylith/v3/env --format requirements.txt --output-file veylith/v3/env/requirements.lock --emit-index-url
env VIRTUAL_ENV=$HOME/veylith-v31-isolated uv sync --project veylith/v3/env --active --frozen
uv pip check --python ~/veylith-v31-isolated/bin/python
```

`uv sync --frozen` uses the explicit source mapping in `pyproject.toml` and
`uv.lock`; it does not resolve `numpy` from the PyTorch index. `uv pip install
-r requirements.lock` can fail under the default first-index strategy because
the PyTorch index also hosts a `numpy` entry. Use `uv sync` for this project.

The wheel can be verified independently:

```fish
curl --fail --location --output ~/veylith-wheel-cache/torch-2.14.1+rocm7.2-cp314-cp314-manylinux_2_28_x86_64.whl https://download-r2.pytorch.org/whl/rocm7.2/torch-2.14.1%2Brocm7.2-cp314-cp314-manylinux_2_28_x86_64.whl
sha256sum ~/veylith-wheel-cache/torch-2.14.1+rocm7.2-cp314-cp314-manylinux_2_28_x86_64.whl
```

Expected: `e378baf90ba06c988e234449787a6fd9dac60dcd869eef35cfcc9546cd3f1f14`.

## Launcher

The committed launcher is `veylith/v3/env/veylith-python`. Symlink it into the
environment's bin directory:

```fish
ln -sf ~/git/AI-Research-v2dev/veylith/v3/env/veylith-python ~/veylith-v31-isolated/bin/veylith-python
ln -sf ~/git/AI-Research-v2dev/veylith/v3/env/veylith-train ~/veylith-v31-isolated/bin/veylith-train
```

The launcher:

- Uses `$HOME/veylith-v31-isolated/bin/python` by default.
- Removes `PYTHONPATH`, `PYTHONHOME`, `LD_LIBRARY_PATH`, `ROCM_PATH` and
  `HIP_PATH`.
- Sets `PYTHONNOUSERSITE=1`.
- Works when invoked by absolute path from fish or a POSIX shell.
- Does not require `source activate`.

`VEYLITH_ENV_DIR` overrides the environment path.

Quick checks:

```fish
~/veylith-v31-isolated/bin/veylith-python -c "import sys, site, torch; print(sys.executable); print(site.ENABLE_USER_SITE); print(torch.__version__, torch.version.hip)"
```

Expected: the isolated interpreter, `False`, `2.14.1+rocm7.2`, `7.2.53211`.

## Fingerprint

`env_fingerprint.py` writes schema version 1 JSON with:

- Python version, executable, `prefix`, `base_prefix`, user-site state,
  `sys.path` and any site-packages leakage.
- Installed distributions and versions.
- Torch version, HIP version, config hash, file path and wheel hashes from the
  lock.
- `ldd` and `readelf` results for `libtorch_hip.so`, including RPATH/RUNPATH,
  ROCm library paths and the count of `/opt/rocm` libraries.
- `/proc/self/maps` ROCm library provenance.
- Host kernel, system ROCm version, relevant pacman package versions and device
  node modes.
- GPU identity from sysfs (`PCI_ID`, subsystem, driver), without HIP init.
- Lockfile SHA-256 for `requirements.lock` and `uv.lock`, plus torch wheel hashes.

Generate the accepted baseline:

```fish
~/veylith-v31-isolated/bin/veylith-python veylith/v3/env/env_fingerprint.py --out veylith/v3/env/env.baseline.json
```

The committed baseline is machine-specific because it records host package
versions and the GPU PCI identity. On another host, generate a new baseline
after the GPU validation ladder, not before.

## Drift guard

`env_check.py` compares the current environment with the accepted baseline:

```fish
~/veylith-v31-isolated/bin/veylith-python veylith/v3/env/env_check.py --expected veylith/v3/env/env.baseline.json --strict-host
```

Critical checks:

- Python path, version, prefix, base prefix and user-site state.
- No system or user site-packages in `sys.path`.
- Exact installed package set and versions.
- Lockfile SHA-256 for `requirements.lock` and `uv.lock`, plus the expected torch wheel hash.
- Torch version, HIP version and build config hash.
- No `/opt/rocm` libraries in `ldd` or `/proc/self/maps`.
- `RPATH` or `RUNPATH` containing `$ORIGIN`.
- GPU PCI identity.

Host kernel, system ROCm, firmware, glibc, gcc-libs, zstd and xz changes are
warnings by default and failures under `--strict-host`. The `veylith-train`
wrapper uses `--strict-host`.

The guard never installs, synchronizes, repairs or upgrades anything. It only
reads the environment and runs read-only inspection commands. With `--json` it
also writes the current fingerprint for provenance. A negative test was run:
changing the baseline numpy version produced
`CRITICAL package.numpy` and exit code 1.

## Checkpoint provenance

`training/checkpoint.py` now records an `environment` block inside `versions()`:

- `sys_prefix`, `base_prefix`
- `user_site`
- `hip`
- `fingerprint_path`
- `fingerprint_sha256`

The fingerprint path comes from `VEYLITH_ENV_FINGERPRINT`. The `veylith-train`
wrapper regenerates `~/veylith-v31-isolated/env.fingerprint.json`, checks it,
sets the variable and then starts `train_v31.py`. New states and artifacts
carry the fingerprint hash. Existing checkpoints load unchanged because
`load_state` does not require the new field and the checkpoint format version
was not changed.

## CPU validation

The isolated interpreter passed the full v3 suite with GPU visibility
disabled:

```fish
env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ~/veylith-v31-isolated/bin/veylith-python -m pytest veylith/v3/tests -q
```

Result: **90 passed, 1 skipped**. This covers model construction, architecture
configuration, forward pass, loss, Dense-Combined MoE, sparse/dense CPU parity,
checkpoint serialization and loading, NumPy interoperability, SentencePiece
tokenizer loading, training configuration parsing and environment provenance.

No GPU tensor allocation, HIP initialization, GPU synchronization or training
was performed. The historical GPU fault reproducer was not executed.

`import torch` in this environment prints one or two `(null): No such file or
directory` lines to stderr. All `ldd` dependencies resolve, the drift guard
passes and the CPU suite passes. The source is not identified; treat it as a
startup diagnostic to watch during the GPU smoke, not as a missing dependency.

## Native provenance

`libtorch_hip.so` has `RPATH=$ORIGIN`. `ldd` resolves all ROCm libraries from
`~/veylith-v31-isolated/lib/python3.14/site-packages/torch/lib`; the count of
libraries from `/opt/rocm` is **0**. Host dependencies are libc, libm,
libstdc++, libgcc, libzstd and liblzma. `/opt/rocm` is not used by the
isolated interpreter.

## Disk usage

- Isolated environment: **15 GB**
- Wheel cache: **5.8 GB**
- Filesystem free space at installation: **335 GB**

The 6.22 GB wheel cache can be removed after the environment is accepted, but
keeping it makes reconstruction cheaper. Do not commit it.

## Upgrade and rollback

Do not upgrade in place. Create a new environment, for example
`~/veylith-v31-isolated-next`, with a new `pyproject.toml`/lock pair. Run the
CPU suite, the fingerprint check and the separately authorized GPU smoke
before switching. Keep the current environment and baseline until the new one
passes.

To roll back, point `VEYLITH_ENV_DIR` or the launcher symlink back to the old
environment. Do not downgrade packages in the accepted environment. Do not
touch the existing `~/veylith-v31-gpu-env` or `~/veylith-v3-env`; they remain
available as comparison and fallback, not as the isolated training stack.

## Host limitations and GPU boundary

The host kernel, amdgpu driver, KFD, firmware, glibc and C++ runtime remain
outside the environment. `pacman -Syu` can change them even though it cannot
change the venv's Python packages. The drift guard detects and reports the
change; it cannot prevent it or reproduce the old host.

The environment is ready for a separately authorized GPU smoke test. That
smoke must verify `torch.cuda.is_available`, gfx1201 identity, bf16 forward and
backward, SDPA causal and sliding-window attention, Dense-Combined MoE forward
and backward, VRAM, and checkpoint loading. It must also confirm that the
`(null)` import diagnostic is benign. Sparse-MoE GPU execution is not part of
the smoke: it is the known unresolved fault and needs its own authorization.
No GPU workload is authorized in this phase.
