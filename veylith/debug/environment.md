# Environment manifest

Raw outputs preserved in `raw/` (`environment.txt`, `rocminfo.txt`, `tools.txt`).

## Hardware

| component | value |
|---|---|
| GPU | AMD Radeon RX 9070 XT (Navi 48) |
| GPU arch | gfx1201 (amdgcn-amd-amdhsa--gfx1201) |
| Chip ID | 0x7550 |
| Compute units | 64 |
| VRAM | 17.1 GB |
| CPU | AMD Ryzen 7 3700X 8-Core |

## Software

| component | version | source |
|---|---|---|
| OS | Arch Linux | /etc/os-release |
| Linux kernel | 7.2.11-zen1-1.1-zen x86_64 | uname -a |
| amdgpu kernel driver | not readable (`/sys/module/amdgpu/version` n/a) | /sys |
| ROCm userspace | 7.2.4 | /opt/rocm/.info/version |
| HIP (PyTorch build) | 7.2.53211 | torch.version.hip |
| PyTorch | 2.14.1 (ROCm build; `torch.version.cuda` is None, expected for ROCm wheels) | pip |
| Python | 3.14.8 | python3 --version |
| sentencepiece | 0.2.2 | pip |
| numpy | 2.5.4 | pip |
| rocm-smi | 4.0.0 (lib 7.8.0) | rocm-smi --version |

## Environment variables

No `HIP*`, `HSA*`, `ROCR*`, `AMD*`, `CUDA*`, or `GPU*` variables are set
in the shell environment. Per-process variables used during the
investigation (documented per run, never system-wide):

- `CUDA_LAUNCH_BLOCKING=1` (test setting: serialize kernel launches)

## Debugging tools availability (ROCm 7.2.4, gfx1201)

| tool | available | notes |
|---|---|---|
| compute-sanitizer | no | not in PATH, not under /opt/rocm; would be the ideal OOB/UAF detector |
| rocgdb | no | not in PATH, no gdb in /opt/rocm/llvm/bin |
| rocprof (v2) | yes | deprecated upstream, but present |
| rocprofv3 | yes | modern profiler; supports --kernel-trace, --hip-trace, --memory-allocation-trace, --stats, --summary |
| rocprofv3-avail | yes | agent/counter query |
| rocpd / rocpd2csv | yes | profile data conversion |
| rocm-smi / amd-smi | yes | GPU monitoring |
| MIOpenDriver, rocblas-bench | yes | library-level microbenchmarks |

No system-wide ROCm, PyTorch, kernel, or GPU configuration was changed
during this investigation (constraint). All test settings are per-process.
