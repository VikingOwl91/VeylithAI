"""v3.1-PERF-2 actual-shape BF16 GEMM benchmark.

One backend per process. Shapes, layouts and call frequencies are taken from
the v3.1-PERF-1 trace (aten::mm input dims and strides). No model, no
optimizer, no backward path: each case is a single torch.mm with BF16 inputs
and the recorded view layout.

Modes:
    time      warmup, 20 synchronized repetitions per case, saves outputs
    dispatch  one call per case under BLAS logging, records kernel names

Backend selection uses torch.backends.cuda.preferred_blas_library:
    cublaslt  -> hipBLASLt (rocblaslt_matmul)
    cublas    -> rocBLAS   (rocblas_gemm_ex)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

SCRATCH = os.path.expanduser("~/veylith-perf2-20261027")

# (name, A dims, A strides, B dims, B strides, calls per update, PERF-1 ms/update)
CASES = [
    ("fwd_expert_w1w2", (4088, 512), (512, 1), (512, 960), (1, 512), 896, 49.09),
    ("dw_expert_w1w2", (960, 4088), (1, 960), (4088, 512), (512, 1), 1792, 47.11),
    ("nn_4088_960_512", (4088, 960), (960, 1), (960, 512), (512, 1), 1792, 39.82),
    ("dw_expert_w3", (512, 4088), (1, 512), (4088, 960), (960, 1), 896, 23.41),
    ("nn_4088_512_960", (4088, 512), (512, 1), (512, 960), (960, 1), 896, 23.14),
    ("fwd_expert_w3", (4088, 960), (960, 1), (960, 512), (1, 960), 448, 21.19),
    ("dw_head", (8192, 4088), (1, 8192), (4088, 512), (512, 1), 16, 6.39),
    ("nn_4088_512_512", (4088, 512), (512, 1), (512, 512), (512, 1), 256, 5.65),
    ("tn_512_4088_512", (512, 4088), (1, 512), (4088, 512), (512, 1), 256, 4.83),
    ("tn_128_4088_512", (128, 4088), (1, 128), (4088, 512), (512, 1), 256, 4.11),
    ("nt_attn_512", (4088, 512), (512, 1), (512, 512), (1, 512), 128, 4.07),
    ("fwd_head", (4088, 512), (512, 1), (512, 8192), (1, 512), 8, 2.69),
    ("nn_4088_8192_512", (4088, 8192), (8192, 1), (8192, 512), (512, 1), 16, 2.54),
    ("dw_dense_mlp", (1365, 4088), (1, 1365), (4088, 512), (512, 1), 32, 2.44),
    ("nn_4088_128_512", (4088, 128), (128, 1), (128, 512), (512, 1), 256, 2.37),
    ("nt_kv_128", (4088, 512), (512, 1), (512, 128), (1, 512), 128, 2.32),
    ("tn_router_8", (8, 4088), (1, 8), (4088, 512), (512, 1), 112, 1.70),
    ("nt_dense_mlp", (4088, 512), (512, 1), (512, 1365), (1, 512), 16, 1.22),
    ("nn_4088_1365_512", (4088, 1365), (1365, 1), (1365, 512), (512, 1), 32, 1.13),
    ("nn_4088_8_512", (4088, 8), (8, 1), (8, 512), (512, 1), 112, 1.00),
    ("tn_512_4088_1365", (512, 4088), (1, 512), (4088, 1365), (1365, 1), 16, 0.86),
    ("nn_4088_1365", (4088, 512), (512, 1), (512, 1365), (1365, 1), 16, 0.70),
    ("nt_8", (4088, 512), (512, 1), (512, 8), (1, 512), 56, 0.60),
    ("nt_1365", (4088, 1365), (1365, 1), (1365, 512), (1, 1365), 8, 0.54),
]

BACKENDS = {
    "cublaslt": "cublaslt",
    "rocblas": "cublas",
}


def make_tensor(dims, strides, seed, device):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if tuple(strides) == (dims[1], 1):
        tensor = torch.randn(dims, generator=generator, dtype=torch.bfloat16,
                             device="cpu").to(device)
    elif tuple(strides) == (1, dims[0]):
        base = torch.randn((dims[1], dims[0]), generator=generator,
                           dtype=torch.bfloat16, device="cpu").to(device)
        tensor = base.t()
    else:
        raise ValueError(f"unsupported stride pattern {dims} {strides}")
    if tuple(tensor.shape) != tuple(dims) or tuple(tensor.stride()) != tuple(strides):
        raise AssertionError(f"layout mismatch {dims} {strides} -> "
                             f"{tuple(tensor.shape)} {tuple(tensor.stride())}")
    return tensor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--mode", choices=("time", "dispatch"), default="time")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    sys.path.insert(0, "veylith/v3")
    import torch

    torch.backends.cuda.preferred_blas_library(BACKENDS[args.backend])
    torch.manual_seed(0)
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tensor_dir = os.path.join(out_dir, "outputs")
    if args.mode == "time":
        os.makedirs(tensor_dir, exist_ok=True)

    records = []
    started = time.monotonic()
    for index, (name, dims_a, strides_a, dims_b, strides_b, calls, ms_baseline) in enumerate(CASES):
        seed = 1000 + index
        A = make_tensor(dims_a, strides_a, seed, device)
        B = make_tensor(dims_b, strides_b, seed + 5000, device)
        entry = {
            "case": index,
            "name": name,
            "A": {"dims": list(dims_a), "strides": list(strides_a)},
            "B": {"dims": list(dims_b), "strides": list(strides_b)},
            "calls_per_update": calls,
            "baseline_ms_per_update": ms_baseline,
        }
        try:
            entry["output_dtype"] = "torch.bfloat16"
            if args.mode == "dispatch":
                entry["dispatch_only"] = True
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                    C = torch.mm(A, B)
                    torch.cuda.synchronize()
                if C.dtype != torch.bfloat16 or tuple(C.shape) != (dims_a[0], dims_b[1]):
                    raise RuntimeError(f"unexpected output {C.dtype} {tuple(C.shape)}")
                entry["output_dims"] = list(C.shape)
                kernels = []
                for event in prof.key_averages():
                    device_time = (getattr(event, "self_device_time_total", 0)
                                   or getattr(event, "self_cuda_time_total", 0))
                    key = event.key
                    if (device_time and not key.startswith("aten::")
                            and "Memcpy" not in key and "Sync" not in key
                            and "Launch" not in key and "void " not in key):
                        kernels.append((key, float(device_time)))
                kernels.sort(key=lambda item: -item[1])
                entry["kernel_names"] = [name for name, _ in kernels[:3]]
                del C
            else:
                out = torch.empty((dims_a[0], dims_b[1]), device=device,
                                  dtype=torch.bfloat16)
                torch.mm(A, B, out=out)
                torch.cuda.synchronize()
                if out.dtype != torch.bfloat16 or tuple(out.shape) != (dims_a[0], dims_b[1]):
                    raise RuntimeError(f"unexpected output {out.dtype} {tuple(out.shape)}")
                entry["output_dims"] = list(out.shape)
                for _ in range(50):
                    torch.mm(A, B, out=out)
                torch.cuda.synchronize()
                if args.iterations:
                    iterations = args.iterations
                else:
                    sample_started = time.perf_counter()
                    for _ in range(50):
                        torch.mm(A, B, out=out)
                    torch.cuda.synchronize()
                    estimate = (time.perf_counter() - sample_started) / 50
                    iterations = max(100, min(8000, int(0.1 / max(estimate, 1e-7))))
                repetitions = 10
                samples = []
                for _ in range(repetitions):
                    torch.cuda.synchronize()
                    rep_started = time.perf_counter()
                    for _ in range(iterations):
                        torch.mm(A, B, out=out)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - rep_started) / iterations)
                entry.update({
                    "iterations": iterations,
                    "repetitions": repetitions,
                    "median_ms": statistics.median(samples) * 1000,
                    "mean_ms": statistics.mean(samples) * 1000,
                    "p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))] * 1000,
                    "min_ms": min(samples) * 1000,
                    "max_ms": max(samples) * 1000,
                    "sd_ms": statistics.stdev(samples) * 1000,
                    "timing_path": "torch.mm(A, B, out=preallocated)",
                })
                if index < 6:
                    allocating = []
                    for _ in range(10):
                        torch.cuda.synchronize()
                        rep_started = time.perf_counter()
                        for _ in range(iterations):
                            torch.mm(A, B)
                        torch.cuda.synchronize()
                        allocating.append((time.perf_counter() - rep_started) / iterations)
                    entry["allocating_median_ms"] = statistics.median(allocating) * 1000
                reference = torch.mm(A.float(), B.float())
                diff = (out.float() - reference).abs()
                entry["fp32_reference"] = {
                    "max_abs": float(diff.max()),
                    "mean_abs": float(diff.mean()),
                    "relative_l2": float(diff.norm() / reference.norm()),
                }
                torch.save(out.detach().cpu(), os.path.join(tensor_dir, f"case{index:02d}.pt"))
                del reference, diff, out
            torch.cuda.synchronize()
            entry["allocated_bytes_after"] = int(torch.cuda.memory_allocated())
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            del A, B
            torch.cuda.empty_cache()
        records.append(entry)

    payload = {
        "backend": args.backend,
        "requested_library": BACKENDS[args.backend],
        "effective_backend": str(torch.backends.cuda.preferred_blas_library()),
        "mode": args.mode,
        "run": args.run,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "gcn_arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
        },
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "cases": records,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
        "elapsed_seconds": time.monotonic() - started,
        "loaded_libraries": [
            line.split()[-1] for line in open("/proc/self/maps").read().splitlines()
            if "libhipblaslt" in line or "librocblas" in line][:6],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    errors = [entry for entry in records if "error" in entry]
    print("backend", args.backend, "mode", args.mode, "cases", len(records),
          "errors", len(errors), "elapsed", round(payload["elapsed_seconds"], 2))
    for entry in errors:
        print("  ERROR", entry["case"], entry["name"], entry["error"])
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
