"""v3.1-PERF-2 analysis: backend comparison, dispatch, workspace, numerics."""

from __future__ import annotations

import csv
import json
import os
import re
import statistics

import torch

SCRATCH = os.path.expanduser("~/veylith-perf2-20261027")
OUT = "veylith/v3/docs/v31-perf2-summary.json"
BACKENDS = {"rocblas": "cublas", "cublaslt": "cublaslt"}
GEMM_BUDGET_MS = 245.0
UPDATE_MS = 780.4


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_runs(backend):
    runs = []
    for run in (1, 2):
        runs.append(read_json(os.path.join(SCRATCH, backend, f"time_run{run}.json")))
    return runs


def parse_logs():
    workspace = []
    hip_trans = []
    roc_trans = []
    with open(os.path.join(SCRATCH, "logs", "dispatch_cublaslt.log"),
              encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "rocblaslt_matmul" in line:
                match = re.search(r"workSpaceSizeInBytes=(\d+)", line)
                workspace.append(int(match.group(1)) if match else None)
                trans = re.search(r"transA=(\w+).*?transB=(\w+)", line)
                hip_trans.append(trans.groups() if trans else None)
    with open(os.path.join(SCRATCH, "logs", "dispatch_rocblas.log"),
              encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("rocblas_gemm_ex"):
                fields = line.strip().split(",")
                roc_trans.append((fields[1], fields[2], int(fields[3]),
                                  int(fields[4]), int(fields[5])))
    return workspace, hip_trans, roc_trans


def main():
    roc_runs = load_runs("rocblas")
    hip_runs = load_runs("cublaslt")
    roc_dispatch = read_json(os.path.join(SCRATCH, "rocblas", "dispatch.json"))
    hip_dispatch = read_json(os.path.join(SCRATCH, "cublaslt", "dispatch.json"))
    workspace, hip_trans, roc_trans = parse_logs()

    cases = []
    total_delta = 0.0
    total_baseline = 0.0
    for index in range(len(roc_runs[0]["cases"])):
        roc = [run["cases"][index] for run in roc_runs]
        hip = [run["cases"][index] for run in hip_runs]
        roc_median = statistics.median(item["median_ms"] for item in roc)
        hip_median = statistics.median(item["median_ms"] for item in hip)
        calls = roc[0]["calls_per_update"]
        delta = (hip_median - roc_median) * calls
        total_delta += delta
        total_baseline += roc[0]["baseline_ms_per_update"]
        cases.append({
            "case": index,
            "name": roc[0]["name"],
            "A": roc[0]["A"],
            "B": roc[0]["B"],
            "output_dims": roc[0]["output_dims"],
            "calls_per_update": calls,
            "perf1_ms_per_update": roc[0]["baseline_ms_per_update"],
            "rocblas_ms": {"run1": roc[0]["median_ms"], "run2": roc[1]["median_ms"],
                           "median": roc_median, "p95": roc[0]["p95_ms"]},
            "hipblaslt_ms": {"run1": hip[0]["median_ms"], "run2": hip[1]["median_ms"],
                             "median": hip_median, "p95": hip[0]["p95_ms"]},
            "ratio_hip_over_roc": hip_median / roc_median,
            "delta_ms_per_update": delta,
            "iterations": roc[0]["iterations"],
            "rocblas_allocating_ms": roc[0].get("allocating_median_ms"),
            "hipblaslt_allocating_ms": hip[0].get("allocating_median_ms"),
            "fp32_reference": roc[0].get("fp32_reference"),
            "hipblaslt_workspace_bytes": workspace[index] if index < len(workspace) else None,
            "hipblaslt_trans": hip_trans[index] if index < len(hip_trans) else None,
            "rocblas_trans": roc_trans[index] if index < len(roc_trans) else None,
            "rocblas_kernel": (roc_dispatch["cases"][index].get("kernel_names") or [None])[0],
            "hipblaslt_kernel": (hip_dispatch["cases"][index].get("kernel_names") or [None])[0],
        })

    numeric = []
    for index in range(len(cases)):
        a = torch.load(os.path.join(SCRATCH, "rocblas", "outputs", f"case{index:02d}.pt")).float()
        b = torch.load(os.path.join(SCRATCH, "cublaslt", "outputs", f"case{index:02d}.pt")).float()
        diff = (a - b).abs()
        numeric.append({"case": index, "max_abs": float(diff.max()),
                        "mean_abs": float(diff.mean())})

    run_variability = {}
    for name, runs in (("rocblas", roc_runs), ("cublaslt", hip_runs)):
        deltas = []
        for a, b in zip(runs[0]["cases"], runs[1]["cases"]):
            mean = (a["median_ms"] + b["median_ms"]) / 2
            deltas.append(abs(a["median_ms"] - b["median_ms"]) / mean * 100)
        run_variability[name] = {"median_rel_pct": statistics.median(deltas),
                                 "max_rel_pct": max(deltas)}

    rows = []
    with open(os.path.join(SCRATCH, "telemetry.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["monotonic"]),
                         int(row["power_uw"]) if row["power_uw"] else 0,
                         int(row["sclk_hz"]) if row["sclk_hz"] else 0,
                         int(row["busy_pct"]) if row["busy_pct"] else 0))
    power = {}
    for name, runs in (("rocblas", roc_runs), ("cublaslt", hip_runs)):
        selected = []
        for run in runs:
            low, high = run["started_monotonic"], run["finished_monotonic"]
            selected.extend([row for row in rows if low <= row[0] <= high])
        if selected:
            power[name] = {
                "mean_power_w": statistics.mean(row[1] for row in selected) / 1e6,
                "max_power_w": max(row[1] for row in selected) / 1e6,
                "mean_sclk_ghz": statistics.mean(row[2] for row in selected) / 1e9,
                "mean_busy_pct": statistics.mean(row[3] for row in selected),
                "samples": len(selected),
            }

    summary = {
        "meta": {
            "experiment": "v3.1-PERF-2 actual-shape BF16 GEMM benchmark",
            "date_utc": "2026-10-27",
            "mode": "benchmark only; no model, optimizer or backward path",
            "git_head_before": "b298798",
            "scratch": SCRATCH,
        },
        "environment": {
            "torch": roc_runs[0]["torch"]["version"],
            "hip": roc_runs[0]["torch"]["hip"],
            "gpu": roc_runs[0]["device"]["name"],
            "gcn_arch": roc_runs[0]["device"]["gcn_arch"],
            "triton": "3.8.0",
        },
        "backend_selection": {
            "api": "torch.backends.cuda.preferred_blas_library",
            "default": "Cublaslt -> hipBLASLt (rocblaslt_matmul)",
            "options": {"cublaslt": "hipBLASLt", "cublas": "rocBLAS",
                        "ck": "installed but unsupported on gfx1201"},
            "ck_rejection": "RuntimeError: architecture supported for CK GEMM: 0",
            "env_var": "TORCH_BLAS_PREFER_HIPBLASLT=1/0 selects the same two settings",
            "dispatch_evidence": {
                "rocblas_log_lines": len(roc_trans),
                "hipblaslt_log_lines": len(hip_trans),
                "rocblas_call": "rocblas_gemm_ex",
                "hipblaslt_call": "rocblaslt_matmul",
            },
        },
        "cases": cases,
        "totals": {
            "cases": len(cases),
            "perf1_gemm_ms_per_update": total_baseline,
            "hipblaslt_minus_rocblas_ms_per_update": total_delta,
            "pct_of_perf1_gemm": total_delta / total_baseline * 100 if total_baseline else None,
            "pct_of_245ms_budget": total_delta / GEMM_BUDGET_MS * 100,
            "pct_of_measured_update": total_delta / UPDATE_MS * 100,
        },
        "numerics": {
            "cases_compared": len(numeric),
            "max_abs_difference": max(item["max_abs"] for item in numeric),
            "all_bitwise_equal": all(item["max_abs"] == 0 for item in numeric),
            "details": numeric,
        },
        "memory": {
            "peak_reserved_gb": {
                "rocblas": max(run["peak_reserved_bytes"] for run in roc_runs) / 1e9,
                "cublaslt": max(run["peak_reserved_bytes"] for run in hip_runs) / 1e9,
            },
            "peak_allocated_gb": {
                "rocblas": max(run["peak_allocated_bytes"] for run in roc_runs) / 1e9,
                "cublaslt": max(run["peak_allocated_bytes"] for run in hip_runs) / 1e9,
            },
            "hipblaslt_workspace_bytes": {
                "min": min(v for v in workspace if v is not None),
                "max": max(v for v in workspace if v is not None),
                "per_case": workspace,
            },
            "rocblas_workspace": "not exposed by the rocBLAS log",
        },
        "run_variability": run_variability,
        "power": power,
        "power_scope": "GPU package PPT only; whole-system wall power not measured",
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
    print("weighted delta", round(total_delta, 2), "ms/update;",
          round(total_delta / total_baseline * 100, 2), "% of inventory")
    print("workspace bytes min/max", min(v for v in workspace if v),
          max(v for v in workspace if v))


if __name__ == "__main__":
    main()
