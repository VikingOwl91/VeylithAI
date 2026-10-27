"""v3.1-PERF-1 analysis: baseline statistics, trace attribution, energy.

Reads the scratch artifacts only and writes one compact summary JSON. Trace
durations in the Chrome JSON are microseconds; everything reported per update
divides by five (the number of profiled updates per trace).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict

SCRATCH = os.path.expanduser("~/veylith-perf1-20261027")
OUT = "veylith/v3/docs/v31-perf1-summary.json"
UPDATES = 5
TARGETS_PER_UPDATE = 32704
CEILING = 0.7169171866886992


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values, point):
    ordered = sorted(values)
    return ordered[int(point * (len(ordered) - 1))]


def kernel_category(name: str) -> str:
    lowered = name.lower()
    if name.startswith("Cijk_") or "gemm" in lowered or "rocblas" in lowered:
        return "gemm"
    if "bwd_kernel" in lowered or "attention" in lowered or "fmha" in lowered or "flash" in lowered:
        return "attention"
    if "softmax" in lowered:
        return "softmax_ce"
    if "multi_tensor_apply" in lowered:
        return "optimizer"
    if "memcpy" in lowered or "copy" in lowered:
        return "copy"
    if ("elementwise" in lowered or "vectorized" in lowered or "unary" in lowered
            or "binary" in lowered or "functor" in lowered
            or "tensoriterator" in lowered):
        return "elementwise"
    if "reduce" in lowered or "sum" in lowered or "norm" in lowered:
        return "reduction"
    if "fill" in lowered or "memset" in lowered:
        return "fill"
    return "other"


def union_seconds(intervals):
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for low, high in intervals[1:]:
        if low > end:
            total += end - start
            start, end = low, high
        else:
            end = max(end, high)
    total += end - start
    return total


def parse_trace(path: str, updates: int, shapes: bool):
    data = read_json(path)
    events = data.get("traceEvents") if isinstance(data, dict) else data
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy")]
    cpu = [e for e in events if e.get("cat") == "cpu_op"]
    runtime = [e for e in events if e.get("cat") == "cuda_runtime"]
    annotations = [e for e in events if e.get("cat") == "user_annotation"]

    total_kernel_us = sum(float(e.get("dur", 0)) for e in kernels)
    categories = defaultdict(lambda: [0.0, 0])
    kernel_names = defaultdict(lambda: [0.0, 0])
    for event in kernels:
        dur = float(event.get("dur", 0))
        category = kernel_category(event.get("name", ""))
        categories[category][0] += dur
        categories[category][1] += 1
        kernel_names[(category, event.get("name", ""))][0] += dur
        kernel_names[(category, event.get("name", ""))][1] += 1

    launch_us = sum(float(e.get("dur", 0)) for e in runtime
                    if "LaunchKernel" in (e.get("name") or ""))

    phase_names = {"data_fetch", "forward", "backward", "optimizer",
                   "grad_clip", "update"}
    phases = [e for e in annotations if e.get("name") in phase_names]
    phase_cpu = defaultdict(lambda: [0.0, 0])
    for event in phases:
        phase_cpu[event["name"]][0] += float(event.get("dur", 0))
        phase_cpu[event["name"]][1] += 1

    priority = {"data_fetch": 0, "forward": 1, "backward": 1,
                "optimizer": 2, "grad_clip": 2, "update": 3}
    intervals = [(event["name"], float(event["ts"]),
                  float(event["ts"]) + float(event["dur"])) for event in phases]
    external = {}
    for event in cpu:
        external_id = (event.get("args") or {}).get("External id")
        if external_id is not None:
            external.setdefault(external_id, float(event.get("ts", 0)))
    phase_gpu = defaultdict(float)
    unmapped = 0
    unmapped_us = 0.0
    for event in kernels:
        external_id = (event.get("args") or {}).get("External id")
        timestamp = external.get(external_id)
        dur = float(event.get("dur", 0))
        if timestamp is None:
            unmapped += 1
            unmapped_us += dur
            continue
        best = None
        for name, low, high in intervals:
            if low <= timestamp <= high:
                if (best is None or priority[name] < priority[best[0]]
                        or (priority[name] == priority[best[0]]
                            and high - low < best[2] - best[1])):
                    best = (name, low, high)
        if best is None:
            unmapped += 1
            unmapped_us += dur
        else:
            phase_gpu[best[0]] += dur

    step_spans = [(float(e["ts"]), float(e["ts"]) + float(e["dur"]))
                  for e in annotations if (e.get("name") or "").startswith("ProfilerStep")]
    busy_us = 0.0
    span_us = 0.0
    for low, high in step_spans:
        span_us += high - low
        clipped = []
        for event in kernels:
            start = float(event.get("ts", 0))
            end = start + float(event.get("dur", 0))
            if end > low and start < high:
                clipped.append((max(start, low), min(end, high)))
        busy_us += union_seconds(clipped)

    attention_ops = Counter()
    gemm_ops = Counter()
    for event in cpu:
        name = event.get("name") or ""
        lowered = name.lower()
        if "attention" in lowered or "scaled_dot_product" in lowered:
            attention_ops[name] += 1
        if name in ("aten::mm", "aten::addmm", "aten::bmm", "aten::linear",
                    "aten::matmul"):
            gemm_ops[name] += 1

    result = {
        "trace": os.path.basename(os.path.dirname(path)),
        "trace_bytes": os.path.getsize(path),
        "events": len(events),
        "kernels": len(kernels),
        "kernels_per_update": len(kernels) / updates,
        "kernel_seconds_total": total_kernel_us / 1e6,
        "kernel_seconds_per_update": total_kernel_us / 1e6 / updates,
        "categories": {
            name: {"seconds": values[0] / 1e6,
                   "seconds_per_update": values[0] / 1e6 / updates,
                   "fraction": values[0] / total_kernel_us,
                   "kernels": values[1],
                   "kernels_per_update": values[1] / updates}
            for name, values in categories.items()},
        "launch_cpu_seconds_per_update": launch_us / 1e6 / updates,
        "phase_cpu_seconds_per_update": {
            name: values[0] / 1e6 / updates for name, values in phase_cpu.items()},
        "phase_gpu_seconds_per_update": {
            name: value / 1e6 / updates for name, value in phase_gpu.items()},
        "phase_gpu_fraction": {
            name: value / total_kernel_us for name, value in phase_gpu.items()},
        "unmapped_kernels": unmapped,
        "unmapped_fraction": unmapped_us / total_kernel_us,
        "gpu_busy_fraction_of_profiled_steps": busy_us / span_us
        if span_us else None,
        "profiled_step_span_seconds": span_us / 1e6,
        "top_kernels": [
            {"category": category, "name": name[:120],
             "milliseconds_per_update": values[0] / 1e3 / updates,
             "kernels_per_update": values[1] / updates}
            for (category, name), values in sorted(
                kernel_names.items(), key=lambda item: -item[1][0])[:12]],
        "attention_ops": dict(attention_ops.most_common(12)),
        "gemm_ops": dict(gemm_ops.most_common(12)),
    }

    if shapes:
        ext_kernel_time = defaultdict(float)
        for event in kernels:
            external_id = (event.get("args") or {}).get("External id")
            if external_id is not None:
                ext_kernel_time[external_id] += float(event.get("dur", 0))
        shape_counts = Counter()
        shape_time = defaultdict(float)
        for event in cpu:
            name = event.get("name")
            if name not in ("aten::mm", "aten::addmm", "aten::bmm"):
                continue
            dims = (event.get("args") or {}).get("Input Dims")
            if not dims:
                continue
            key = name + " " + " | ".join(
                "x".join(str(v) for v in dim) if isinstance(dim, list) else str(dim)
                for dim in dims)
            shape_counts[key] += 1
            external_id = (event.get("args") or {}).get("External id")
            shape_time[key] += ext_kernel_time.get(external_id, 0.0)
        result["gemm_shapes"] = [
            {"shape": key, "calls_per_update": shape_counts[key] / updates,
             "ms_per_update": shape_time[key] / 1e3 / updates}
            for key in sorted(shape_time, key=lambda k: -shape_time[k])[:12]]

    return result


def window_stats(rows, low, high):
    times = [row[0] for row in rows]
    selected = [index for index, stamp in enumerate(times) if low <= stamp <= high]
    if not selected:
        return None
    powers = [rows[index][1] for index in selected]
    clocks = [rows[index][2] for index in selected]
    busy = [rows[index][3] for index in selected]
    mean_power = statistics.mean(powers) / 1e6
    return {
        "mean_power_w": mean_power,
        "energy_j": mean_power * (high - low),
        "seconds": high - low,
        "samples": len(selected),
        "mean_sclk_ghz": statistics.mean(clocks) / 1e9,
        "mean_busy_pct": statistics.mean(busy),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    baseline = {}
    pooled = []
    loader = read_json(os.path.join(SCRATCH, "loader_probe.json"))
    for rep in (1, 2, 3):
        run_dir = os.path.join(SCRATCH, "baseline", f"rep{rep}")
        result = read_json(os.path.join(run_dir, "baseline_result.json"))
        curve = [json.loads(line) for line in
                 open(os.path.join(run_dir, "curve.jsonl"), encoding="utf-8")]
        train = [row for row in curve if "step_seconds" in row]
        measured = [row for row in train if 11 <= row["step"] <= 60]
        seconds = [row["step_seconds"] for row in measured]
        targets = len(measured) * TARGETS_PER_UPDATE
        loader_seconds = 50 * loader["seconds"]["mean"]
        entry = {
            "updates": len(measured),
            "ce_targets": targets,
            "final_state_bytes": os.path.getsize(
                os.path.join(run_dir, "ckpt_state_step60.pt"))
            if os.path.exists(os.path.join(run_dir, "ckpt_state_step60.pt")) else None,
            "step_seconds": {
                "median": statistics.median(seconds),
                "mean": statistics.mean(seconds),
                "p95": percentile(seconds, 0.95),
                "min": min(seconds),
                "max": max(seconds),
                "sd": statistics.stdev(seconds),
            },
            "ce_targets_per_s_training": targets / sum(seconds),
            "ce_targets_per_s_including_loader": targets / (sum(seconds) + loader_seconds),
            "median_update_ce_targets_per_s": statistics.median(
                TARGETS_PER_UPDATE / value for value in seconds),
            "loop_wall_seconds_per_update": (
                train[-1]["elapsed_s"] - train[0]["elapsed_s"]) / (len(train) - 1),
            "loss_step1": train[0]["train_main_ce"],
            "grad_step1": train[0]["grad_norm"],
            "loss_last": train[-1]["train_main_ce"],
            "entropy_last": train[-1]["router"]["mean_entropy_norm"],
            "wall_seconds": result["wall_seconds"],
            "process_cpu_seconds": result["process_cpu_seconds"],
            "peak_allocated_gb": result["memory"]["peak_allocated_bytes"] / 1e9,
            "peak_reserved_gb": result["memory"]["peak_reserved_bytes"] / 1e9,
            "fingerprint_match": result["fingerprint_match"],
        }
        baseline[f"rep{rep}"] = entry
        pooled.extend(seconds)

    baseline["pooled"] = {
        "updates": len(pooled),
        "median": statistics.median(pooled),
        "mean": statistics.mean(pooled),
        "p95": percentile(pooled, 0.95),
        "sd": statistics.stdev(pooled),
        "ce_targets_per_s_training": (len(pooled) * TARGETS_PER_UPDATE) / sum(pooled),
        "rep_median_spread": max(baseline[f"rep{rep}"]["step_seconds"]["median"]
                                 for rep in (1, 2, 3))
        - min(baseline[f"rep{rep}"]["step_seconds"]["median"] for rep in (1, 2, 3)),
    }
    baseline["run_peak"] = {
        "allocated_gb": max(baseline[f"rep{rep}"]["peak_allocated_gb"] for rep in (1, 2, 3)),
        "reserved_gb": max(baseline[f"rep{rep}"]["peak_reserved_gb"] for rep in (1, 2, 3)),
    }

    profiles = {}
    for trace in (1, 2):
        run_dir = os.path.join(SCRATCH, "profile", f"trace{trace}")
        result = read_json(os.path.join(run_dir, "profile_result.json"))
        parsed = parse_trace(os.path.join(run_dir, "trace.json"), UPDATES,
                             shapes=(trace == 2))
        parsed.update({
            "record_shapes": result["record_shapes"],
            "profiled_block_seconds": result["profiled_block_seconds"],
            "measured_step_seconds": [row["step_seconds"] for row in result["measured"]],
            "measured_data_seconds": [row["data_seconds"] for row in result["measured"]],
            "median_measured_step_seconds": statistics.median(
                row["step_seconds"] for row in result["measured"]),
            "peak_reserved_gb": result["memory"]["peak_reserved_bytes"] / 1e9,
            "warmup_losses": [row["main_ce"] for row in result["warmup"]],
            "warmup_grads": [row["grad_norm"] for row in result["warmup"]],
            "warmup_entropy": [row["router_mean_entropy_norm"] for row in result["warmup"]],
            "warmup_step_seconds": [row["step_seconds"] for row in result["warmup"]],
        })
        profiles[f"trace{trace}"] = parsed
        del parsed

    telemetry_rows = []
    with open(os.path.join(SCRATCH, "telemetry.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            telemetry_rows.append((
                float(row["monotonic"]),
                int(row["power_uw"]) if row["power_uw"] else 0,
                int(row["sclk_hz"]) if row["sclk_hz"] else 0,
                int(row["busy_pct"]) if row["busy_pct"] else 0,
                int(row["temp_millic"]) if row["temp_millic"] else 0,
                row["kfd_pids"].split(),
                row["wall_utc"],
            ))
    power_values = [row[1] for row in telemetry_rows]
    clocks = [row[2] for row in telemetry_rows]
    pids = set()
    for row in telemetry_rows:
        pids.update(row[5])
    energy = {
        "sensor": "/sys/class/drm/card1/device/hwmon/hwmon3/power1_average",
        "sensor_label": "PPT",
        "units": "microwatt",
        "sample_interval_s": 0.1,
        "samples": len(telemetry_rows),
        "power_w": {"min": min(power_values) / 1e6,
                    "median": statistics.median(power_values) / 1e6,
                    "max": max(power_values) / 1e6,
                    "mean": statistics.mean(power_values) / 1e6},
        "sclk_ghz": {"min": min(clocks) / 1e9,
                     "median": statistics.median(clocks) / 1e9,
                     "max": max(clocks) / 1e9},
        "kfd_pids_seen": sorted(pids),
        "scope": "GPU package power only; whole-system wall power was not measured",
    }

    windows = {}
    total_energy = 0.0
    total_targets = 0
    for trace in (1, 2):
        result = read_json(os.path.join(SCRATCH, "profile", f"trace{trace}",
                                        "profile_result.json"))
        for label in ("warmup", "measured"):
            per_update = []
            for row in result[label]:
                stats = window_stats(telemetry_rows, row["t_start"], row["t_end"])
                if stats:
                    per_update.append(stats)
            energy_block = sum(item["energy_j"] for item in per_update)
            targets = len(per_update) * TARGETS_PER_UPDATE
            windows[f"trace{trace}_{label}"] = {
                "updates": len(per_update),
                "ce_targets": targets,
                "energy_j": energy_block,
                "j_per_million_ce_targets": (
                    energy_block / (targets / 1e6)) if targets else None,
                "mean_power_w": statistics.mean(item["mean_power_w"]
                                                for item in per_update)
                if per_update else None,
                "mean_sclk_ghz": statistics.mean(item["mean_sclk_ghz"]
                                                 for item in per_update)
                if per_update else None,
                "mean_busy_pct": statistics.mean(item["mean_busy_pct"]
                                                 for item in per_update)
                if per_update else None,
            }
            if label == "warmup":
                total_energy += energy_block
                total_targets += targets
    energy["measured_windows"] = windows
    energy["j_per_million_ce_targets_unprofiled_warmup"] = (
        total_energy / (total_targets / 1e6)) if total_targets else None

    # Align each baseline measured window to wall time using the run's internal
    # elapsed clock. t_start is derived from the result write time minus the
    # non-training tail (wall minus the internal stop elapsed). This carries a
    # one-to-two second alignment uncertainty, noted in the report.
    baseline_energy = {}
    for rep in (1, 2, 3):
        result = read_json(os.path.join(SCRATCH, "baseline", f"rep{rep}",
                                        "baseline_result.json"))
        curve = [json.loads(line) for line in
                 open(os.path.join(SCRATCH, "baseline", f"rep{rep}", "curve.jsonl"),
                      encoding="utf-8")]
        train = [row for row in curve if "step_seconds" in row]
        stop = [row for row in curve if row.get("event") == "stopped"][0]
        end_mono = None
        for row in telemetry_rows:
            if row[6] >= result["time_utc"]:
                end_mono = row[0]
                break
        if end_mono is None:
            continue
        start_mono = end_mono - result["wall_seconds"]
        low = start_mono + [r for r in train if r["step"] == 10][0]["elapsed_s"]
        high = start_mono + [r for r in train if r["step"] == 60][0]["elapsed_s"]
        selected = [row for row in telemetry_rows if low <= row[0] <= high]
        if not selected:
            continue
        powers = [row[1] / 1e6 for row in selected]
        clocks = [row[2] / 1e9 for row in selected]
        busy = [row[3] for row in selected]
        mean_power = statistics.mean(powers)
        block = mean_power * (high - low)
        targets = 50 * TARGETS_PER_UPDATE
        baseline_energy[f"rep{rep}"] = {
            "seconds": high - low,
            "samples": len(selected),
            "mean_power_w": mean_power,
            "energy_j": block,
            "j_per_million_ce_targets": block / (targets / 1e6),
            "mean_sclk_ghz": statistics.mean(clocks),
            "mean_busy_pct": statistics.mean(busy),
        }
    if baseline_energy:
        energy["baseline_aligned"] = baseline_energy
        energy["baseline_j_per_million_ce_targets"] = statistics.mean(
            entry["j_per_million_ce_targets"] for entry in baseline_energy.values())
        energy["baseline_mean_power_w"] = statistics.mean(
            entry["mean_power_w"] for entry in baseline_energy.values())
        energy["baseline_mean_sclk_ghz"] = statistics.mean(
            entry["mean_sclk_ghz"] for entry in baseline_energy.values())
    if len(telemetry_rows) > 1:
        run_energy = sum(
            (telemetry_rows[i + 1][0] - telemetry_rows[i][0])
            * (telemetry_rows[i + 1][1] + telemetry_rows[i][1]) / 2 / 1e6
            for i in range(len(telemetry_rows) - 1))
        energy["whole_run_gpu_energy_j"] = run_energy
        energy["whole_run_seconds"] = telemetry_rows[-1][0] - telemetry_rows[0][0]

    loader = read_json(os.path.join(SCRATCH, "loader_probe.json"))

    summary = {        "meta": {
            "experiment": "v3.1-PERF-1 synchronized baseline",
            "date_utc": "2026-10-27",
            "mode": "Stage 1 measurement only; no optimization",
            "updates_total": 180 + 20,
            "supervisor": os.path.join(SCRATCH, "supervisor.sh"),
            "git_head_before": "f71d027",
        },
        "baseline": baseline,
        "loader": {
            "method": "iterate 8 source-pure windows per update with the frozen loader, no model",
            "windows_measured": loader["windows"],
            "median_ms": loader["seconds"]["median"] * 1000,
            "mean_ms": loader["seconds"]["mean"] * 1000,
            "p95_ms": loader["seconds"]["p95"] * 1000,
            "max_ms": loader["seconds"]["max"] * 1000,
        },
        "profiler": profiles,
        "energy": energy,
        "reproduction": {
            "fingerprint_match": baseline["rep1"]["fingerprint_match"],
            "step1_loss_v31l": 9.1219,
            "step1_loss_measured": baseline["rep1"]["loss_step1"],
            "step1_grad_v31l": 1.4846,
            "step1_grad_measured": baseline["rep1"]["grad_step1"],
            "harness_warmup_losses": profiles["trace1"]["warmup_losses"],
            "harness_warmup_grads": profiles["trace1"]["warmup_grads"],
            "peak_reserved_v31l_gb": 6.107,
            "peak_reserved_measured_gb": baseline["run_peak"]["reserved_gb"],
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("wrote", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
