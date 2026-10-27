"""PERF-5D: one bounded GPU reproduction with the rolling profiler.

Single execution, fresh process, idle-gated, `ulimit -c 0`, 120 s timeout, no
serialization, no retry, no reset. Writes the case record, then runs the trace
analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time

SCRATCH = os.path.expanduser("~/veylith-perf5d-20261027")
PYTHON = os.path.expanduser("~/veylith-v31-isolated/bin/python")
REPO = os.path.expanduser("~/git/AI-Research-v2dev")
ROLLING = os.path.join(REPO, "veylith/v3/perf/v31p5d_rolling.py")
TIMEOUT_S = 120
REPRO_ARGS = ["--no_optimizer", "--grad_filter", "experts", "--iters", "100"]

_FAULT = re.compile(r"Memory access fault by GPU node-\d+ .*? on address (0x[0-9a-fA-F]+)")
_ILLEGAL = re.compile(r"illegal memory access|hipErrorIllegalAddress|AcceleratorError")
_RESET = re.compile(r"(gpu reset|GPU reset|ring \S+ reset|amdgpu_device_gpu_recover|device wedged|GPU hang)")


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def gpu_state():
    out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True,
                         timeout=20).stdout
    return {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "showpids": out.strip(),
            "idle": "No KFD PIDs currently running" in out}


def kernel_window(since):
    try:
        return subprocess.run(["journalctl", "-k", "--since", since, "--utc",
                               "--no-pager"], capture_output=True, text=True,
                              timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        return f"journalctl unavailable: {exc!r}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("one GPU execution, 120 s timeout, no serialization, rolling profiler")
        print(f"schedule: wait=0 warmup=0 active=1 repeat=0 (infinite in this build)")
        print(f"step boundary: patched sparse_moe_v2_train.get_batch, before the fetch")
        print(f"argv: {[PYTHON, ROLLING, '--mode', 'repro', '--repro-args'] + REPRO_ARGS}")
        return 0
    if not args.execute:
        parser.error("choose --dry-run or --execute")

    case_dir = os.path.join(SCRATCH, "case")
    os.makedirs(case_dir, exist_ok=True)
    before = gpu_state()
    if not before["idle"]:
        raise SystemExit(f"REFUSED: GPU not idle: {before['showpids']}")
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "ROCM_PATH",
                        "HIP_PATH", "HIPBLASLT_LOG_LEVEL", "AMD_SERIALIZE_KERNEL",
                        "AMD_SERIALIZE_COPY")}
    env["PYTHONNOUSERSITE"] = "1"
    env["HIPBLASLT_LOG_MASK"] = "10"
    env["HIPBLASLT_LOG_FILE"] = os.path.join(case_dir, "hipblaslt-%i.log")
    argv = ([PYTHON, ROLLING, "--mode", "repro", "--out-dir", case_dir,
             "--repro-args"] + REPRO_ARGS)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    timeout = False
    try:
        proc = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True,
                              timeout=TIMEOUT_S, preexec_fn=no_core)
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timeout = True
        code = None
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    time.sleep(3)
    after = gpu_state()
    window = kernel_window(started)
    open(os.path.join(case_dir, "stdout.txt"), "w").write(out)
    open(os.path.join(case_dir, "stderr.txt"), "w").write(err)
    open(os.path.join(case_dir, "kernel-window.txt"), "w").write(window)

    fault = _FAULT.search(err)
    signum = -code if code is not None and code < 0 else None
    if fault:
        mode = "gpuvm-abort"
    elif code == 134 or signum == 6:
        mode = "sigabrt"
    elif _ILLEGAL.search(err):
        mode = "hip-illegal-address"
    else:
        mode = None
    rolling = {}
    rolling_path = os.path.join(case_dir, "rolling.json")
    if os.path.exists(rolling_path):
        rolling = json.load(open(rolling_path))
    traces = sorted(n for n in os.listdir(case_dir) if n.startswith("trace-cycle-"))
    record = {
        "execution": 1, "of_max": 1,
        "faulted": mode is not None, "fault_mode": mode,
        "fault_address": fault.group(1) if fault else None,
        "exit_code": code, "signal": signum, "timeout": timeout,
        "reset_lines": [ln.strip()[:200] for ln in window.splitlines() if _RESET.search(ln)],
        "gpu_before": before, "gpu_after": after,
        "kernel_fault_addresses": sorted(set(re.findall(
            r"in page starting at address (0x[0-9a-fA-F]+)", window))),
        "kernel_fault_records": len(re.findall(r"page fault", window)),
        "serialization": "none",
        "schedule": "wait=0 warmup=0 active=1 repeat=0",
        "iterations_started": rolling.get("iterations_started"),
        "cycles_completed": len(rolling.get("cycles") or []),
        "cycle_rows": rolling.get("cycles"),
        "step_errors": rolling.get("step_errors"),
        "exception": rolling.get("exception"),
        "trace_files": traces,
        "started_utc": started, "finished_utc": finished,
    }
    json.dump(record, open(os.path.join(case_dir, "record.json"), "w"),
              indent=2, sort_keys=True)
    print("EXECUTION " + json.dumps({k: record[k] for k in
                                     ("faulted", "fault_mode", "exit_code",
                                      "iterations_started", "cycles_completed",
                                      "kernel_fault_addresses")}))

    analysis = subprocess.run([PYTHON, os.path.join(REPO, "veylith/v3/perf/v31p5d_analyze.py"),
                               "--case", case_dir],
                              capture_output=True, text=True, timeout=300)
    print(analysis.stdout[-4000:])
    if analysis.returncode != 0:
        print("analysis stderr:", analysis.stderr[-2000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
