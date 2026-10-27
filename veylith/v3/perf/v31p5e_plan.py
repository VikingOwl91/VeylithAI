"""PERF-5E runner: one bounded GPU reproduction with tensor observation.

Fresh process, idle-gated, `ulimit -c 0`, 120 s timeout, no serialization, no
backend switch, no retry. Writes the case record, then runs the analyzer.
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

SCRATCH = os.path.expanduser("~/veylith-perf5e-20261027")
PYTHON = os.path.expanduser("~/veylith-v31-isolated/bin/python")
REPO = os.path.expanduser("~/git/AI-Research-v2dev")
WATCH = os.path.join(REPO, "veylith/v3/perf/v31p5e_tensorwatch.py")
TIMEOUT_S = 120
REPRO_ARGS = ["--no_optimizer", "--grad_filter", "experts", "--iters", "100"]

_FAULT = re.compile(r"Memory access fault by GPU node-\d+ .*? on address (0x[0-9a-fA-F]+)")
_ILLEGAL = re.compile(r"illegal memory access|hipErrorIllegalAddress|AcceleratorError")
_RESET = re.compile(r"(gpu reset|GPU reset|ring \S+ reset|amdgpu_device_gpu_recover|device wedged|GPU hang)")


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("one GPU execution; rolling profiler with record_shapes=True plus a")
        print("TorchDispatchMode recording mm/copy_ metadata; no serialization")
        return 0
    if not args.execute:
        parser.error("choose --dry-run or --execute")

    case = os.path.join(SCRATCH, "case")
    os.makedirs(case, exist_ok=True)
    pids = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                          text=True, timeout=20).stdout
    if "No KFD PIDs currently running" not in pids:
        raise SystemExit(f"REFUSED: GPU not idle: {pids.strip()}")
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "ROCM_PATH",
                        "HIP_PATH", "HIPBLASLT_LOG_LEVEL", "AMD_SERIALIZE_KERNEL",
                        "AMD_SERIALIZE_COPY")}
    env["PYTHONNOUSERSITE"] = "1"
    env["HIPBLASLT_LOG_MASK"] = "10"
    env["HIPBLASLT_LOG_FILE"] = os.path.join(case, "hipblaslt-%i.log")
    argv = ([PYTHON, WATCH, "--mode", "repro", "--out-dir", case, "--repro-args"]
            + REPRO_ARGS)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        proc = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True,
                              timeout=TIMEOUT_S, preexec_fn=no_core)
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        code = None
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    time.sleep(3)
    pids_after = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                                text=True, timeout=20).stdout
    try:
        window = subprocess.run(["journalctl", "-k", "--since", started, "--utc",
                                 "--no-pager"], capture_output=True, text=True,
                                timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        window = f"journalctl unavailable: {exc!r}"
    open(os.path.join(case, "stdout.txt"), "w").write(out)
    open(os.path.join(case, "stderr.txt"), "w").write(err)
    open(os.path.join(case, "kernel-window.txt"), "w").write(window)
    fault = _FAULT.search(err)
    signum = -code if code is not None and code < 0 else None
    mode = ("gpuvm-abort" if fault else "sigabrt" if code == 134 or signum == 6
            else "hip-illegal-address" if _ILLEGAL.search(err) else None)
    record = {
        "execution": 1, "of_max": 1,
        "faulted": mode is not None, "fault_mode": mode,
        "fault_address": fault.group(1) if fault else None,
        "exit_code": code, "signal": signum,
        "kernel_fault_addresses": sorted(set(re.findall(
            r"in page starting at address (0x[0-9a-fA-F]+)", window))),
        "kernel_fault_records": len(re.findall(r"page fault", window)),
        "reset_lines": [ln.strip()[:200] for ln in window.splitlines()
                        if _RESET.search(ln)],
        "gpu_idle_after": "No KFD PIDs currently running" in pids_after,
        "serialization": "none", "record_shapes": True,
        "started_utc": started,
    }
    json.dump(record, open(os.path.join(case, "record.json"), "w"), indent=2,
              sort_keys=True)
    print("EXECUTION " + json.dumps({k: record[k] for k in
                                     ("faulted", "fault_mode", "exit_code",
                                      "kernel_fault_addresses", "gpu_idle_after")}))
    analysis = subprocess.run([PYTHON, os.path.join(REPO, "veylith/v3/perf/v31p5e_analyze.py"),
                               "--case", case], capture_output=True, text=True,
                              timeout=300)
    print(analysis.stdout[-4000:])
    if analysis.returncode != 0:
        print("analysis stderr:", analysis.stderr[-1500:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
