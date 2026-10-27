"""PERF-5C runner: one reference reproduction, one attribution reproduction.

Two GPU executions maximum, each in a fresh process, crash-contained and
idle-gated. No retries, no resets, no production changes.

  --dry-run   print the plan and the instrumentation rationale; no CUDA
  --execute   run the two cases and write the records
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

SCRATCH = os.path.expanduser("~/veylith-perf5c-20261027")
PYTHON = os.path.expanduser("~/veylith-v31-isolated/bin/python")
REPO = os.path.expanduser("~/git/AI-Research-v2dev")
ATTR = os.path.join(REPO, "veylith/v3/perf/v31p5c_attr.py")
MAX_EXECUTIONS = 2
TIMEOUT_S = 120
REPRO_ARGS = ["--no_optimizer", "--grad_filter", "experts", "--iters", "100"]

_FAULT = re.compile(r"Memory access fault by GPU node-\d+ .*? on address (0x[0-9a-fA-F]+)")
_ILLEGAL = re.compile(r"illegal memory access|hipErrorIllegalAddress|AcceleratorError")
_RESET = re.compile(r"(gpu reset|GPU reset|ring \S+ reset|amdgpu_device_gpu_recover|device wedged|GPU hang)")
_TRACE = re.compile(r"\[HIPBLASLT\]\[(\d+)\]\[Trace\]\[rocblaslt_matmul\]")


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def gpu_state():
    state = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pids = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                          text=True, timeout=20).stdout
    state["showpids"] = pids.strip()
    state["idle"] = "No KFD PIDs currently running" in pids
    return state


def kernel_window(since):
    try:
        return subprocess.run(["journalctl", "-k", "--since", since, "--utc",
                               "--no-pager"], capture_output=True, text=True,
                              timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        return f"journalctl unavailable: {exc!r}"


def classify(err, code):
    if _FAULT.search(err):
        return True, "gpuvm-abort", _FAULT.search(err).group(1)
    if code == 134:
        return True, "sigabrt", None
    if _ILLEGAL.search(err):
        return True, "hip-illegal-address", None
    return False, None, None


def run_case(name, mode, extra_env, expect_fault, state):
    if state["executions"] >= MAX_EXECUTIONS:
        raise SystemExit("budget exhausted")
    before = gpu_state()
    if not before["idle"]:
        raise SystemExit(f"REFUSED: GPU not idle before {name}")
    case_dir = os.path.join(SCRATCH, "cases", name)
    os.makedirs(case_dir, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "ROCM_PATH",
                        "HIP_PATH", "HIPBLASLT_LOG_LEVEL", "AMD_SERIALIZE_KERNEL",
                        "AMD_SERIALIZE_COPY")}
    env["PYTHONNOUSERSITE"] = "1"
    env["HIPBLASLT_LOG_MASK"] = "10"
    env["HIPBLASLT_LOG_FILE"] = os.path.join(case_dir, "hipblaslt-%i.log")
    env.update(extra_env)
    argv = ([PYTHON, ATTR, "--mode", mode,
             "--out", os.path.join(case_dir, "wrapper.json"),
             "--trace", os.path.join(case_dir, "trace.json"),
             "--repro-args"] + REPRO_ARGS)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["executions"] += 1
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
    faulted, mode_name, addr = classify(err, code)
    wrapper = {}
    wrapper_path = os.path.join(case_dir, "wrapper.json")
    if os.path.exists(wrapper_path):
        wrapper = json.load(open(wrapper_path))
    gemm_calls = 0
    log = next((os.path.join(case_dir, n) for n in os.listdir(case_dir)
                if n.startswith("hipblaslt-")), None)
    last_gemm = None
    if log and os.path.exists(log):
        text = open(log, errors="replace").read()
        lines = [ln for ln in text.splitlines() if _TRACE.search(ln)]
        gemm_calls = len(lines)
        if lines:
            last_gemm = lines[-1][:400]
    record = {
        "name": name, "mode": mode, "expected": "fault" if expect_fault else "any",
        "extra_env": extra_env, "faulted": faulted, "fault_mode": mode_name,
        "fault_address": addr, "exit_code": code, "timeout": timeout,
        "reset_lines": [ln.strip()[:200] for ln in window.splitlines()
                        if _RESET.search(ln)],
        "gpu_before": before, "gpu_after": after,
        "wrapper": wrapper, "gemm_calls": gemm_calls, "last_gemm_line": last_gemm,
        "started_utc": started, "finished_utc": finished, "case_dir": case_dir,
        "kernel_fault_records": len(re.findall(r"page fault", window)),
        "kernel_fault_addresses": sorted(set(re.findall(
            r"in page starting at address (0x[0-9a-fA-F]+)", window))),
    }
    json.dump(record, open(os.path.join(case_dir, "record.json"), "w"),
              indent=2, sort_keys=True)
    print(f"[{name}] exit={code} fault={faulted} mode={mode_name} "
          f"kernels={(wrapper.get('summary') or {}).get('kernel_events')} "
          f"gemm_calls={gemm_calls}", flush=True)
    return record


def dry_run():
    print(f"budget {MAX_EXECUTIONS} GPU executions, timeout {TIMEOUT_S}s, fresh process each\n")
    print("case 1 baseline   : reproducer as in PERF-5B, hipBLASLt Trace logging, no serialization")
    print("case 2 attribution: same, plus AMD_SERIALIZE_KERNEL=3 and torch.profiler with")
    print("                    CUDA activity; trace exported from the exception path")
    print("\nwhy serialization: the fault surfaces as a sticky HIP error at the next API call,")
    print("so without serialization several kernels can be in flight; with one dispatch in")
    print("flight the last recorded kernel is the faulting one, and the trace's 'External id'")
    print("links it to the aten op that launched it.")
    print("\nnot used: rocprofv3 (cannot finalize after a GPU fault), hipBLASLt solution ids")
    print("(absent from this build), backend comparison (independence not yet demonstrated).")
    return 0


def execute():
    os.makedirs(os.path.join(SCRATCH, "cases"), exist_ok=True)
    state = {"executions": 0}
    baseline = run_case("baseline", "plain", {}, True, state)
    if baseline["timeout"] or baseline["reset_lines"] or not baseline["gpu_after"]["idle"]:
        raise SystemExit("baseline unhealthy; stopping")
    if not baseline["faulted"]:
        raise SystemExit("baseline did not fault; reference not established, stopping")

    attribution = run_case("attribution", "profiled",
                           {"AMD_SERIALIZE_KERNEL": "3"}, True, state)
    summary = {
        "executions": state["executions"],
        "baseline": {k: baseline[k] for k in ("faulted", "fault_mode", "fault_address",
                                              "exit_code", "kernel_fault_records",
                                              "kernel_fault_addresses", "gemm_calls")},
        "attribution": {k: attribution[k] for k in ("faulted", "fault_mode",
                                                    "fault_address", "exit_code",
                                                    "timeout", "gemm_calls",
                                                    "kernel_fault_records",
                                                    "kernel_fault_addresses")},
    }
    if attribution["faulted"] and attribution["wrapper"].get("summary"):
        tail = attribution["wrapper"]["summary"]["tail"]
        summary["attribution"]["trace_kernel_events"] = attribution["wrapper"]["summary"]["kernel_events"]
        summary["attribution"]["tail"] = tail[-3:]
        summary["attribution"]["last_kernel"] = tail[-1]
    else:
        summary["attribution"]["note"] = "no fault under instrumentation, or no trace"
    json.dump(summary, open(os.path.join(SCRATCH, "attribution.json"), "w"),
              indent=2, sort_keys=True)
    print("\nATTRIBUTION " + json.dumps(summary["attribution"], indent=1)[:2000], flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return dry_run()
    if args.execute:
        return execute()
    parser.error("choose --dry-run or --execute")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
