"""PERF-5G runner: reference (A) then padded (B), two executions maximum.

A must reproduce the fault before B runs. Each execution is a fresh process with
the idle gate, `ulimit -c 0`, a 120 s timeout, kernel-log capture and a health
check afterwards. No serialization, no retries.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import resource
import subprocess
import time

SCRATCH = os.path.expanduser("~/veylith-perf5g-20261027")
PYTHON = os.path.expanduser("~/veylith-v31-isolated/bin/python")
REPO = os.path.expanduser("~/git/AI-Research-v2dev")
PAD = os.path.join(REPO, "veylith/v3/perf/v31p5g_pad.py")
TIMEOUT_S = 120
REPRO_ARGS = ["--no_optimizer", "--grad_filter", "experts", "--iters", "100"]
MIB = 2 << 20

_FAULT = re.compile(r"Memory access fault by GPU node-\d+ .*? on address (0x[0-9a-fA-F]+)")
_ILLEGAL = re.compile(r"illegal memory access|hipErrorIllegalAddress|AcceleratorError")
_RESET = re.compile(r"(gpu reset|GPU reset|ring \S+ reset|amdgpu_device_gpu_recover|device wedged|GPU hang)")
_LAY = re.compile(r"Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)")
_PTR = re.compile(r"C=(0x[0-9a-fA-F]+)")


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def idle():
    out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True,
                         timeout=20).stdout
    return "No KFD PIDs currently running" in out, out.strip()


def run(case, pad):
    ok, pids = idle()
    if not ok:
        raise SystemExit(f"REFUSED: GPU not idle: {pids}")
    case_dir = os.path.join(SCRATCH, f"case_{case}")
    os.makedirs(case_dir, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "ROCM_PATH",
                        "HIP_PATH", "HIPBLASLT_LOG_LEVEL", "AMD_SERIALIZE_KERNEL",
                        "AMD_SERIALIZE_COPY")}
    env["PYTHONNOUSERSITE"] = "1"
    env["HIPBLASLT_LOG_MASK"] = "10"
    env["HIPBLASLT_LOG_FILE"] = os.path.join(case_dir, "hipblaslt-%i.log")
    env["V31P5G_PRINT_ALLOC"] = "1" if pad else "0"
    argv = [PYTHON, PAD, "--mode", "repro", "--pad", str(int(pad)),
            "--out", os.path.join(case_dir, "geometry.json"), "--repro-args"] + REPRO_ARGS
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
    ok_after, pids_after = idle()
    try:
        window = subprocess.run(["journalctl", "-k", "--since", started, "--utc",
                                 "--no-pager"], capture_output=True, text=True,
                                timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        window = f"journalctl unavailable: {exc!r}"
    open(os.path.join(case_dir, "stdout.txt"), "w").write(out)
    open(os.path.join(case_dir, "stderr.txt"), "w").write(err)
    open(os.path.join(case_dir, "kernel-window.txt"), "w").write(window)
    fault = _FAULT.search(err)
    signum = -code if code is not None and code < 0 else None
    mode = ("gpuvm-abort" if fault else "sigabrt" if code == 134 or signum == 6
            else "hip-illegal-address" if _ILLEGAL.search(err) else None)
    pages = sorted(set(re.findall(r"in page starting at address (0x[0-9a-fA-F]+)", window)))
    allocs = [json.loads(line[9:]) for line in out.splitlines() if line.startswith("PADALLOC ")]
    log = glob.glob(os.path.join(case_dir, "hipblaslt-*.log"))
    wide = []
    if log:
        for line in open(log[0], errors="replace"):
            if "rocblaslt_matmul" not in line:
                continue
            m, p = _LAY.search(line), _PTR.search(line)
            if m and p and int(m.group(3)) == 3075:
                c = int(p.group(1), 16)
                extent = int(m.group(2)) * int(m.group(3)) * 2
                wide.append({"C": hex(c), "rows": int(m.group(2)), "cols": int(m.group(3)),
                             "ld": int(m.group(4)), "extent": extent,
                             "gap_to_next_page": (((c + extent + MIB - 1) // MIB) * MIB)
                             - (c + extent)})
    record = {
        "case": case, "pad": bool(pad), "pad_mode": int(pad),
        "faulted": mode is not None, "fault_mode": mode,
        "fault_address": fault.group(1) if fault else None, "exit_code": code,
        "signal": signum, "kernel_fault_addresses": pages,
        "kernel_fault_records": len(re.findall(r"page fault", window)),
        "reset_lines": [ln for ln in window.splitlines() if _RESET.search(ln)],
        "gpu_idle_after": ok_after, "started_utc": started,
        "padded_allocs": len(allocs),
        "padded_slack_min": min((a["slack_after_tensor"] for a in allocs), default=None),
        "padded_all_contiguous": all(a["contiguous"] for a in allocs) if allocs else None,
        "padded_sample": allocs[:2],
        "wide_calls": len(wide),
        "wide_min_gap": min((w["gap_to_next_page"] for w in wide), default=None),
        "wide_descriptors": sorted({(w["rows"], w["cols"], w["ld"]) for w in wide})[:4],
        "wide_tight": [w for w in wide if w["gap_to_next_page"] <= 64][:3],
    }
    json.dump(record, open(os.path.join(case_dir, "record.json"), "w"), indent=2,
              sort_keys=True)
    print(f"CASE {case} " + json.dumps({k: record[k] for k in
                                        ("faulted", "fault_mode", "exit_code",
                                         "padded_allocs", "padded_slack_min",
                                         "wide_calls", "wide_min_gap",
                                         "wide_descriptors", "gpu_idle_after")}))
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("A reference (pad 0), then B padded (pad 1) only if A faults")
        return 0
    if not args.execute:
        parser.error("choose --dry-run or --execute")
    a = run("A", 0)
    if not a["faulted"]:
        print("STOP: reference did not fault; B not run")
        return 0
    b = run("B", 1)
    json.dump({"A": a, "B": b}, open(os.path.join(SCRATCH, "matrix.json"), "w"),
              indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
