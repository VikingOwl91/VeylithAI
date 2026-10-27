"""PERF-5F runner: one case per invocation, hard budget, explicit stop rule.

  A  reference reproduction with the PERF-5D rolling profiler (no shapes)
  B  standalone bf16->fp32 cast on the boundary shape
  C  reference reproduction with allocator rounding (block slack)
  D  reserved

Refuses B or C unless A actually faulted, so later clean results are never
interpreted against an unverified reference. Records every execution in
executions.json and refuses beyond four.
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

SCRATCH = os.path.expanduser("~/veylith-perf5f-20261027")
PYTHON = os.path.expanduser("~/veylith-v31-isolated/bin/python")
REPO = os.path.expanduser("~/git/AI-Research-v2dev")
ROLLING = os.path.join(REPO, "veylith/v3/perf/v31p5d_rolling.py")
CAST = os.path.join(REPO, "veylith/v3/perf/v31p5f_cast.py")
TIMEOUT_S = 120
MAX_EXECUTIONS = 4
REPRO_ARGS = ["--no_optimizer", "--grad_filter", "experts", "--iters", "100"]
MIB = 2 << 20

_FAULT = re.compile(r"Memory access fault by GPU node-\d+ .*? on address (0x[0-9a-fA-F]+)")
_ILLEGAL = re.compile(r"illegal memory access|hipErrorIllegalAddress|AcceleratorError")
_RESET = re.compile(r"(gpu reset|GPU reset|ring \S+ reset|amdgpu_device_gpu_recover|device wedged|GPU hang)")
_LAY = re.compile(r"Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)")
_PTR = re.compile(r"C=(0x[0-9a-fA-F]+)")


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def budget():
    path = os.path.join(SCRATCH, "executions.json")
    os.makedirs(SCRATCH, exist_ok=True)
    if os.path.exists(path):
        return json.load(open(path))
    return {"executions": []}


def save_budget(state):
    json.dump(state, open(os.path.join(SCRATCH, "executions.json"), "w"),
              indent=2, sort_keys=True)


def gpu_idle():
    out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                         text=True, timeout=20).stdout
    return "No KFD PIDs currently running" in out, out.strip()


def geometry(case_dir, fault_pages):
    log = glob.glob(os.path.join(case_dir, "hipblaslt-*.log"))
    if not log:
        return {"available": False}
    calls = []
    for line in open(log[0], errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        m, p = _LAY.search(line), _PTR.search(line)
        if m and p:
            calls.append({"C": int(p.group(1), 16),
                          "extent": int(m.group(2)) * int(m.group(3)) * 2,
                          "rows": int(m.group(2)), "cols": int(m.group(3))})
    wide = [c for c in calls if c["cols"] in (3075, 682) or c["rows"] == 682]
    rows = []
    for call in wide:
        end = call["C"] + call["extent"]
        next_page = ((end + MIB - 1) // MIB) * MIB
        rows.append({"C": hex(call["C"]), "rows": call["rows"], "cols": call["cols"],
                     "extent": call["extent"], "tensor_end": hex(end),
                     "gap_to_next_page": next_page - end})
    tight = [r for r in rows if r["gap_to_next_page"] <= 64]
    return {"available": True, "calls": len(calls), "wide_calls": len(rows),
            "min_gap_to_page": min((r["gap_to_next_page"] for r in rows), default=None),
            "calls_within_64B_of_page": len(tight),
            "examples_tight": tight[:3],
            "fault_pages": fault_pages}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    state = budget()
    if args.case != "A" and not args.execute:
        pass
    if args.dry_run:
        print(f"case {args.case}; executions so far {len(state['executions'])}")
        print("A reference with the 5D rolling profiler; B standalone cast on the boundary shape;")
        print("C reference with PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:4; D reserved")
        return 0
    if not args.execute:
        parser.error("choose --dry-run or --execute")

    if len(state["executions"]) >= MAX_EXECUTIONS:
        raise SystemExit("REFUSED: four executions already used")
    if args.case in ("B", "C", "D"):
        reference = next((e for e in state["executions"] if e["case"] == "A"), None)
        if reference is None:
            raise SystemExit("REFUSED: run case A first")
        if not reference.get("faulted"):
            raise SystemExit("REFUSED: case A did not fault; stop and reassess")

    idle, pids = gpu_idle()
    if not idle:
        raise SystemExit(f"REFUSED: GPU not idle: {pids}")

    case_dir = os.path.join(SCRATCH, f"case_{args.case}")
    os.makedirs(case_dir, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "ROCM_PATH",
                        "HIP_PATH", "HIPBLASLT_LOG_LEVEL", "AMD_SERIALIZE_KERNEL",
                        "AMD_SERIALIZE_COPY", "PYTORCH_HIP_ALLOC_CONF",
                        "PYTORCH_CUDA_ALLOC_CONF")}
    env["PYTHONNOUSERSITE"] = "1"
    env["HIPBLASLT_LOG_MASK"] = "10"
    env["HIPBLASLT_LOG_FILE"] = os.path.join(case_dir, "hipblaslt-%i.log")
    if args.case == "A":
        argv = [PYTHON, ROLLING, "--mode", "repro", "--out-dir", case_dir,
                "--repro-args"] + REPRO_ARGS
    elif args.case == "B":
        argv = [PYTHON, CAST, "--out", os.path.join(case_dir, "cast.json")]
    elif args.case == "C":
        env["PYTORCH_HIP_ALLOC_CONF"] = "roundup_power2_divisions:4"
        argv = [PYTHON, ROLLING, "--mode", "repro", "--out-dir", case_dir,
                "--repro-args"] + REPRO_ARGS
    else:
        raise SystemExit("case D is not defined yet")

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
    idle_after, pids_after = gpu_idle()
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
    record = {
        "case": args.case, "argv": argv,
        "env_padding": env.get("PYTORCH_HIP_ALLOC_CONF"),
        "faulted": mode is not None, "fault_mode": mode,
        "fault_address": fault.group(1) if fault else None,
        "exit_code": code, "signal": signum,
        "kernel_fault_addresses": pages,
        "kernel_fault_records": len(re.findall(r"page fault", window)),
        "reset_lines": [ln.strip()[:200] for ln in window.splitlines() if _RESET.search(ln)],
        "gpu_idle_after": idle_after,
        "geometry": geometry(case_dir, pages),
        "started_utc": started,
    }
    json.dump(record, open(os.path.join(case_dir, "record.json"), "w"), indent=2,
              sort_keys=True)
    state["executions"].append({"case": args.case, "faulted": record["faulted"],
                                "fault_mode": mode, "fault_address": record["fault_address"],
                                "min_gap_to_page": record["geometry"].get("min_gap_to_page"),
                                "utc": started})
    save_budget(state)
    print(f"CASE {args.case} " + json.dumps({k: record[k] for k in
                                            ("faulted", "fault_mode", "exit_code",
                                             "kernel_fault_addresses", "gpu_idle_after")}))
    print("geometry " + json.dumps(record["geometry"])[:600])
    print(f"executions used {len(state['executions'])}/{MAX_EXECUTIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
