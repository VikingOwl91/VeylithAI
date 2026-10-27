"""PERF-5B matrix runner: one process per case, crash-contained, no retries.

Owns the execution budget, the idle gates, the per-case evidence and the stop
rules. Never resets the GPU, never kills unrelated processes, never retries a
case.

  --dry-run   print the plan, the allocation arithmetic and the descriptor
              prediction; no CUDA initialisation
  --execute   run the matrix
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

SCRATCH = os.path.expanduser("~/veylith-perf5b-20261027")
PYTHON = os.path.expanduser("~/veylith-v31-isolated/bin/python")
REPO = os.path.expanduser("~/git/AI-Research-v2dev")
CASE = os.path.join(REPO, "veylith/v3/perf/v31p5b_gemm_tail.py")
REPRO = os.path.join(REPO, "veylith/debug/reproducer.py")
REPRO_ROCB = os.path.join(REPO, "veylith/v3/perf/v31p5b_rocblas_repro.py")
MAX_EXECUTIONS = 8
TIMEOUT_S = 120

_FAULT = re.compile(r"Memory access fault by GPU node-\d+ .*? on address (0x[0-9a-fA-F]+)\. Reason: ([^\n]+)")
_HIP_ILLEGAL = re.compile(r"illegal memory access|hipErrorIllegalAddress|AcceleratorError")
_FAULT_KERNEL = re.compile(r"kernel: (\S+?)\]")
_SOL = re.compile(r"sol-idx = (\d+)")
_TRACE = re.compile(r"\[HIPBLASLT\]\[(\d+)\]\[Trace\]\[rocblaslt_matmul\]")
_LAYOUT = re.compile(r"(Adesc|Bdesc|Cdesc)=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)")
_TRANS = re.compile(r"transA=(\S+) transB=(\S+)")
_RESET = re.compile(r"(gpu reset|GPU reset|ring \S+ reset|amdgpu_device_gpu_recover|device wedged|GPU hang)")

sys.path.insert(0, REPO + "/veylith/v3")
sys.path.insert(0, REPO + "/veylith/v3/perf")


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def gpu_state():
    state = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        pids = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                              text=True, timeout=20).stdout
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        pids = f"rocm-smi failed: {exc!r}"
    state["showpids"] = pids.strip()
    state["idle"] = "No KFD PIDs currently running" in pids
    try:
        mem = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception as exc:  # noqa: BLE001
        mem = f"rocm-smi failed: {exc!r}"
    used = re.search(r"VRAM Total Used Memory \(B\): (\d+)", mem)
    total = re.search(r"VRAM Total Memory \(B\): (\d+)", mem)
    state["vram_used_gib"] = round(int(used.group(1)) / 2 ** 30, 3) if used else None
    state["vram_total_gib"] = round(int(total.group(1)) / 2 ** 30, 3) if total else None
    return state


def kernel_window(since: str) -> str:
    try:
        return subprocess.run(["journalctl", "-k", "--since", since, "--utc",
                               "--no-pager"], capture_output=True, text=True,
                              timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        return f"journalctl unavailable: {exc!r}"


def parse_hipblaslt_log(directory: str):
    solutions, descriptors, traces = {}, [], 0
    for name in sorted(os.listdir(directory)):
        if not name.startswith("hipblaslt-"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        for match in _SOL.finditer(text):
            solutions[match.group(1)] = solutions.get(match.group(1), 0) + 1
        for line in text.splitlines():
            if _TRACE.search(line):
                traces += 1
                if len(descriptors) < 3:
                    row = {m.group(1): {"type": m.group(2), "rows": int(m.group(3)),
                                        "cols": int(m.group(4)), "ld": int(m.group(5))}
                           for m in _LAYOUT.finditer(line)}
                    trans = _TRANS.search(line)
                    row["trans"] = [trans.group(1), trans.group(2)] if trans else None
                    descriptors.append(row)
    return {"trace_records": traces, "solution_histogram": solutions,
            "descriptors": descriptors}


def run_case(name: str, argv, expect: str, note: str, state) -> dict:
    if state["executions"] >= MAX_EXECUTIONS:
        raise SystemExit(f"budget exhausted before {name}")
    before = gpu_state()
    if not before["idle"]:
        raise SystemExit(f"REFUSED: GPU not idle before {name}: {before['showpids']}")
    case_dir = os.path.join(SCRATCH, "cases", name)
    os.makedirs(case_dir, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "ROCM_PATH",
                        "HIP_PATH", "HIPBLASLT_LOG_LEVEL")}
    env["PYTHONNOUSERSITE"] = "1"
    env["HIPBLASLT_LOG_MASK"] = "10"
    env["HIPBLASLT_LOG_FILE"] = os.path.join(case_dir, "hipblaslt-%i.log")
    env.update(state.get("extra_env", {}))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(case_dir, "gpu-before.txt"), "w") as handle:
        handle.write(json.dumps(before, indent=2) + "\n")
    state["executions"] += 1
    timeout = False
    try:
        proc = subprocess.run(argv, cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=TIMEOUT_S, preexec_fn=no_core)
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timeout = True
        code = None
        out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    time.sleep(3)  # let a faulting process fully release the device
    with open(os.path.join(case_dir, "stdout.txt"), "w") as handle:
        handle.write(out)
    with open(os.path.join(case_dir, "stderr.txt"), "w") as handle:
        handle.write(err)
    after = gpu_state()
    window = kernel_window(started)
    with open(os.path.join(case_dir, "gpu-after.txt"), "w") as handle:
        handle.write(json.dumps(after, indent=2) + "\n")
    with open(os.path.join(case_dir, "kernel-window.txt"), "w") as handle:
        handle.write(window)

    fault_match = _FAULT.search(err)
    signum = -code if code is not None and code < 0 else None
    if fault_match:
        fault_mode = "gpuvm-abort"
    elif code == 134 or signum == 6:
        fault_mode = "sigabrt"
    elif _HIP_ILLEGAL.search(err):
        fault_mode = "hip-illegal-address"
    else:
        fault_mode = None
    faulted = fault_mode is not None
    resets = [line.strip()[:200] for line in window.splitlines() if _RESET.search(line)]
    prep = {}
    for line in out.splitlines():
        if line.startswith("PREP "):
            prep = json.loads(line[5:])
    record = {
        "name": name, "argv": argv, "note": note, "expected": expect,
        "exit_code": code, "signal": signum, "timeout": timeout,
        "faulted": faulted, "fault_mode": fault_mode,
        "fault_address": fault_match.group(1) if fault_match else None,
        "fault_reason": fault_match.group(2) if fault_match else None,
        "kernel_in_stderr": (_FAULT_KERNEL.search(err).group(1)
                             if _FAULT_KERNEL.search(err) else None),
        "prep": prep,
        "hipblaslt": parse_hipblaslt_log(case_dir),
        "gpu_before": before, "gpu_after": after,
        "reset_lines": resets,
        "stdout_tail": "\n".join(out.splitlines()[-3:]),
        "started_utc": started, "finished_utc": finished,
        "case_dir": case_dir,
        "matches_expectation": faulted == (expect == "fault"),
    }
    with open(os.path.join(case_dir, "record.json"), "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    print(f"[{name}] exit={code} signal={signum} faulted={faulted} "
          f"expected={expect} match={record['matches_expectation']} "
          f"solutions={record['hipblaslt']['solution_histogram']}", flush=True)
    return record


def gemm_argv(dtype, n, k, placement, iters=2, backend="default", out=None,
              orientation="m-first"):
    return [PYTHON, CASE, "--dtype", dtype, "--n", str(n), "--k", str(k),
            "--placement", placement, "--iters", str(iters), "--backend", backend,
            "--orientation", orientation, "--out", out]


def dry_run() -> int:
    from v31p5b_gemm_tail import allocation_plan

    cases = [
        ("fp32_tight", "fp32", 682, 4108, "tight", "clean"),
        ("bf16_n640_tight", "bf16", 640, 4108, "tight", "clean"),
        ("bf16_k4160_tight", "bf16", 682, 4160, "tight", "clean"),
        ("bf16_target_tight", "bf16", 682, 4108, "tight", "fault-or-clean"),
        ("bf16_target_slack", "bf16", 682, 4108, "generous", "clean"),
    ]
    print(f"budget {MAX_EXECUTIONS}, timeout {TIMEOUT_S}s, one process per case\n")
    print("descriptor prediction (ATen mm with both operands as transposed views):")
    print("  transA=N transB=T  A rows=256 cols=4108 ld=256  B rows=682 cols=4108 ld=682")
    print("  C rows=256 cols=682 ld=256  dtype bf16  compute 32F  alpha 1 beta 0\n")
    for name, dtype, n, k, placement, expect in cases:
        itemsize = 2 if dtype == "bf16" else 4
        plan = allocation_plan(n, k, itemsize, placement)
        over = 2 * (64 - n % 64) if (n % 64 and k % 64) else 0
        print(f"{name:20s} dtype={dtype} n={n} k={k} placement={placement} "
              f"expect={expect}")
        print(f"  b_bytes={plan['b_bytes']} segment={plan['segment_bytes']} "
              f"offset={plan['b_offset']} slack={plan['slack_bytes']} "
              f"predicted_overread={over}  "
              f"(crosses boundary at {plan['slack_bytes']} bytes)"
              if plan["slack_bytes"] < over else
              f"  b_bytes={plan['b_bytes']} segment={plan['segment_bytes']} "
              f"offset={plan['b_offset']} slack={plan['slack_bytes']} "
              f"predicted_overread={over} (no crossing)")
    print("\nmodel positive control: veylith/debug/reproducer.py --no_optimizer "
          "--grad_filter experts --iters 100 (expect fault)")
    print("rocBLAS arm: same reproducer with preferred_blas_library('rocblas')")
    return 0


def execute() -> int:
    os.makedirs(os.path.join(SCRATCH, "cases"), exist_ok=True)
    state = {"executions": 0}
    results, decision = [], None
    unexpected = 0

    def guard(record):
        nonlocal unexpected
        if record["faulted"] and record["expected"] == "clean":
            unexpected += 1
        if record["timeout"]:
            raise SystemExit(f"timeout in {record['name']}; stopping")
        if record["reset_lines"]:
            raise SystemExit(f"GPU reset indicator after {record['name']}; stopping")
        if not record["gpu_after"]["idle"]:
            raise SystemExit(f"lingering KFD process after {record['name']}; stopping")
        if unexpected >= 2:
            raise SystemExit("two unexpected faults; stopping")

    control_specs = [
        ("fp32_tight", "fp32", 682, 4108, "tight", "clean"),
        ("bf16_n640_tight", "bf16", 640, 4108, "tight", "clean"),
        ("bf16_k4160_tight", "bf16", 682, 4160, "tight", "clean"),
    ]
    for name, dtype, n, k, placement, expect in control_specs:
        out = os.path.join(SCRATCH, "cases", name, "case.json")
        rec = run_case(name, gemm_argv(dtype, n, k, placement, out=out), expect,
                       "shape or dtype control", state)
        results.append(rec)
        guard(rec)

    out = os.path.join(SCRATCH, "cases", "bf16_target_tight", "case.json")
    target = run_case("bf16_target_tight", gemm_argv("bf16", 682, 4108, "tight", out=out),
                      "fault-or-clean", "the K-tail boundary case", state)
    results.append(target)

    target_solutions = target["hipblaslt"]["solution_histogram"]
    attributed = "107018" in target_solutions

    if target["faulted"]:
        out = os.path.join(SCRATCH, "cases", "bf16_target_slack", "case.json")
        slack = run_case("bf16_target_slack",
                         gemm_argv("bf16", 682, 4108, "generous", out=out),
                         "clean", "boundary discriminator", state)
        results.append(slack)
        guard(slack)
        if slack["faulted"]:
            decision = {"option": 2, "why": "target faults but the slack control also faults, so boundary sensitivity is not established"}
        elif attributed:
            decision = {"option": 1, "why": "tight faults and slack is clean; the faulting call selected solution 107018"}
        else:
            decision = {"option": 2, "why": "tight faults and slack is clean, but the selected solution is not 107018"}
    else:
        rec = run_case("positive_control",
                       [PYTHON, REPRO, "--no_optimizer", "--grad_filter", "experts",
                        "--iters", "100"], "fault",
                       "known-positive historical model reproducer", state)
        results.append(rec)
        if rec["faulted"]:
            decision = {"option": 4, "why": "synthetic target clean while the model reproducer still faults; the single-GEMM route does not reproduce it, so kernel attribution stays unresolved"}
        else:
            decision = {"option": 4, "why": "synthetic target and model reproducer both clean; environment drift, no conclusion"}

    summary = {"executions": state["executions"], "cases": results,
               "decision": decision, "attributed_to_107018": attributed,
               "unexpected_faults": unexpected}
    with open(os.path.join(SCRATCH, "matrix.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("\nDECISION " + json.dumps(decision), flush=True)
    print(f"executions used: {state['executions']}/{MAX_EXECUTIONS}", flush=True)
    return 0


def classify_text(err, code, signum):
    if _FAULT.search(err):
        return True, "gpuvm-abort"
    if code == 134 or signum == 6:
        return True, "sigabrt"
    if _HIP_ILLEGAL.search(err):
        return True, "hip-illegal-address"
    return False, None


def geometry_of(record):
    """The target dispatch (the one with K-like extent 4108) and its operand roles."""
    rows = record["hipblaslt"]["descriptors"]
    target = next((r for r in rows if (r.get("Bdesc") or {}).get("cols") == 4108
                   or (r.get("Adesc") or {}).get("cols") == 4108), None)
    if target is None:
        return {"known": False}
    a, b = target.get("Adesc") or {}, target.get("Bdesc") or {}
    return {"known": True, "Adesc": a, "Bdesc": b, "trans": target.get("trans"),
            "kernel_B_rows": b.get("rows"),
            "kernel_B_is_boundary_operand": b.get("rows") == 682,
            "kernel_N_mod_64": (b.get("rows") or 0) % 64,
            "kernel_K_mod_64": (b.get("cols") or 0) % 64}


def dispatch_profile(log_path):
    """Count the faulting run's dispatches by (M, N, K) and by over-read precondition."""
    layout = re.compile(r"(Adesc|Bdesc)=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)")
    trans = re.compile(r"transA=(\S+) transB=(\S+)")
    profile = {"calls": 0, "distinct": set(), "partial_n_and_k": 0,
               "sixteen_e_geometry": 0, "n_values": {}, "k_values": {}}
    for line in open(log_path, errors="replace"):
        if "[Trace][rocblaslt_matmul]" not in line:
            continue
        dims = {m.group(1): (int(m.group(3)), int(m.group(4)), int(m.group(5)))
                for m in layout.finditer(line)}
        flags = trans.search(line)
        if "Adesc" not in dims or "Bdesc" not in dims or flags is None:
            continue
        (ar, ac, _), (br, bc, _) = dims["Adesc"], dims["Bdesc"]
        if flags.group(1) == "OP_N":
            m, k = ar, ac
        else:
            k, m = ar, ac
        if flags.group(2) == "OP_N":
            k2, n = br, bc
        else:
            n, k2 = br, bc
        if k != k2:
            continue
        profile["calls"] += 1
        profile["distinct"].add((m, n, k))
        profile["n_values"][n] = profile["n_values"].get(n, 0) + 1
        profile["k_values"][k] = profile["k_values"].get(k, 0) + 1
        if n % 64 and k % 64:
            profile["partial_n_and_k"] += 1
        if (m, n, k) in {(256, 682, 4108), (682, 256, 4108)}:
            profile["sixteen_e_geometry"] += 1
    out = dict(profile)
    out["distinct"] = len(profile["distinct"])
    out["n_values"] = dict(sorted(profile["n_values"].items(), key=lambda kv: -kv[1])[:6])
    out["k_values"] = dict(sorted(profile["k_values"].items(), key=lambda kv: -kv[1])[:6])
    return out


def finalize() -> int:
    path = os.path.join(SCRATCH, "matrix.json")
    with open(path) as handle:
        summary = json.load(handle)
    for record in summary["cases"]:
        err = open(os.path.join(record["case_dir"], "stderr.txt")).read()
        faulted, mode = classify_text(err, record["exit_code"], record["signal"])
        record["faulted"] = faulted
        record["fault_mode"] = mode
        record["matches_expectation"] = faulted == (record["expected"] == "fault")
        record["geometry"] = geometry_of(record)
    positive = next(c for c in summary["cases"] if c["name"] == "positive_control")
    log = os.path.join(positive["case_dir"],
                       next(n for n in os.listdir(positive["case_dir"])
                            if n.startswith("hipblaslt-")))
    profile = dispatch_profile(log)
    synthetic = [c for c in summary["cases"] if c["name"].startswith("bf16_")]
    target = next(c for c in synthetic if "boundary_first" in c["name"] and "slack" not in c["name"])
    window = open(os.path.join(positive["case_dir"], "kernel-window.txt")).read()
    pages = re.findall(r"page fault \(src_id:0 ring:(\d+) vmid:(\d+) pasid:(\d+)\)", window)
    faults = re.findall(r"in page starting at address (0x[0-9a-fA-F]+) from client (\d+)", window)
    clients = re.findall(r"Faulty UTCL2 client ID: (\S+) \((0x[0-9a-fA-F]+)\)", window)
    perms = re.findall(r"PERMISSION_FAULTS: (0x[0-9a-fA-F]+)", window)
    rw = re.findall(r"RW: (0x[0-9a-fA-F]+)", window)
    addresses = sorted({a for a, _ in faults})
    kernel_fault = {
        "page_fault_records": len(pages),
        "addresses": addresses,
        "address_2mib_aligned": bool(addresses) and all(int(a, 16) % (2 << 20) == 0 for a in addresses),
        "utcl2_client": clients[0] if clients else None,
        "permission_flags": perms[0] if perms else None,
        "rw_flag": rw[0] if rw else None,
        "reset_lines": [l for l in window.splitlines() if _RESET.search(l)],
    }
    summary["findings"] = {
        "positive_control_faulted": positive["faulted"],
        "positive_fault_mode": positive["fault_mode"],
        "positive_last_printed_iteration": 25,
        "positive_dispatch": profile,
        "target_geometry": target["geometry"],
        "target_faulted": target["faulted"],
        "synthetic_all_clean": not any(c["faulted"] for c in synthetic),
        "positive_kernel_fault": kernel_fault,
    }
    roc = next((c for c in summary["cases"] if c["name"] == "positive_control_rocblas"), None)
    mix = {}
    if roc is not None:
        err = open(os.path.join(roc["case_dir"], "stderr.txt")).read()
        for name in ("hipblaslt", "tensile"):
            mix[name] = len(re.findall(
                r"rocblas_internal,rocblas_gemm_" + name + r"_backend", err))
        mix["rocblas_gemm_ex"] = len(re.findall(r"rocblas_gemm_ex", err))
    clean_control = roc is not None and mix.get("hipblaslt", 0) == 0
    summary["findings"]["rocblas_arm"] = {
        "ran": roc is not None, "faulted": bool(roc and roc["faulted"]),
        "fault_mode": roc["fault_mode"] if roc else None,
        "backend_mix": mix,
        "clean_control": clean_control,
        "why_not_clean": None if clean_control else
        "rocBLAS delegated calls to rocblas_gemm_hipblaslt_backend, so the arm still exercised "
        "hipBLASLt kernels and cannot separate the library from the fault",
    }
    if roc is not None and roc["faulted"] and not clean_control:
        summary["findings"]["backend_split"] = ("the hipblas backend also faults, but it routed "
                                                f"{mix.get('hipblaslt', 0)} calls back to hipBLASLt, "
                                                "so this is not a discriminator")
    decision = {
        "option": 2,
        "why": ("the historical fault reproduces in the model reproducer with the same kernel-log "
                "signature (2 MiB-aligned page, TCP client, permission fault, no reset), but the "
                "faulting kernel and hipBLASLt solution stay unresolved. The single-GEMM form of the "
                "hypothesis is not supported on this build: the call whose descriptors match the "
                "archived 16E capture exactly is clean at 16-byte slack, and the faulting 8E run "
                "never dispatches that geometry. The rocBLAS arm is not a clean control because "
                "rocBLAS delegated its calls back to hipBLASLt kernels. The mechanism is not "
                "confirmed and no mitigation should be designed from this evidence."),
    }
    summary["decision"] = decision
    summary["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary["findings"], indent=1, sort_keys=True))
    print("DECISION " + json.dumps(decision))
    return 0


def followup() -> int:
    path = os.path.join(SCRATCH, "matrix.json")
    with open(path) as handle:
        summary = json.load(handle)
    cases = summary["cases"]
    for record in cases:
        err = open(os.path.join(record["case_dir"], "stderr.txt")).read()
        faulted, mode = classify_text(err, record["exit_code"], record["signal"])
        record["faulted"] = faulted
        record["fault_mode"] = mode
        record["matches_expectation"] = faulted == (record["expected"] == "fault")
        record.setdefault("geometry", {"known": False})
    state = {"executions": summary["executions"]}

    out = os.path.join(SCRATCH, "cases", "bf16_boundary_first_tight", "case.json")
    tight = run_case("bf16_boundary_first_tight",
                     gemm_argv("bf16", 682, 4108, "tight", out=out,
                               orientation="boundary-first"),
                     "fault-or-clean",
                     "controlled 682-row operand in the kernel's B slot, 16-byte slack",
                     state)
    tight["geometry"] = geometry_of(tight)
    cases.append(tight)
    print("geometry:", json.dumps(tight["geometry"]), flush=True)

    slack = None
    if tight["faulted"]:
        out = os.path.join(SCRATCH, "cases", "bf16_boundary_first_slack", "case.json")
        slack = run_case("bf16_boundary_first_slack",
                         gemm_argv("bf16", 682, 4108, "generous", out=out,
                                   orientation="boundary-first"),
                         "clean", "boundary discriminator in the corrected orientation",
                         state)
        slack["geometry"] = geometry_of(slack)
        cases.append(slack)

    positive = next((c for c in cases if c["name"] == "positive_control"), None)
    if tight["faulted"] and slack is not None and not slack["faulted"] and tight["geometry"].get("kernel_B_is_boundary_operand"):
        decision = {"option": 1, "why": "the corrected orientation faults with 16-byte slack and is clean with slack; the kernel read the controlled buffer as its unchecked B operand with a partial N tile"}
    elif tight["faulted"] and slack is not None and slack["faulted"]:
        decision = {"option": 2, "why": "both the tight and the slack placement fault, so the boundary is not the discriminator in this orientation"}
    elif positive is not None and positive["faulted"]:
        decision = {"option": 4, "why": "synthetic cases stay clean while the model reproducer faults with a hip illegal-address error; the synthetic route does not exercise the failing dispatch, so kernel attribution stays unresolved"}
    else:
        decision = {"option": 4, "why": "no reproducing case in this matrix"}

    summary.update({"cases": cases, "executions": state["executions"],
                    "decision": decision})
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("\nDECISION " + json.dumps(decision), flush=True)
    print(f"executions used: {state['executions']}/{MAX_EXECUTIONS}", flush=True)
    return 0


def rocblas_arm() -> int:
    path = os.path.join(SCRATCH, "matrix.json")
    with open(path) as handle:
        summary = json.load(handle)
    state = {"executions": summary["executions"], "extra_env": {"ROCBLAS_LAYER": "9"}}
    rec = run_case("positive_control_rocblas",
                   [PYTHON, REPRO_ROCB, "--no_optimizer", "--grad_filter", "experts",
                    "--iters", "100"],
                   "clean", "same reproducer with rocBLAS instead of hipBLASLt",
                   state)
    summary["cases"].append(rec)
    summary["executions"] = state["executions"]
    hip = next((c for c in summary["cases"] if c["name"] == "positive_control"), None)
    summary["findings"]["rocblas_arm"] = {
        "faulted": rec["faulted"], "fault_mode": rec["fault_mode"],
        "hipblaslt_control_faulted": bool(hip and hip["faulted"]),
        "caveat": "switching the backend changes every GEMM kernel and the workspace, so a clean "
                  "rocBLAS run separates the hipBLASLt code path from the rest, not the library "
                  "from its allocation layout",
    }
    completed = "COMPLETED" in (rec.get("stdout_tail") or "")
    summary["findings"]["rocblas_arm"]["completed"] = completed
    if hip and hip["faulted"] and completed:
        summary["findings"]["backend_split"] = "hipBLASLt path faults, rocBLAS path completes"
    elif rec["faulted"]:
        summary["findings"]["backend_split"] = "both backends fault; the fault is not hipBLASLt-specific"
    elif not completed:
        summary["findings"]["backend_split"] = "rocBLAS arm did not complete and did not fault; not usable"
    summary["decision"]["why"] += (" A rocBLAS run of the same reproducer "
                                   + ("completed, localizing the fault to the hipBLASLt code path."
                                      if hip and hip["faulted"] and completed
                                      else "also faulted, so the fault is not hipBLASLt-specific."
                                      if rec["faulted"]
                                      else "did not reach completion, so it is not evidence."))
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("DECISION " + json.dumps(summary["decision"]), flush=True)
    print(f"executions used: {state['executions']}/{MAX_EXECUTIONS}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--rocblas-arm", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return dry_run()
    if args.execute:
        return execute()
    if args.followup:
        return followup()
    if args.finalize:
        return finalize()
    if args.rocblas_arm:
        return rocblas_arm()
    parser.error("choose --dry-run, --execute, --followup, --finalize or --rocblas-arm")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
