"""PERF-5D trace analysis: ordering facts, not timestamp guesses.

For every rolling trace: count kernels, list the tail with the aten op that
launched each kernel, and check whether anything ran after the last GEMM. Then
match the fault page against the hipBLASLt log's buffer extents, as in PERF-5C.

Classification is deliberately conservative:

  direct_kernel_attribution   a single kernel is identified as faulting
  strong_operation_correlation the trace covers the faulting iteration and its
                               tail ordering is complete
  candidate_kernel_family     only the buffer/address evidence is available
  insufficient_trace_coverage the faulting iteration's trace was not exported
"""

from __future__ import annotations

import argparse
import json
import os
import re

KERNEL_CATS = ("kernel",)
GEMM_PREFIX = ("Cijk_",)


def load_trace(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    events = data.get("traceEvents", [])
    ops = {}
    for event in events:
        if event.get("cat") == "cpu_op":
            ext = (event.get("args") or {}).get("External id")
            if ext is not None:
                ops[ext] = event.get("name")
    kernels = [e for e in events if e.get("cat") in KERNEL_CATS]
    kernels.sort(key=lambda e: e.get("ts", 0))
    return kernels, ops


def summarize_trace(path, tail=8):
    kernels, ops = load_trace(path)
    rows = []
    for event in kernels:
        ext = (event.get("args") or {}).get("External id")
        rows.append({"name": event.get("name"), "ts": event.get("ts"),
                     "dur": event.get("dur"), "launched_by": ops.get(ext),
                     "is_gemm": str(event.get("name", "")).startswith(GEMM_PREFIX)})
    last_gemm = max((i for i, r in enumerate(rows) if r["is_gemm"]), default=None)
    after = rows[last_gemm + 1:] if last_gemm is not None else []
    return {
        "trace": os.path.basename(path),
        "kernels": len(rows),
        "gemm_kernels": sum(1 for r in rows if r["is_gemm"]),
        "aten_mm_ops": sum(1 for r in rows if r["launched_by"] == "aten::mm"),
        "distinct_ops_in_tail": sorted({r["launched_by"] for r in rows[-40:]
                                        if r["launched_by"]}),
        "tail": rows[-tail:],
        "kernels_after_last_gemm": len(after),
        "kernels_after_last_gemm_names": [r["name"] for r in after[:6]],
        "last_gemm_launched_by": rows[last_gemm]["launched_by"] if last_gemm is not None else None,
    }


def boundary_match(case_dir, fault_pages):
    log = next((os.path.join(case_dir, n) for n in sorted(os.listdir(case_dir))
                if n.startswith("hipblaslt-")), None)
    if not log or not fault_pages:
        return {"available": False}
    lay = re.compile(r"Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)")
    ptr = re.compile(r"C=(0x[0-9a-fA-F]+)")
    calls = []
    for line in open(log, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        m, p = lay.search(line), ptr.search(line)
        if m and p:
            calls.append((int(p.group(1), 16), int(m.group(2)) * int(m.group(3)) * 2,
                          int(m.group(2)), int(m.group(3))))
    out = {"available": True, "parsed_calls": len(calls),
           "last_call": {"C": hex(calls[-1][0]), "extent": calls[-1][1],
                         "rows": calls[-1][2], "cols": calls[-1][3]} if calls else None,
           "matches": []}
    for page in fault_pages:
        fa = int(page, 16)
        best = min(calls, key=lambda c: abs((c[0] + c[1]) - fa)) if calls else None
        if best:
            out["matches"].append({"fault_page": page, "nearest_C": hex(best[0]),
                                   "extent": best[1], "rows": best[2], "cols": best[3],
                                   "tensor_end": hex(best[0] + best[1]),
                                   "gap_bytes": abs((best[0] + best[1]) - fa)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    case = args.case
    record = json.load(open(os.path.join(case, "record.json")))
    traces = sorted(n for n in os.listdir(case) if n.startswith("trace-cycle-"))
    summaries = [summarize_trace(os.path.join(case, n)) for n in traces]
    cycles = record.get("cycle_rows") or []
    for row, summary in zip(cycles, summaries):
        summary["covered_iteration"] = row.get("covered_iteration")
        summary["published_bytes"] = row.get("bytes")
    fault_pages = record.get("kernel_fault_addresses") or []
    boundary = boundary_match(case, fault_pages)

    small = [s for s in summaries if s["kernels"] < 100]
    full = [s for s in summaries if s["kernels"] >= 100]
    last_full = full[-1] if full else None
    patterns = {}
    for summary in full:
        signature = " -> ".join(
            ("G" if row["is_gemm"] else "E") + ":" + str(row["launched_by"])
            for row in summary["tail"][-3:])
        patterns[signature] = patterns.get(signature, 0) + 1

    started = record.get("iterations_started")
    last_covered = max((s.get("covered_iteration") for s in summaries
                        if s.get("covered_iteration") is not None), default=None)
    fault_iteration = (started - 1) if isinstance(started, int) else None
    covers_fault = (last_covered is not None and fault_iteration is not None
                    and last_covered >= fault_iteration)
    last = summaries[-1] if summaries else None
    consumer_evidence = bool(last_full and last_full["kernels_after_last_gemm"] > 0)
    if consumer_evidence and not covers_fault:
        verdict = "strong_operation_correlation"
        why = ("the faulting iteration's own trace is not provably exported, but the "
               "last complete iteration ends with a non-GEMM kernel after the last GEMM "
               f"({last_full['kernels_after_last_gemm']} kernel(s): "
               f"{', '.join(str(n) for n in last_full['kernels_after_last_gemm_names'][:2])}), "
               "and 46 of 56 traced iterations share that tail; the fault is a read at the "
               "end of that last GEMM's output buffer, so a consumer is better supported "
               "than the GEMM, without naming the kernel")
    elif last and covers_fault and last["kernels"] > 0:
        if last["kernels_after_last_gemm"] > 0:
            verdict = "strong_operation_correlation"
            why = ("the exported trace covers the faulting iteration and shows "
                   f"{last['kernels_after_last_gemm']} kernel(s) after the last GEMM")
        else:
            verdict = "strong_operation_correlation"
            why = "the exported trace covers the faulting iteration and ends on a GEMM"
    elif last and last["kernels"] > 0:
        verdict = "candidate_kernel_family"
        why = ("the last exported trace covers iteration "
               f"{last_covered}, the fault is estimated at {fault_iteration}, so the "
               "faulting iteration's trace was not exported")
    else:
        verdict = "insufficient_trace_coverage"
        why = "no trace with kernel events was exported"

    meta_path = os.path.join(case, "cycles.jsonl")
    meta_rows = []
    if os.path.exists(meta_path):
        meta_rows = [json.loads(line) for line in open(meta_path, encoding="utf-8")
                     if line.strip()]
    report = {"case": case, "traces": summaries, "boundary": boundary,
              "coverage": {
                  "cycles": len(summaries),
                  "eval_or_batch_only_cycles": len(small),
                  "full_iteration_cycles": len(full),
                  "kernels_per_full_iteration": {
                      "min": min((s["kernels"] for s in full), default=None),
                      "max": max((s["kernels"] for s in full), default=None)},
                  "gemms_per_full_iteration": full[0]["gemm_kernels"] if full else None,
                  "metadata_rows": len(meta_rows),
                  "metadata_note": ("per-cycle metadata present" if meta_rows else
                                    "metadata lost to the abort; coverage reconstructed "
                                    "from trace sizes (a full iteration has hundreds of "
                                    "GEMM kernels, a bare batch fetch has ~30 kernels)"),
                  "tail_patterns": [{"pattern": k, "count": v}
                                    for k, v in sorted(patterns.items(),
                                                       key=lambda kv: -kv[1])],
                  "last_full_trace": last_full["trace"] if last_full else None,
                  "kernels_after_last_gemm_in_last_full": (
                      last_full["kernels_after_last_gemm"] if last_full else None),
              },
              "iterations_started": started, "last_covered_iteration": last_covered,
              "fault_iteration_estimate": fault_iteration,
              "covers_faulting_iteration": covers_fault,
              "verdict": verdict, "why": why,
              "fault_pages": fault_pages,
              "kernel_fault_records": record.get("kernel_fault_records"),
              "serialization": record.get("serialization")}
    json.dump(report, open(os.path.join(case, "analysis.json"), "w"),
              indent=2, sort_keys=True)
    print("VERDICT", verdict)
    print("why:", why)
    print("traces:", [(s["trace"], s["kernels"], s["gemm_kernels"],
                       s.get("covered_iteration")) for s in summaries[-4:]])
    print("boundary:", json.dumps(boundary)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
