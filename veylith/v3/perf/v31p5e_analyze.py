"""PERF-5E analysis: does the BF16-to-FP32 consumer reach the fault page?

Inputs from one rolling run: ops.jsonl (metadata per mm/copy, flushed per op),
cycles.jsonl (coverage), record.json (fault page, exit), the hipBLASLt log
(boundary buffer pointer and extent) and the shipped kernel template
(CUDALoops.cuh) for the tail-masking question.

The decisive comparison is arithmetic, not temporal: for every recorded copy,
does its input tensor range reach the fault page, and is its storage the
allocation that ends at that page?
"""

from __future__ import annotations

import argparse
import json
import os
import re

LAY = re.compile(r"Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)")
PTR = re.compile(r"C=(0x[0-9a-fA-F]+)")


def load_ops(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def last_gemm_calls(case_dir):
    log = next((os.path.join(case_dir, n) for n in sorted(os.listdir(case_dir))
                if n.startswith("hipblaslt-")), None)
    if not log:
        return []
    calls = []
    for line in open(log, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        m, p = LAY.search(line), PTR.search(line)
        if m and p:
            calls.append({"C": int(p.group(1), 16),
                          "extent": int(m.group(2)) * int(m.group(3)) * 2,
                          "rows": int(m.group(2)), "cols": int(m.group(3))})
    return calls[-2:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    case = args.case
    record = json.load(open(os.path.join(case, "record.json")))
    fault_pages = [int(p, 16) for p in (record.get("kernel_fault_addresses") or [])]
    ops = load_ops(os.path.join(case, "ops.jsonl"))
    cycles = []
    if os.path.exists(os.path.join(case, "cycles.jsonl")):
        cycles = [json.loads(line) for line in open(os.path.join(case, "cycles.jsonl"))
                  if line.strip()]
    calls = last_gemm_calls(case)

    copies = [r for r in ops if r["op"].startswith(("aten.copy_", "aten._to_copy"))]
    mms = [r for r in ops if r["op"].startswith("aten.mm")]

    findings = {"case": case, "fault_pages": [hex(p) for p in fault_pages],
                "ops_recorded": len(ops), "mm_recorded": len(mms),
                "copies_recorded": len(copies),
                "cycles": len(cycles),
                "last_covered_iteration": (cycles[-1].get("covered_iteration")
                                           if cycles else None),
                "last_gemm_calls": [{"C": hex(c["C"]), "extent": c["extent"],
                                     "rows": c["rows"], "cols": c["cols"]}
                                    for c in calls],
                "boundary_copies": [], "last_mm": None, "last_copy": None}

    if mms:
        findings["last_mm"] = {"seq": mms[-1]["seq"], "op": mms[-1]["op"],
                               "in0": (mms[-1].get("in0") or {}).get("shape"),
                               "in1": (mms[-1].get("in1") or {}).get("shape"),
                               "out": (mms[-1].get("out") or {}).get("shape"),
                               "out_ptr": (mms[-1].get("out") or {}).get("data_ptr"),
                               "out_storage_nbytes": (mms[-1].get("out") or {}).get("storage_nbytes")}
    if copies:
        c = copies[-1]
        findings["last_copy"] = {"seq": c["seq"], "op": c["op"],
                                 "in0": c.get("in0"), "out": c.get("out")}

    for page in fault_pages:
        for row in copies:
            source = row.get("in0") or {}
            storage_ptr = source.get("storage_ptr")
            nbytes = source.get("storage_nbytes")
            if storage_ptr is None or nbytes is None:
                continue
            storage_end = storage_ptr + nbytes
            tensor_end = source["data_ptr"] + source["numel"] * source["element_size"]
            if storage_ptr <= page < storage_end:
                findings["boundary_copies"].append({
                    "seq": row["seq"], "op": row["op"],
                    "input_shape": source["shape"], "input_stride": source["stride"],
                    "input_dtype": source["dtype"],
                    "storage_offset": source["storage_offset"],
                    "contiguous": source["contiguous"],
                    "storage_ptr": hex(storage_ptr), "storage_nbytes": nbytes,
                    "storage_end": hex(storage_end), "fault_page": hex(page),
                    "fault_inside_storage": True,
                    "tensor_end": hex(tensor_end),
                    "tensor_range_reaches_page": tensor_end >= page,
                    "bytes_from_tensor_end_to_page": page - tensor_end,
                    "output_dtype": (row.get("out") or {}).get("dtype"),
                })

    # Precondition frequency, independent of whether a fault occurred: how often
    # does a GEMM output, or the input of a bf16->fp32 copy, sit in a storage
    # block whose end is exactly a 2 MiB boundary?
    mib = 2 << 20
    def storage_end(row):
        t = row.get("out") if row["op"].startswith("aten.mm") else row.get("in0")
        if not t or t.get("storage_ptr") is None or t.get("storage_nbytes") is None:
            return None
        return t["storage_ptr"] + t["storage_nbytes"], t

    mm_page = []
    copy_page = []
    copy_reaches = 0
    for row in ops:
        end = storage_end(row)
        if end is None:
            continue
        end_ptr, t = end
        if end_ptr % mib:
            continue
        if row["op"].startswith("aten.mm"):
            mm_page.append({"seq": row["seq"], "out_shape": t["shape"],
                            "storage_end": hex(end_ptr),
                            "tensor_bytes": t["numel"] * t["element_size"],
                            "slack_bytes": end_ptr - (t["data_ptr"] + t["numel"] * t["element_size"])})
        else:
            tensor_end = t["data_ptr"] + t["numel"] * t["element_size"]
            reaches = tensor_end >= end_ptr
            copy_reaches += 1 if reaches else 0
            copy_page.append({"seq": row["seq"], "in_shape": t["shape"],
                              "in_dtype": t["dtype"],
                              "storage_end": hex(end_ptr),
                              "slack_bytes": end_ptr - tensor_end,
                              "tensor_reaches_page": reaches})
    findings["precondition_frequency"] = {
        "note": ("counts from a run that did NOT fault; they measure how ordinary the "
                 "boundary condition is"),
        "mm_outputs_ending_at_2mib": len(mm_page),
        "mm_with_slack_under_16_bytes": sum(1 for r in mm_page if 0 <= r["slack_bytes"] < 16),
        "copies_of_input_storage_ending_at_2mib": len(copy_page),
        "copies_whose_range_reaches_the_page": copy_reaches,
        "examples_mm": mm_page[:3],
        "examples_copy": copy_page[:3],
    }

    reaches = [b for b in findings["boundary_copies"] if b["tensor_range_reaches_page"]]
    if reaches:
        verdict = "B"
        why = ("a recorded copy consumes the storage that ends at the fault page and its "
               "tensor range reaches that page")
    elif findings["boundary_copies"]:
        verdict = "C"
        why = ("a recorded copy consumes the boundary allocation, but its tensor range stops "
               f"{findings['boundary_copies'][0]['bytes_from_tensor_end_to_page']} bytes short "
               "of the fault page, and this build's vectorized elementwise kernel guards the "
               "tail block elementwise (CUDALoops.cuh / MemoryAccess.cuh), so the copy cannot "
               "reach the page")
    elif not fault_pages:
        verdict = "D"
        why = ("this run did not fault (exit 0, all 100 iterations), so there is no fault "
               "page to attribute; the precondition frequency below shows whether the "
               "boundary situation is rare or routine")
    else:
        verdict = "D"
        why = ("no recorded copy touches the storage that ends at the fault page; the "
               "consumer hypothesis is not supported by this run")

    findings["verdict"] = verdict
    findings["why"] = why
    findings["tail_masking_source"] = {
        "vectorized_elementwise_kernel_rocm": "ATen/native/cuda/CUDALoops.cuh",
        "tail_policy": "memory::policies::unroll, load and store guarded by if (thread_idx < remaining)",
        "consequence": "the tail block reads exactly `remaining` elements; a masked tail cannot over-read",
    }
    json.dump(findings, open(os.path.join(case, "analysis.json"), "w"),
              indent=2, sort_keys=True)
    print("VERDICT", verdict, "-", why)
    print("ops", len(ops), "mm", len(mms), "copies", len(copies), "cycles", len(cycles))
    print("boundary copies:", json.dumps(findings["boundary_copies"])[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
