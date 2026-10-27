"""PERF-5K analysis, CPU only.

Runs the PERF-5J gate analysis on the replication, then adds the two things 5K
needs beyond it: a boundary-exposure census for both product arms, and the
side-by-side comparison. The intervention is unchanged, so the gates are the
same code path.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5j_analyze as j5  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5k-20261027")
CASE = os.path.join(SCRATCH, "case_H")
REF = os.path.expanduser("~/veylith-perf5g-20261027/case_A")
PEER = os.path.expanduser("~/veylith-perf5j-20261027/case_H")
MIB = 2 << 20


def r2mib(n):
    return ((n + MIB - 1) // MIB) * MIB


def consumer(crows, ccols, trans_b):
    if trans_b == "OP_T" and crows == 256 and ccols == 682:
        return "w3_weight_grad"
    if trans_b == "OP_N" and crows in (256,):
        return "w3_forward_or_inputgrad"
    return f"other_c{crows}x{ccols}"


def exposure(case_dir):
    """Boundary exposure and the reads that landed on it."""
    geo = json.load(open(os.path.join(case_dir, "geometry.json")))
    rec = json.load(open(os.path.join(case_dir, "record.json")))
    products = [r for r in geo.get("buffers", []) if r.get("kind") == "swiglu_product"]
    by_ptr = {}
    for r in products:
        by_ptr.setdefault(r["data_ptr"], r)

    tails = collections.Counter()
    for r in products:
        end = int(r["data_ptr"], 16) + r["tensor_bytes"]
        tails[r2mib(end) - end] += 1
    alloc_le64 = [r for r in products
                  if r2mib(int(r["data_ptr"], 16) + r["tensor_bytes"])
                  - (int(r["data_ptr"], 16) + r["tensor_bytes"]) <= 64]

    logs = sorted(f for f in os.listdir(case_dir) if f.startswith("hipblaslt-"))
    log_path = os.path.join(case_dir, logs[0]) if logs else None
    padded_reads, stock_reads = [], []
    if log_path:
        for line in open(log_path, errors="replace"):
            if "rocblaslt_matmul" not in line:
                continue
            for role, rx in (("A", j5.A_DESC), ("B", j5.B_DESC)):
                m = rx.search(line)
                if not m:
                    continue
                ptr = m.group(1)
                rws, cols, ld = int(m.group(3)), int(m.group(4)), int(m.group(5))
                if ld != rws:          # packed column-major only, exact byte count
                    continue
                nbytes = rws * cols * 2
                end = int(ptr, 16) + nbytes
                tail = r2mib(end) - end
                if tail > 64:
                    continue
                prod = by_ptr.get(ptr)
                if prod is not None and prod["tensor_bytes"] != nbytes:
                    prod = None
                c, tr = j5.C_DESC.search(line), j5.TRANS.search(line)
                kind = consumer(int(c.group(3)), int(c.group(4)),
                                tr.group(2)) if c and tr else "unknown"
                row = {"role": role, "ptr": ptr, "nbytes": nbytes, "tail": tail,
                       "consumer": kind,
                       "stamp": line[1:20]}
                (padded_reads if prod is not None else stock_reads).append(row)

    padded_le64 = [r for r in padded_reads if r["tail"] <= 64]
    return {
        "product_allocations": len(products),
        "product_tensor_end_tail_histogram": dict(sorted(tails.items())),
        "product_allocations_tail_le_64": len(alloc_le64),
        "product_allocations_tail_1_64": sum(v for t, v in tails.items() if 1 <= t <= 64),
        "padded_product_reads": len(padded_reads),
        "padded_product_reads_tail_le_64": len(padded_le64),
        "padded_product_reads_tail_le_64_by_consumer":
            dict(collections.Counter(r["consumer"] for r in padded_le64)),
        "padded_product_reads_tail_le_64_detail": padded_le64[:8],
        "stock_reads_tail_1_64": [r for r in stock_reads if 1 <= r["tail"] <= 64][:8],
        "stock_reads_tail_1_64_count": sum(1 for r in stock_reads if 1 <= r["tail"] <= 64),
        "stock_reads_tail_0_count": sum(1 for r in stock_reads if r["tail"] == 0),
        "faulted": rec.get("faulted"),
        "fault_pages": rec.get("kernel_fault_addresses"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", default=CASE)
    ap.add_argument("--peer", default=PEER)
    args = ap.parse_args()
    out_path = os.path.join(args.pad, "gates.json")

    sys.argv = ["v31p5j_analyze", "--ref", REF, "--pad", args.pad]
    j5.main()

    gates = json.load(open(out_path))
    mine = exposure(args.pad)
    peer = exposure(args.peer)
    gates["boundary_exposure"] = mine
    gates["replication_comparison"] = {
        "perf_5j": peer, "perf_5k": mine,
        "same_product_allocation_count":
            peer["product_allocations"] == mine["product_allocations"],
        "both_clean": (not peer["faulted"]) and (not mine["faulted"]),
        "exposure_present_in_5k": mine["padded_product_reads_tail_le_64"] > 0,
        "first_loss_equal": (gates["gate_a_numerics"]["first_loss_full_precision"]
                             == json.load(open(os.path.join(args.peer, "geometry.json")))
                             .get("post_check", {}).get("iter0_loss_loop")),
    }

    an, bo = gates["gate_a_numerics"], gates["gate_b_operand_storage"]
    numerics_ok = bool(an["display_identity"] and an["anchor_matches_reference_4dp"])
    descriptors_ok = bool(gates["gate_a_descriptors"]["w3_group_equal"])
    operand_ok = bool(bo["product_operand_consumed"] and bo["product_slack_ok"]
                      and bo["product_all_offset_zero"]
                      and bo["product_all_contiguous"]
                      and bo["product_all_pointer_preserved"]
                      and not bo["product_any_resized"])
    if mine["faulted"]:
        code, text = 3, "valid padded run with GPUVM fault"
    elif not numerics_ok or not descriptors_ok:
        code, text = 4, "invalid due to numerical or descriptor mismatch"
    elif not operand_ok:
        code, text = 5, "inconclusive"
    elif mine["padded_product_reads_tail_le_64"] > 0:
        code, text = 1, "valid clean replication with boundary exposure"
    else:
        code, text = 2, "valid clean replication without boundary exposure"
    gates["classification"] = {
        "code": code, "text": text, "numerics_ok": numerics_ok,
        "descriptors_ok": descriptors_ok, "operand_ok": operand_ok,
        "faulted": mine["faulted"],
        "boundary_exposure_reads": mine["padded_product_reads_tail_le_64"],
        "completed": gates["gate_d_integrity"]["completed"],
    }
    json.dump(gates, open(out_path, "w"), indent=2, sort_keys=True)
    print(json.dumps({"classification": gates["classification"]}, indent=2,
                     sort_keys=True))
    print(json.dumps({"exposure_5k": {k: mine[k] for k in
                                      ("product_allocations",
                                       "product_allocations_tail_le_64",
                                       "product_allocations_tail_1_64",
                                       "padded_product_reads_tail_le_64",
                                       "padded_product_reads_tail_le_64_by_consumer",
                                       "stock_reads_tail_1_64_count",
                                       "stock_reads_tail_0_count")},
                      "exposure_5j": {k: peer[k] for k in
                                      ("product_allocations",
                                       "product_allocations_tail_le_64",
                                       "product_allocations_tail_1_64",
                                       "padded_product_reads_tail_le_64",
                                       "padded_product_reads_tail_le_64_by_consumer",
                                       "stock_reads_tail_1_64_count",
                                       "stock_reads_tail_0_count")}},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
