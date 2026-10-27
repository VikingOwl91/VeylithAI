"""PERF-5J analysis, CPU only.

Base gates come from the PERF-5H analyser (descriptors, storage, integrity).
This file adds what 5J needs: the product allocations, whether the padded
product is really the consumed GEMM operand of both candidate consumers, the
full-precision first loss, and fault-page attribution against product storage.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5h_analyze as base  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5j-20261027")
CASE = os.path.join(SCRATCH, "case_H")
REF = os.path.expanduser("~/veylith-perf5g-20261027/case_A")
MIB = 2 << 20
A_DESC = re.compile(r"A=(0x[0-9a-fA-F]+) Adesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
B_DESC = re.compile(r"B=(0x[0-9a-fA-F]+) Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
C_DESC = re.compile(r"C=(0x[0-9a-fA-F]+) Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
TRANS = re.compile(r"transA=(\S+) transB=(\S+)")
STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def r4k(n):
    return ((n + 4095) // 4096) * 4096


def r2mib(n):
    return ((n + MIB - 1) // MIB) * MIB


def consumer_kind(crows, ccols, trans_b):
    if trans_b == "OP_T" and crows == 256 and ccols == 682:
        return "w3_weight_grad"
    if trans_b == "OP_N" and crows == 256:
        return "w3_forward"
    return f"other_c{crows}x{ccols}"


def scan_operands(log_path, product_by_ptr):
    """Every dispatch that reads a padded product as its B operand."""
    matched, all_calls = [], 0
    if not log_path:
        return matched, all_calls
    for line in open(log_path, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        all_calls += 1
        b, c = B_DESC.search(line), C_DESC.search(line)
        if not b or not c:
            continue
        ptr = b.group(1)
        brows, bcols = int(b.group(3)), int(b.group(4))
        # Match on storage bytes rather than shape: a 3-D product from the dense
        # block is flattened by F.linear, so its descriptor covers numel elements
        # while the allocation keeps the 3-D shape.
        rec = product_by_ptr.get(ptr)
        if not rec or rec["tensor_bytes"] != brows * bcols * 2:
            continue
        shape = list(rec["shape"])
        trans = TRANS.search(line)
        kind = consumer_kind(int(c.group(3)), int(c.group(4)),
                             trans.group(2) if trans else "?")
        matched.append({"consumer": kind, "B": ptr, "Bdesc": [brows, bcols],
                        "Cdesc": [int(c.group(3)), int(c.group(4))],
                        "trans": list(trans.groups()) if trans else None,
                        "product_shape": shape,
                        "slack_after_tensor": rec["slack_after_tensor"],
                        "stamp": STAMP.match(line).group(1) if STAMP.match(line) else None})
    return matched, all_calls


def operand_proximity(log_path, product_by_ptr):
    """Reads whose last byte lands within 64 bytes of a 2 MiB boundary.

    This is the class of access that faulted in PERF-5I: an operand read with a
    page-ending tail.
    """
    rows = []
    if not log_path:
        return rows
    for line in open(log_path, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        for role, rx in (("A", A_DESC), ("B", B_DESC)):
            m = rx.search(line)
            if not m:
                continue
            ptr = m.group(1)
            rws, cols, ld = int(m.group(3)), int(m.group(4)), int(m.group(5))
            end = int(ptr, 16) + ld * cols * 2
            gap = r2mib(end) - end
            if gap > 64:
                continue
            rec = product_by_ptr.get(ptr)
            rows.append({"role": role, "ptr": ptr, "descriptor": [rws, cols, ld],
                         "tail_bytes": gap, "is_padded_product": bool(rec),
                         "product_kind": rec.get("kind") if rec else None,
                         "stamp": STAMP.match(line).group(1) if STAMP.match(line) else None})
    return rows


def attribute(pages, log_path, rows):
    result = {"pages": list(pages or []), "product_containment": 0,
              "all_padded_containment": 0, "nearest_padded_gap_bytes": None,
              "page_exact": [], "block_exact_count": 0}
    if not pages:
        return result
    for pstr in pages:
        P = int(pstr, 16)
        for r in rows:
            sbase = int(r["storage_ptr"], 16)
            send = sbase + r["storage_bytes"]
            if sbase <= P < send:
                result["all_padded_containment"] += 1
                if r.get("kind") == "swiglu_product":
                    result["product_containment"] += 1
        gaps = []
        for r in rows:
            sbase = int(r["storage_ptr"], 16)
            send = sbase + r["storage_bytes"]
            gaps.append(0 if sbase <= P <= send else min(abs(P - sbase), abs(P - send)))
        result["nearest_padded_gap_bytes"] = min(gaps) if gaps else None
        if not log_path:
            continue
        for line in open(log_path, errors="replace"):
            if "rocblaslt_matmul" not in line:
                continue
            for role, rx in (("A", A_DESC), ("B", B_DESC), ("C", C_DESC)):
                m = rx.search(line)
                if not m:
                    continue
                ptr = m.group(1)
                rws, cols, ld = int(m.group(3)), int(m.group(4)), int(m.group(5))
                end = int(ptr, 16) + ld * cols * 2
                if r2mib(end) == P:
                    result["block_exact_count"] += 1
                if r4k(end) != P:
                    continue
                rec = next((r for r in rows if r["data_ptr"] == ptr
                            and r["shape"] == [cols, rws]), None)
                result["page_exact"].append({
                    "page": pstr, "role": role, "ptr": ptr,
                    "descriptor": [rws, cols, ld], "tensor_end": hex(end),
                    "padded_kind": rec.get("kind") if rec else None,
                    "slack_after_tensor": rec["slack_after_tensor"] if rec else None,
                    "stamp": STAMP.match(line).group(1) if STAMP.match(line) else None,
                })
    return result


def first_loss(stdout_path):
    if not os.path.exists(stdout_path):
        return None
    for line in open(stdout_path, errors="replace"):
        if line.startswith("FIRSTLOSS "):
            return json.loads(line[len("FIRSTLOSS "):])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--pad", default=CASE)
    args = ap.parse_args()
    out_path = os.path.join(args.pad, "gates.json")

    sys.argv = ["v31p5h_analyze", "--ref", args.ref, "--pad", args.pad,
                "--out", out_path]
    base.main()

    gates = json.load(open(out_path))
    geo = json.load(open(os.path.join(args.pad, "geometry.json")))
    rec = json.load(open(os.path.join(args.pad, "record.json")))
    rows = geo.get("buffers", [])
    pc = geo.get("post_check") or {}
    logs = sorted(f for f in os.listdir(args.pad) if f.startswith("hipblaslt-"))
    log_path = os.path.join(args.pad, logs[0]) if logs else None
    stdout_path = os.path.join(args.pad, "stdout.txt")

    products = [r for r in rows if r.get("kind") == "swiglu_product"]
    product_by_ptr = {}
    for r in products:
        product_by_ptr.setdefault(r["data_ptr"], r)
    matched, all_calls = scan_operands(log_path, product_by_ptr)
    proximity = operand_proximity(log_path, product_by_ptr)
    by_kind = {}
    for m in matched:
        by_kind.setdefault(m["consumer"], []).append(m)

    ref_losses, ref_evals = base.losses(os.path.join(args.ref, "stdout.txt"))
    pad_losses, pad_evals = base.losses(stdout_path)
    fl = first_loss(stdout_path) or {}

    gates["gate_a_numerics"] = {
        "reference_display_losses": ref_losses,
        "padded_display_losses": pad_losses,
        "reference_display_evals": ref_evals,
        "padded_display_evals": pad_evals,
        "display_identity": bool(all(ref_losses[k] == pad_losses.get(k)
                                     for k in ref_losses)
                                 and all(a == b for a, b in zip(ref_evals, pad_evals))),
        "display_values_compared": len(set(ref_losses) & set(pad_losses)),
        "first_loss_full_precision": fl.get("iter0_loss"),
        "first_loss_4dp": fl.get("iter0_loss_4dp"),
        "anchor_matches_reference_4dp": ((fl.get("iter0_loss_4dp")
                                          or pad_losses.get(0)) == ref_losses.get(0)),
        "post_check": pc or None,
        "post_check_ran": bool(pc),
        "optimizer_state": "AdamW is constructed but --no_optimizer never steps",
    }

    gates["gate_b_operand_storage"] = {
        "product_allocations": len(products),
        "product_kinds": sorted({r.get("kind") for r in rows if r.get("kind")}),
        "product_shapes_sample": sorted({tuple(r["shape"]) for r in products})[:6],
        "product_slack_min": min((r["slack_after_tensor"] for r in products), default=None),
        "product_slack_ok": (bool(products) and min(r["slack_after_tensor"]
                                                   for r in products) >= 65536),
        "product_all_offset_zero": all(r["storage_offset"] == 0 for r in products),
        "product_all_contiguous": all(r["contiguous"] for r in products),
        "product_all_pointer_preserved": all(r.get("pointer_preserved") is True
                                             for r in products),
        "product_any_resized": any(r.get("resized_by_out") is True for r in products),
        "product_page_distance_range": [min((r["bytes_from_tensor_end_to_page_end"]
                                             for r in products), default=None),
                                        max((r["bytes_from_tensor_end_to_page_end"]
                                             for r in products), default=None)],
        "gemm_calls_logged": all_calls,
        "product_operand_dispatches": len(matched),
        "product_operand_by_consumer": {k: len(v) for k, v in by_kind.items()},
        "product_operand_consumed": len(matched) > 0,
        "both_consumers_see_padded_product": ("w3_forward" in by_kind
                                              and "w3_weight_grad" in by_kind),
        "product_page_distance_range": [min((r["bytes_from_tensor_end_to_page_end"]
                                             for r in products), default=None),
                                        max((r["bytes_from_tensor_end_to_page_end"]
                                             for r in products), default=None)],
        "operand_reads_within_64B_of_a_page": len(proximity),
        "operand_reads_within_64B_stock": sum(1 for r in proximity
                                              if not r["is_padded_product"]),
        "operand_reads_within_64B_padded_product": sum(1 for r in proximity
                                                       if r["is_padded_product"]),
        "operand_proximity_sample": proximity[:6],
        "sample_dispatches": matched[:4],
        "mapping_note": "allocation slack is recorded; GPU mapping accessibility "
                        "is inferred, not directly observed",
    }

    gates["gate_d_fault_attribution"] = attribute(
        rec.get("kernel_fault_addresses"), log_path, rows)

    ga = gates["gate_a_descriptors"]
    gb_storage = gates["gate_b_output_storage"]
    gd = gates["gate_d_integrity"]
    fa = gates["gate_d_fault_attribution"]
    an = gates["gate_a_numerics"]
    bo = gates["gate_b_operand_storage"]

    numerics_ok = bool(an["display_identity"] and an["anchor_matches_reference_4dp"])
    descriptors_ok = bool(ga["w3_group_equal"])
    operand_ok = bool(bo["product_operand_consumed"] and bo["product_slack_ok"]
                      and bo["product_all_offset_zero"] and bo["product_all_contiguous"]
                      and bo["product_all_pointer_preserved"]
                      and not bo["product_any_resized"])
    if not numerics_ok or not descriptors_ok:
        code, text = 4, "invalid due to numerical or descriptor mismatch"
    elif not operand_ok:
        code, text = 5, "inconclusive due to insufficient pointer or storage evidence"
    elif rec.get("faulted"):
        if fa["product_containment"] > 0:
            code, text = 3, "valid padded operand execution, faulted at the protected operand"
        else:
            code, text = 2, "valid padded operand execution, faulted at another allocation"
    else:
        code, text = 1, "valid padded operand execution, clean"
    gates["classification"] = {
        "code": code, "text": text, "numerics_ok": numerics_ok,
        "descriptors_ok": descriptors_ok, "operand_ok": operand_ok,
        "faulted": rec.get("faulted"), "completed": gd["completed"],
        "fault_page_in_product_storage": fa["product_containment"] > 0,
        "fault_page_in_any_padded_storage": fa["all_padded_containment"] > 0,
    }
    json.dump(gates, open(out_path, "w"), indent=2, sort_keys=True)
    print(json.dumps({k: gates[k] for k in ("classification",)}, indent=2,
                     sort_keys=True, default=str))
    print(json.dumps({"numerics": {k: an[k] for k in
                                   ("display_identity", "first_loss_full_precision",
                                    "anchor_matches_reference_4dp", "post_check_ran")},
                      "operand": {k: bo[k] for k in
                                  ("product_allocations", "product_slack_min",
                                   "product_operand_dispatches",
                                   "product_operand_by_consumer",
                                   "both_consumers_see_padded_product")},
                      "fault": {k: fa[k] for k in
                                ("pages", "product_containment",
                                 "all_padded_containment",
                                 "nearest_padded_gap_bytes")}},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
