"""PERF-5N analysis, CPU only.

Base gates come from the PERF-5H analyser. This adds the 2 MiB front-padding
arm's own evidence: product isolation, the critical-geometry census over the
products actually consumed by the w3 GEMMs, fault-page attribution, and a
reconciliation block comparing the tight-boundary tensors across PERF-5I to 5N.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5h_analyze as base  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5n-20261027")
CASE = os.path.join(SCRATCH, "case_H")
REF = os.path.expanduser("~/veylith-perf5g-20261027/case_A")
MIB = 2 << 20
PAGE = 4096
FRONT = 2 << 20
PRODUCT_BYTES = 4194300

A_OP = re.compile(r"A=(0x[0-9a-fA-F]+) Adesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
B_OP = re.compile(r"B=(0x[0-9a-fA-F]+) Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
C_OP = re.compile(r"C=(0x[0-9a-fA-F]+) Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
TRANS = re.compile(r"transA=(\S+) transB=(\S+)")

# tail-4 (M, 682) bf16 GEMM operands seen across the series, all 2 MiB aligned
SERIES = {
  "PERF-5I": {"ptr": "0x7ff321000000", "faulted": True,
              "arm": "mode 1, product stock, gradient padded"},
  "PERF-5J": {"ptr": "0x7fdf79800000", "faulted": False,
              "arm": "mode 2, product back-padded, gradient stock"},
  "PERF-5K": {"ptr": "0x7f594be00000", "faulted": False,
              "arm": "mode 2 replication"},
  "PERF-5L": {"ptr": "0x7fcccde00000", "faulted": True,
              "arm": "mode 1 replication"},
  "PERF-5N": {"ptr": "0x7f505b200000", "faulted": True,
              "arm": "mode 4, product front-padded by 2 MiB"},
}


def page_dist(ptr, nbytes):
    end = ptr + nbytes
    return ((end + PAGE - 1) // PAGE) * PAGE - end


def grid_dist(ptr, nbytes):
    end = ptr + nbytes
    return ((end + MIB - 1) // MIB) * MIB - end


def consumer(role, crows, ccols, trans_b):
    if role == "B" and crows == 256 and trans_b == "OP_N":
        return "forward_shape_C_Mx256"
    if (role == "A" and crows == 682 and ccols == 256) or (
            role == "B" and crows == 256 and ccols == 682 and trans_b == "OP_T"):
        return "weight_grad_shape_C_256x682"
    return f"other_{role}_c{crows}x{ccols}"


def scan(case_dir, prod):
    logs = sorted(f for f in os.listdir(case_dir) if f.startswith("hipblaslt-"))
    log = os.path.join(case_dir, logs[0]) if logs else None
    reads, near, critical = 0, [], {}
    if not log:
        return reads, near, critical
    for line in open(log, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        for role, rx in (("A", A_OP), ("B", B_OP)):
            m = rx.search(line)
            if not m:
                continue
            rws, cols, ld = int(m.group(3)), int(m.group(4)), int(m.group(5))
            if ld != rws:
                continue
            nb = rws * cols * 2
            ptr_s = m.group(1)
            rec = prod.get((ptr_s, nb))
            if rec is None:
                continue
            reads += 1
            ptr = int(ptr_s, 16)
            p, g = page_dist(ptr, nb), grid_dist(ptr, nb)
            c, tr = C_OP.search(line), TRANS.search(line)
            cls = consumer(role, int(c.group(3)) if c else 0,
                           int(c.group(4)) if c else 0,
                           tr.group(2) if tr else "?")
            if p <= 64 or g <= 64:
                key = (ptr_s, nb)
                e = critical.setdefault(key, {
                    "ptr": ptr_s, "shape": rec["shape"], "tensor_bytes": nb,
                    "storage_bytes": rec["storage_bytes"],
                    "offset_bytes": rec.get("offset_bytes"),
                    "trailing_bytes": rec["slack_after_tensor"],
                    "page_dist": p, "grid_2mib_dist": g,
                    "storage_end": hex(int(rec["storage_ptr"], 16)
                                       + rec["storage_bytes"]),
                    "consumers": []})
                if cls not in e["consumers"]:
                    e["consumers"].append(cls)
                if p <= 64:
                    near.append({"ptr": ptr_s, "page_dist": p, "grid": g,
                                 "consumer": cls, "stamp": line[1:20]})
    return reads, near, critical


def fault_scan(case_dir, page):
    """Every GEMM operand or output whose page-rounded end is the fault page."""
    out = []
    logs = sorted(f for f in os.listdir(case_dir) if f.startswith("hipblaslt-"))
    if not logs:
        return out
    log = os.path.join(case_dir, logs[0])
    for line in open(log, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        for role, rx in (("A", A_OP), ("B", B_OP), ("C", C_OP)):
            m = rx.search(line)
            if not m:
                continue
            rws, cols, ld = int(m.group(3)), int(m.group(4)), int(m.group(5))
            if ld != rws:
                continue
            nb = rws * cols * 2
            ptr = int(m.group(1), 16)
            if ((ptr + nb + PAGE - 1) // PAGE) * PAGE != page:
                continue
            c, tr = C_OP.search(line), TRANS.search(line)
            out.append({"role": role, "ptr": m.group(1), "nbytes": nb,
                        "descriptor": [rws, cols, ld],
                        "page_dist": page_dist(ptr, nb),
                        "grid_2mib_dist": grid_dist(ptr, nb),
                        "Cdesc": [int(c.group(3)), int(c.group(4))] if c else None,
                        "trans": list(tr.groups()) if tr else None,
                        "stamp": line[1:20]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", default=CASE)
    args = ap.parse_args()
    out_path = os.path.join(args.pad, "gates.json")

    sys.argv = ["v31p5h_analyze", "--ref", REF, "--pad", args.pad, "--out", out_path]
    base.main()

    gates = json.load(open(out_path))
    rec = json.load(open(os.path.join(args.pad, "record.json")))
    geo = json.load(open(os.path.join(args.pad, "geometry.json")))
    pc = geo.get("post_check") or {}
    rows = geo.get("buffers", [])
    prod = {(r["data_ptr"], r["tensor_bytes"]): r for r in rows}

    isolation = {
        "kinds_recorded": sorted({r.get("kind") for r in rows}),
        "product_allocations": len(rows),
        "offset_bytes_set": sorted({r.get("offset_bytes") for r in rows}),
        "trailing_bytes_set": sorted({r["slack_after_tensor"] for r in rows}),
        "all_end_at_storage_end": all(r.get("ends_at_storage_end") is True for r in rows),
        "all_contiguous": all(r["contiguous"] for r in rows),
        "all_pointer_preserved": all(r.get("pointer_preserved") for r in rows),
        "any_resized": any(r.get("resized_by_out") for r in rows),
        "padded_gradient_records": sum(1 for r in rows if r.get("kind") == "w3_grad"),
    }
    reads, near, critical = scan(args.pad, prod)
    exposure = {
        "product_operand_reads": reads,
        "products_read_within_64B": len(critical),
        "products_read_within_64B_detail": sorted(critical.values(),
                                                  key=lambda e: (e["page_dist"],
                                                                 e["grid_2mib_dist"]))[:6],
        "min_page_dist_product_read": min((e["page_dist"] for e in critical.values()),
                                          default=None),
        "min_grid_dist_product_read": min((e["grid_2mib_dist"] for e in critical.values()),
                                          default=None),
        "critical_geometry_products": [e for e in critical.values()
                                       if e["page_dist"] == 4
                                       and e["grid_2mib_dist"] == 4
                                       and e["trailing_bytes"] == 0],
        "critical_geometry_present": any(e["page_dist"] == 4
                                         and e["grid_2mib_dist"] == 4
                                         and e["trailing_bytes"] == 0
                                         for e in critical.values()),
        "allocator_block": "not observable in the record; the 2 MiB grid is not "
                           "treated as a proven mapping boundary",
    }

    pages = rec.get("kernel_fault_addresses") or []
    fault = {"pages": pages, "padded_containment": 0, "page_exact": [],
             "critical_match": False}
    for pstr in pages:
        P = int(pstr, 16)
        for r in rows:
            sbase = int(r["storage_ptr"], 16)
            if sbase <= P < sbase + r["storage_bytes"]:
                fault["padded_containment"] += 1
        for e in fault_scan(args.pad, P):
            e["padded"] = (e["ptr"], e["nbytes"]) in prod
            fault["page_exact"].append({**e, "page": pstr})
        fault["critical_match"] = any(
            (not e["padded"]) and e["nbytes"] == PRODUCT_BYTES
            and e["page_dist"] == 4 and e["grid_2mib_dist"] == 4
            for e in fault["page_exact"])

    series = {}
    for name, s in SERIES.items():
        ptr = int(s["ptr"], 16)
        series[name] = {"arm": s["arm"], "faulted": s["faulted"],
                        "ptr": s["ptr"], "ptr_mod_2mib": ptr % MIB,
                        "page_dist": page_dist(ptr, PRODUCT_BYTES),
                        "grid_2mib_dist": grid_dist(ptr, PRODUCT_BYTES),
                        "is_padded_product_in_that_arm": False}

    ref_losses, ref_evals = base.losses(os.path.join(REF, "stdout.txt"))
    pad_losses, pad_evals = base.losses(os.path.join(args.pad, "stdout.txt"))
    fl = {}
    for line in open(os.path.join(args.pad, "stdout.txt"), errors="replace"):
        if line.startswith("FIRSTLOSS "):
            fl = json.loads(line[len("FIRSTLOSS "):])
            break
    gates["gate_a_numerics"] = {
        "reference_display_losses": ref_losses,
        "padded_display_losses": pad_losses,
        "reference_display_evals": ref_evals,
        "padded_display_evals": pad_evals,
        "display_identity": bool(all(ref_losses[k] == pad_losses.get(k)
                                     for k in ref_losses)
                                 and all(a == b for a, b in zip(ref_evals, pad_evals))),
        "first_loss_full_precision": fl.get("iter0_loss"),
        "first_loss_4dp": fl.get("iter0_loss_4dp"),
        "anchor_matches_reference_4dp": fl.get("iter0_loss_4dp") == ref_losses.get(0),
        "post_check_ran": bool(pc),
        "post_check_note": "the loop faulted before the post-loop recompute",
    }
    numerics_ok = bool(gates["gate_a_numerics"]["display_identity"]
                       and gates["gate_a_numerics"]["anchor_matches_reference_4dp"])
    isolation_ok = bool(isolation["kinds_recorded"] == ["swiglu_product_front2mib"]
                        and isolation["offset_bytes_set"] == [FRONT]
                        and isolation["trailing_bytes_set"] == [0]
                        and isolation["all_end_at_storage_end"]
                        and isolation["all_contiguous"]
                        and isolation["all_pointer_preserved"]
                        and not isolation["any_resized"]
                        and isolation["padded_gradient_records"] == 0)
    descriptors_ok = bool(gates["gate_a_descriptors"]["w3_group_equal"])
    if not (numerics_ok and isolation_ok and descriptors_ok):
        code, text = 5, "invalid due to correctness, descriptor or storage mismatch"
    elif rec.get("faulted"):
        code, text = ((1, "valid critical exposure with fault at the product boundary")
                      if exposure["critical_geometry_present"]
                      and fault["critical_match"]
                      and fault["padded_containment"] > 0
                      else (4, "fault elsewhere"))
    else:
        code, text = ((2, "valid critical exposure with clean completion")
                      if exposure["critical_geometry_present"]
                      else (3, "valid run without critical exposure"))

    gates["gate_b_isolation"] = isolation
    gates["gate_d_critical_geometry"] = exposure
    gates["fault_attribution"] = fault
    gates["series_reconciliation"] = {
        "note": "every tight-boundary tensor in the series is a 2 MiB aligned "
                "(M,682) bf16 GEMM operand ending 4 bytes before a 2 MiB line; "
                "three of these runs faulted and two did not, so the geometry "
                "alone does not separate them",
        "tensors": series,
        "descriptor_ambiguity": "the same descriptor pairs are produced by the "
                                "w3 forward/weight-grad pair reading the product "
                                "and by the w1/w2 input-grad/weight-grad pair "
                                "reading the product derivative, so a log alone "
                                "cannot name which (M,682) tensor was read",
    }
    gates["classification"] = {
        "code": code, "text": text, "faulted": rec.get("faulted"),
        "numerics_ok": numerics_ok, "isolation_ok": isolation_ok,
        "descriptors_ok": descriptors_ok,
        "product_critical_geometry_present": exposure["critical_geometry_present"],
        "fault_page_inside_padded_product": fault["padded_containment"] > 0,
        "fault_matches_stock_tail4_tensor": fault["critical_match"],
    }
    json.dump(gates, open(out_path, "w"), indent=2, sort_keys=True)
    print(json.dumps({"classification": gates["classification"]}, indent=2,
                     sort_keys=True))
    print(json.dumps({"isolation": {k: isolation[k] for k in
                                    ("kinds_recorded", "product_allocations",
                                     "offset_bytes_set", "trailing_bytes_set",
                                     "all_end_at_storage_end",
                                     "padded_gradient_records")},
                      "exposure": {k: exposure[k] for k in
                                   ("product_operand_reads",
                                    "products_read_within_64B",
                                    "min_page_dist_product_read",
                                    "min_grid_dist_product_read",
                                    "critical_geometry_present")},
                      "fault": {"pages": fault["pages"],
                                "padded_containment": fault["padded_containment"],
                                "page_exact": fault["page_exact"],
                                "critical_match": fault["critical_match"]},
                      "series": series}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
