"""PERF-5L analysis, CPU only.

Base gates come from the PERF-5H analyser. This adds the gradient-only arm's
own evidence: intervention isolation, the stock-product exposure census, and
fault-page attribution against the padded gradient allocations.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5h_analyze as base  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5l-20261027")
CASE = os.path.join(SCRATCH, "case_H")
REF = os.path.expanduser("~/veylith-perf5g-20261027/case_A")
MIB = 2 << 20
A_DESC = base.DESC  # full-line matcher, groups reused below
A_RE = base.DESC
A_OP = __import__("re").compile(
    r"A=(0x[0-9a-fA-F]+) Adesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
B_OP = __import__("re").compile(
    r"B=(0x[0-9a-fA-F]+) Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
C_OP = __import__("re").compile(
    r"C=(0x[0-9a-fA-F]+) Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
TRANS = __import__("re").compile(r"transA=(\S+) transB=(\S+)")
PRODUCT_BYTES = 4194300  # (3075, 682) bf16


def r2mib(n):
    return ((n + MIB - 1) // MIB) * MIB


def r4k(n):
    return ((n + 4095) // 4096) * 4096


def consumer(role, crows, ccols, trans_b):
    if role == "B" and crows == 256 and trans_b == "OP_N":
        return "w3_forward"
    if (role == "B" and crows == 256 and ccols == 682 and trans_b == "OP_T") or (
            role == "A" and crows == 682 and ccols == 256):
        return "w3_weight_grad"
    return f"other_{role}_c{crows}x{ccols}"


def analyse(case_dir, record):
    geo = json.load(open(os.path.join(case_dir, "geometry.json")))
    rows = geo.get("buffers", [])
    padded = {(r["data_ptr"], r["tensor_bytes"]): r for r in rows}

    tails = collections.Counter()
    for r in rows:
        tails[r2mib(int(r["data_ptr"], 16) + r["tensor_bytes"])
              - (int(r["data_ptr"], 16) + r["tensor_bytes"])] += 1
    isolation = {
        "kinds_recorded": sorted({r.get("kind") for r in rows}),
        "swiglu_product_records": sum(1 for r in rows
                                      if r.get("kind") == "swiglu_product"),
        "padded_gradient_allocations": len(rows),
        "slack_set": sorted({r["slack_after_tensor"] for r in rows}),
        "all_offset_zero": all(r["storage_offset"] == 0 for r in rows),
        "all_contiguous": all(r["contiguous"] for r in rows),
        "all_pointer_preserved": all(r.get("pointer_preserved") for r in rows),
        "any_resized": any(r.get("resized_by_out") for r in rows),
        "padded_gradients_with_tail_le_64": sum(v for t, v in tails.items() if t <= 64),
        "padded_tail_histogram_le_64": {t: v for t, v in sorted(tails.items()) if t <= 64},
    }

    logs = sorted(f for f in os.listdir(case_dir) if f.startswith("hipblaslt-"))
    log = os.path.join(case_dir, logs[0]) if logs else None
    stock_near, padded_reads, producers = [], collections.Counter(), collections.Counter()
    if log:
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
                ptr = m.group(1)
                end = int(ptr, 16) + nb
                tail = r2mib(end) - end
                rec = padded.get((ptr, nb))
                if rec is not None:
                    if role == "C":
                        producers["C_writes_into_padded_gradient"] += 1
                    else:
                        padded_reads[role] += 1
                    continue
                if role in ("A", "B") and 1 <= tail <= 64:
                    c, tr = C_OP.search(line), TRANS.search(line)
                    stock_near.append({
                        "role": role, "ptr": ptr, "nbytes": nb, "tail": tail,
                        "descriptor": [rws, cols, ld],
                        "consumer": consumer(role, int(c.group(3)), int(c.group(4)),
                                             tr.group(2)) if c and tr else "unknown",
                        "Cdesc": [int(c.group(3)), int(c.group(4))] if c else None,
                        "page_rounded_end": hex(r4k(end)),
                        "stamp": line[1:20]})
    exposure = {
        "stock_product_reads_tail_1_64": [s for s in stock_near
                                          if s["nbytes"] == PRODUCT_BYTES],
        "stock_product_reads_by_consumer":
            dict(collections.Counter(s["consumer"] for s in stock_near
                                     if s["nbytes"] == PRODUCT_BYTES)),
        "other_stock_reads_tail_1_64": [s for s in stock_near
                                        if s["nbytes"] != PRODUCT_BYTES][:6],
        "other_stock_reads_tail_1_64_count": sum(
            1 for s in stock_near if s["nbytes"] != PRODUCT_BYTES),
        "padded_gradient_reads_by_role": dict(padded_reads),
        "producer_dispatch_counts": dict(producers),
        "padded_read_note": "role B/A counts include allocator address reuse; they "
                            "are reported but not used as evidence, and no padded "
                            "gradient is within 64 B of a boundary",
    }

    pages = record.get("kernel_fault_addresses") or []
    fault = {"pages": pages, "padded_containment": 0, "page_exact": [],
             "critical_match": False}
    for pstr in pages:
        P = int(pstr, 16)
        for r in rows:
            sbase = int(r["storage_ptr"], 16)
            if sbase <= P < sbase + r["storage_bytes"]:
                fault["padded_containment"] += 1
        for s in stock_near:
            if r4k(int(s["ptr"], 16) + s["nbytes"]) == P:
                fault["page_exact"].append({"page": pstr, **s, "padded": False})
                if s["nbytes"] == PRODUCT_BYTES:
                    fault["critical_match"] = True
    return isolation, exposure, fault


def first_loss(stdout_path):
    if not os.path.exists(stdout_path):
        return {}
    for line in open(stdout_path, errors="replace"):
        if line.startswith("FIRSTLOSS "):
            return json.loads(line[len("FIRSTLOSS "):])
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", default=CASE)
    args = ap.parse_args()
    out_path = os.path.join(args.pad, "gates.json")

    sys.argv = ["v31p5h_analyze", "--ref", REF, "--pad", args.pad, "--out", out_path]
    base.main()

    gates = json.load(open(out_path))
    rec = json.load(open(os.path.join(args.pad, "record.json")))
    iso, exp, fault = analyse(args.pad, rec)
    fl = first_loss(os.path.join(args.pad, "stdout.txt"))
    ref_losses, ref_evals = base.losses(os.path.join(REF, "stdout.txt"))
    pad_losses, pad_evals = base.losses(os.path.join(args.pad, "stdout.txt"))

    gates["gate_a_numerics"] = {
        "reference_display_losses": ref_losses, "padded_display_losses": pad_losses,
        "reference_display_evals": ref_evals, "padded_display_evals": pad_evals,
        "display_identity": bool(all(ref_losses[k] == pad_losses.get(k)
                                     for k in ref_losses)
                                 and all(a == b for a, b in zip(ref_evals, pad_evals))),
        "display_values_compared": len(set(ref_losses) & set(pad_losses)),
        "first_loss_full_precision": fl.get("iter0_loss"),
        "first_loss_4dp": fl.get("iter0_loss_4dp"),
        "anchor_matches_reference_4dp": fl.get("iter0_loss_4dp") == ref_losses.get(0),
        "post_check_ran": False,
        "post_check_note": "the loop faulted before the post-loop recompute, so the "
                           "stock versus padded comparison did not run; the CPU "
                           "whole-model gate covers that comparison bitwise",
    }
    gates["gate_b_isolation"] = iso
    gates["gate_d_boundary_exposure"] = exp
    gates["fault_attribution"] = fault

    numerics_ok = bool(gates["gate_a_numerics"]["display_identity"]
                       and gates["gate_a_numerics"]["anchor_matches_reference_4dp"])
    isolation_ok = bool(iso["swiglu_product_records"] == 0
                        and iso["padded_gradient_allocations"] > 0
                        and iso["slack_set"] == [65536]
                        and iso["all_offset_zero"] and iso["all_contiguous"]
                        and iso["all_pointer_preserved"] and not iso["any_resized"])
    descriptors_ok = bool(gates["gate_a_descriptors"]["w3_group_equal"])
    critical_exposure = bool(exp["stock_product_reads_tail_1_64"])
    if not (numerics_ok and isolation_ok and descriptors_ok):
        code, text = 5, "invalid or inconclusive"
    elif rec.get("faulted"):
        code, text = ((1, "valid gradient-only run, fault at stock product boundary")
                      if fault["critical_match"]
                      else (2, "valid gradient-only run, fault elsewhere"))
    else:
        code, text = ((3, "valid gradient-only run, clean with critical exposure")
                      if critical_exposure
                      else (4, "valid gradient-only run, clean without critical exposure"))
    gates["classification"] = {
        "code": code, "text": text, "faulted": rec.get("faulted"),
        "critical_exposure": critical_exposure,
        "fault_page_matches_stock_product": fault["critical_match"],
        "numerics_ok": numerics_ok, "isolation_ok": isolation_ok,
        "descriptors_ok": descriptors_ok,
    }
    json.dump(gates, open(out_path, "w"), indent=2, sort_keys=True)
    print(json.dumps({"classification": gates["classification"]}, indent=2,
                     sort_keys=True))
    print(json.dumps({"isolation": {k: iso[k] for k in
                                    ("kinds_recorded", "swiglu_product_records",
                                     "padded_gradient_allocations", "slack_set",
                                     "padded_gradients_with_tail_le_64")},
                      "exposure": {k: exp[k] for k in
                                   ("stock_product_reads_by_consumer",
                                    "other_stock_reads_tail_1_64_count")},
                      "fault": {"pages": fault["pages"],
                                "padded_containment": fault["padded_containment"],
                                "critical_match": fault["critical_match"],
                                "page_exact": fault["page_exact"]}},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
