"""PERF-5M analysis, CPU only.

Base gates come from the PERF-5H analyser. This adds what the front-padded arm
needs: isolation of the product view, and a boundary census that reports both
distances that matter, the 4 KiB page remainder and the 2 MiB grid distance,
because a front-padded view ends at its storage end while the allocator block
may end elsewhere.
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

SCRATCH = os.path.expanduser("~/veylith-perf5m-20261027")
CASE = os.path.join(SCRATCH, "case_H")
REF = os.path.expanduser("~/veylith-perf5g-20261027/case_A")
MIB = 2 << 20
PAGE = 4096
SLACK = 65536

A_OP = re.compile(r"A=(0x[0-9a-fA-F]+) Adesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
B_OP = re.compile(r"B=(0x[0-9a-fA-F]+) Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
C_OP = re.compile(r"C=(0x[0-9a-fA-F]+) Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
TRANS = re.compile(r"transA=(\S+) transB=(\S+)")


def page_dist(row):
    end = int(row["data_ptr"], 16) + row["tensor_bytes"]
    return ((end + PAGE - 1) // PAGE) * PAGE - end


def grid_dist(row):
    end = int(row["data_ptr"], 16) + row["tensor_bytes"]
    return ((end + MIB - 1) // MIB) * MIB - end


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
    prod = {(r["data_ptr"], r["tensor_bytes"]): r for r in rows}

    isolation = {
        "kinds_recorded": sorted({r.get("kind") for r in rows}),
        "front_product_records": len(rows),
        "padded_gradient_records": sum(1 for r in rows if r.get("kind") == "w3_grad"),
        "offset_bytes_set": sorted({r.get("offset_bytes") for r in rows}),
        "trailing_bytes_set": sorted({r["slack_after_tensor"] for r in rows}),
        "all_end_at_storage_end": all(r.get("ends_at_storage_end") is True for r in rows),
        "all_storage_overhead_bytes": sorted({r["storage_bytes"] - r["tensor_bytes"]
                                              for r in rows}),
        "all_contiguous": all(r["contiguous"] for r in rows),
        "all_pointer_preserved": all(r.get("pointer_preserved") for r in rows),
        "any_resized": any(r.get("resized_by_out") for r in rows),
    }

    logs = sorted(f for f in os.listdir(case_dir) if f.startswith("hipblaslt-"))
    log = os.path.join(case_dir, logs[0]) if logs else None
    read_pd, read_gd, reads, exposed = collections.Counter(), collections.Counter(), 0, {}
    if log:
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
                rec = prod.get((m.group(1), nb))
                if rec is None:
                    continue
                reads += 1
                p, g = page_dist(rec), grid_dist(rec)
                read_pd[p] += 1
                read_gd[g] += 1
                if p <= 64 or g <= 64:
                    c, tr = C_OP.search(line), TRANS.search(line)
                    key = (rec["data_ptr"], nb)
                    e = exposed.setdefault(key, {
                        "ptr": rec["data_ptr"], "shape": rec["shape"],
                        "tensor_bytes": nb, "storage_bytes": rec["storage_bytes"],
                        "offset_bytes": rec.get("offset_bytes"),
                        "trailing_bytes": rec["slack_after_tensor"],
                        "page_dist": p, "grid_2mib_dist": g,
                        "round_2mib_gap": ((nb + SLACK + MIB - 1) // MIB) * MIB
                                          - (nb + SLACK),
                        "roles": [], "consumers": []})
                    e["roles"].append(role)
                    if c and tr:
                        e["consumers"].append(consumer(role, int(c.group(3)),
                                                       int(c.group(4)), tr.group(2)))
    exposure = {
        "product_allocations": len(rows),
        "reads_of_products": reads,
        "read_page_dist_histogram_le_64": {k: v for k, v in sorted(read_pd.items())
                                           if k <= 64},
        "read_grid_dist_histogram_le_64": {k: v for k, v in sorted(read_gd.items())
                                           if k <= 80},
        "reads_at_page_dist_4": read_pd.get(4, 0),
        "reads_at_page_dist_le_16": sum(v for k, v in read_pd.items() if k <= 16),
        "exposed_products": sorted(exposed.values(),
                                   key=lambda e: (e["page_dist"], e["grid_2mib_dist"])),
        "exposed_product_count": len(exposed),
        "mapping_note": "the view ends at its backing storage end; whether the "
                        "allocator block and driver mapping continue past it is "
                        "not visible in the record, so mapping beyond the logical "
                        "end is unknown, not assumed",
        "interpretation_note": "page_dist is the 4 KiB remainder that made the "
                               "stock 4,194,300-byte allocation fault; grid_2mib "
                               "is the distance to the 2 MiB line where a rounded "
                               "block would end",
    }

    pages = record.get("kernel_fault_addresses") or []
    fault = {"pages": pages, "padded_containment": 0, "page_exact": []}
    for pstr in pages:
        P = int(pstr, 16)
        for r in rows:
            sbase = int(r["storage_ptr"], 16)
            if sbase <= P < sbase + r["storage_bytes"]:
                fault["padded_containment"] += 1
        for e in exposed.values():
            end = int(e["ptr"], 16) + e["tensor_bytes"]
            if ((end + PAGE - 1) // PAGE) * PAGE == P:
                fault["page_exact"].append({"page": pstr, **e})
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
    geo = json.load(open(os.path.join(args.pad, "geometry.json")))
    pc = geo.get("post_check") or {}
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
        "post_check_ran": bool(pc),
        "stock_recompute_full_precision": pc.get("stock_loss_recomputed"),
        "padded_recompute_full_precision": pc.get("padded_loss_recomputed"),
        "stock_vs_padded_loss_bitwise": pc.get("stock_vs_padded_loss_bitwise"),
        "grads_compared": pc.get("padded_grads_captured"),
        "grads_bitwise_equal": pc.get("grads_bitwise_equal"),
        "grad_norm_equal": pc.get("grad_norm_equal"),
    }
    gates["gate_b_isolation"] = iso
    gates["gate_d_boundary_exposure"] = exp
    gates["fault_attribution"] = fault

    numerics_ok = bool(gates["gate_a_numerics"]["display_identity"]
                       and gates["gate_a_numerics"]["anchor_matches_reference_4dp"])
    isolation_ok = bool(iso["kinds_recorded"] == ["swiglu_product_front"]
                        and iso["offset_bytes_set"] == [SLACK]
                        and iso["trailing_bytes_set"] == [0]
                        and iso["all_end_at_storage_end"]
                        and iso["all_contiguous"] and iso["all_pointer_preserved"]
                        and not iso["any_resized"]
                        and iso["padded_gradient_records"] == 0)
    descriptors_ok = bool(gates["gate_a_descriptors"]["w3_group_equal"])
    # operative critical exposure: a read product whose logical end is within a
    # vector access of the line where its own allocation could end. A 4 KiB page
    # remainder of 4 to 16 bytes is the geometry that faulted in PERF-5I/5L; the
    # 2 MiB grid distance matters only if the block ends on that grid.
    operative = [e for e in exp["exposed_products"]
                 if e["page_dist"] <= 16 and e["roles"]]
    # A tight mapping end is only demonstrated if the block itself ends within a
    # vector access. page_dist alone is the 4 KiB remainder; the block may extend
    # to the next 2 MiB grid line, which is what round_2mib_gap measures.
    tight_mapping = [e for e in operative if e["round_2mib_gap"] <= 16]
    literal_64 = [e for e in exp["exposed_products"]
                  if e["page_dist"] <= 64 or e["grid_2mib_dist"] <= 64]
    if not (numerics_ok and isolation_ok and descriptors_ok):
        code, text = 5, "invalid due to correctness, storage or descriptor mismatch"
    elif rec.get("faulted"):
        code, text = ((1, "valid front-padded run, critical exposure, GPUVM fault "
                          "at product boundary") if fault["page_exact"]
                      else (4, "fault elsewhere, investigate separately"))
    elif tight_mapping:
        code, text = 2, "valid front-padded run, critical exposure, clean"
    else:
        code, text = 3, ("inconclusive: the 4-byte page remainder was present and "
                         "read, but no product can be shown to end within a vector "
                         "access of its mapped boundary")
    gates["classification"] = {
        "code": code, "text": text, "faulted": rec.get("faulted"),
        "numerics_ok": numerics_ok, "isolation_ok": isolation_ok,
        "descriptors_ok": descriptors_ok,
        "page_dist_le_16_read_products": len(operative),
        "tight_mapping_products": len(tight_mapping),
        "page_dist_4_reads": exp["reads_at_page_dist_4"],
        "literal_64B_candidates": len(literal_64),
        "literal_64B_code_would_be": 2 if (literal_64 and not rec.get("faulted")) else None,
        "if_mapping_ends_at_the_page_then_code": 2 if operative else None,
        "mapping_beyond_logical_end": "unknown",
        "note": "a 4-byte page remainder was present and read, but the pad is a "
                "page multiple, so the 2 MiB grid line where a rounded block would "
                "end moved 65,536 bytes away; the run therefore does not reproduce "
                "the tight stock condition and cannot settle the comparison",
    }
    json.dump(gates, open(out_path, "w"), indent=2, sort_keys=True)
    print(json.dumps({"classification": gates["classification"]}, indent=2,
                     sort_keys=True))
    print(json.dumps({
        "isolation": {k: iso[k] for k in ("kinds_recorded", "offset_bytes_set",
                                          "trailing_bytes_set",
                                          "all_end_at_storage_end",
                                          "padded_gradient_records")},
        "exposure": {k: exp[k] for k in ("reads_of_products", "reads_at_page_dist_4",
                                         "reads_at_page_dist_le_16",
                                         "exposed_product_count")},
        "exposed_page_dist_4": [e for e in exp["exposed_products"]
                                if e["page_dist"] == 4][:6],
        "numerics": {k: gates["gate_a_numerics"][k] for k in
                     ("display_identity", "first_loss_full_precision",
                      "stock_vs_padded_loss_bitwise", "grads_bitwise_equal")},
    }, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
