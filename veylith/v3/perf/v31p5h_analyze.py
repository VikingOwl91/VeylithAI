"""PERF-5H gate analysis, CPU only.

Reads the PERF-5G reference case and the PERF-5H padded case, and decides the
four validity gates before any fault interpretation:

    A  GEMM descriptors (fields, not just shape)
    B  output storage geometry and no implicit resize
    C  numerical agreement with the recorded reference values
    D  execution integrity

Writes gates.json next to the padded case.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

REF = os.path.expanduser("~/veylith-perf5g-20261027/case_A")
PAD = os.path.expanduser("~/veylith-perf5h-20261027/case_H")
MIB = 2 << 20

DESC = re.compile(
    r"A=(0x[0-9a-fA-F]+) Adesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\] "
    r"B=(0x[0-9a-fA-F]+) Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\] "
    r"C=(0x[0-9a-fA-F]+) Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\] "
    r"D=(0x[0-9a-fA-F]+) Ddesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\] "
    r"computeDesc=\[computeType=(\S+) scaleType=(\S+) transA=(\S+) transB=(\S+) "
    r"epilogue=(\S+) biasPointer=(\S+)\] workSpace=0x[0-9a-fA-F]+ "
    r"workSpaceSizeInBytes=(\d+) alpha=(\S+) beta=(\S+)")
LOSS = re.compile(r"iter\s+(\d+) \| loss (\S+)")
EVAL = re.compile(r"eval \| val loss (\S+)")


def parse_log(path):
    """Return (signature_counts, targets, total_calls) for one hipBLASLt log."""
    if not path:
        return {}, [], 0
    # 0-based group indices of DESC: 0 A ptr, 1 A type, 2 A rows, 3 A cols,
    # 4 A ld, 5 B ptr, 6 B type, 7 B rows, 8 B cols, 9 B ld, 10 C ptr,
    # 11 C type, 12 C rows, 13 C cols, 14 C ld, 15 D ptr, 16 D type,
    # 17 D rows, 18 D cols, 19 D ld, 20 computeType, 21 scaleType, 22 transA,
    # 23 transB, 24 epilogue, 25 biasPointer, 26 workspace, 27 alpha, 28 beta
    signatures = {}
    targets = []
    calls = 0
    for line in open(path, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        m = DESC.search(line)
        if not m:
            continue
        calls += 1
        g = m.groups()
        sig = (g[1], int(g[2]), int(g[3]), int(g[4]),
               g[6], int(g[7]), int(g[8]), int(g[9]),
               g[11], int(g[12]), int(g[13]), int(g[14]),
               g[20], g[21], g[22], g[23], g[24],
               "bias_zero" if g[25] == "0x0" else "bias_set",
               int(g[26]), g[27], g[28])
        signatures[sig] = signatures.get(sig, 0) + 1
        if int(g[13]) == 3075:  # the w3 input-gradient GEMM group
            targets.append({"C": g[10], "rows": int(g[12]), "cols": int(g[13]),
                            "ld": int(g[14]), "dtype": g[11],
                            "Adesc": [g[1], int(g[2]), int(g[3]), int(g[4])],
                            "transA": g[22], "transB": g[23],
                            "computeType": g[20], "scaleType": g[21],
                            "epilogue": g[24], "alpha": g[27], "beta": g[28]})
    return signatures, targets, calls


def losses(stdout_path):
    values, evals = {}, []
    if not os.path.exists(stdout_path):
        return values, evals
    for line in open(stdout_path, errors="replace"):
        m = LOSS.search(line)
        if m:
            values[int(m.group(1))] = m.group(2)
        e = EVAL.search(line)
        if e:
            evals.append(e.group(1))
    return values, evals


def case_files(case_dir):
    log = sorted(glob.glob(os.path.join(case_dir, "hipblaslt-*.log")))
    rec = os.path.join(case_dir, "record.json")
    return (log[0] if log else None,
            json.load(open(rec)) if os.path.exists(rec) else {},
            os.path.join(case_dir, "geometry.json"),
            os.path.join(case_dir, "stdout.txt"),
            os.path.join(case_dir, "kernel-window.txt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--pad", default=PAD)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.pad, "gates.json")

    ref_log, ref_rec, ref_geo, ref_out, ref_win = case_files(args.ref)
    pad_log, pad_rec, pad_geo_p, pad_out, pad_win = case_files(args.pad)
    ref_sigs, ref_targets, ref_calls = parse_log(ref_log)
    pad_sigs, pad_targets, pad_calls = parse_log(pad_log)
    ref_losses, ref_evals = losses(ref_out)
    pad_losses, pad_evals = losses(pad_out)
    geo = json.load(open(pad_geo_p)) if os.path.exists(pad_geo_p) else {}

    def sig_key(s):
        # (C rows, C cols, C ld) identifies the group; report full field tuples
        return s

    ref_set, pad_set = set(ref_sigs), set(pad_sigs)
    groups = sorted({(s[9], s[10], s[11]) for s in ref_set | pad_set})
    ref_by_group, pad_by_group = {}, {}
    for s in ref_set:
        ref_by_group.setdefault((s[9], s[10], s[11]), set()).add(sig_key(s))
    for s in pad_set:
        pad_by_group.setdefault((s[9], s[10], s[11]), set()).add(sig_key(s))
    w3_group = (682, 3075, 682)
    gate_a = {
        "w3_group_in_reference": sorted(ref_by_group.get(w3_group, set())),
        "w3_group_in_padded": sorted(pad_by_group.get(w3_group, set())),
        "w3_group_equal": ref_by_group.get(w3_group) == pad_by_group.get(w3_group),
        "w3_group_counts_reference": sorted(
            [{"calls": ref_sigs[s], "Adesc": [s[0], s[1], s[2], s[3]],
              "transA": s[14], "transB": s[15]}
             for s in ref_set if (s[9], s[10], s[11]) == w3_group],
            key=lambda d: d["calls"]),
        "w3_group_counts_padded": sorted(
            [{"calls": pad_sigs[s], "Adesc": [s[0], s[1], s[2], s[3]],
              "transA": s[14], "transB": s[15]}
             for s in pad_set if (s[9], s[10], s[11]) == w3_group],
            key=lambda d: d["calls"]),
        "group_sets_equal": set(ref_by_group) == set(pad_by_group),
        "groups_only_in_padded": sorted(set(pad_by_group) - set(ref_by_group)),
        "groups_only_in_padded_meaning": "an extra GEMM shape the reference never "
                                         "issued would appear here",
        "groups_only_in_reference": sorted(set(ref_by_group) - set(pad_by_group)),
        "signature_sets_equal": ref_set == pad_set,
        "signatures_only_in_padded": sorted(pad_set - ref_set),
        "target_group243": {
            "reference": sorted(ref_by_group.get((256, 3075, 256), set())),
            "padded": sorted(pad_by_group.get((256, 3075, 256), set())),
        },
        "note": "matching descriptors do not prove identical kernel selection; "
                "hipBLASLt in this build logs no solution id",
    }

    buffers = geo.get("buffers", [])
    targets = [b for b in buffers if b.get("shape") == [3075, 682]]
    gate_b = {
        "buffers_recorded": len(buffers),
        "buffers_total": geo.get("buffers_total"),
        "all_contiguous": geo.get("all_contiguous"),
        "all_offset_zero": geo.get("all_offset_zero"),
        "all_pointer_preserved": geo.get("all_pointer_preserved"),
        "all_storage_preserved": geo.get("all_storage_preserved"),
        "any_resized_by_out": geo.get("any_resized_by_out"),
        "slack_min": geo.get("slack_min"),
        "slack_ok": geo.get("slack_min") is not None and geo["slack_min"] >= 65536,
        "target_count": len(targets),
        "target_bytes_to_page_end_min": geo.get("target_bytes_to_page_end_min"),
        "target_page_gap_ok": (geo.get("target_bytes_to_page_end_min") is not None
                               and geo["target_bytes_to_page_end_min"] > 44),
        "target_sample": targets[:2],
        "mapping_note": "page-rounded end is inferred from the storage allocation; "
                        "GPU mapping accessibility is not directly observed",
        "pointer_match_note": "a bare pointer match is unreliable because the "
                              "caching allocator reuses addresses; the match here "
                              "also requires the recorded shape to equal the "
                              "descriptor's C shape",
    }
    # The w3 input-gradient GEMM is the (682, 256, 682) OP_N variant: its weight
    # is (256, 682), so hipBLASLt sees the same bytes as a 682 x 256 column-major
    # operand. The other producer of a (3075, 682) output is w1's forward, which
    # is the (256, 682, 256) OP_T variant and is NOT padded by this harness.
    def _gap2mib(ptr, extent):
        base = int(ptr, 16) + extent
        return ((base + MIB - 1) // MIB) * MIB - base

    w3_bwd, other = [], []
    for t in pad_targets:
        extent = t["rows"] * t["cols"] * 2
        rec = [b for b in targets
               if b["data_ptr"] == t["C"] and b["shape"] == [t["cols"], t["rows"]]]
        row = {"C": t["C"], "Cdesc": [t["rows"], t["cols"], t["ld"]],
               "Adesc": t["Adesc"], "transA": t["transA"], "transB": t["transB"],
               "gap_to_next_2mib": _gap2mib(t["C"], extent),
               "matched_padded_buffer": bool(rec),
               "slack_after_tensor": rec[0]["slack_after_tensor"] if rec else None,
               "bytes_from_tensor_end_to_page_end":
                   rec[0]["bytes_from_tensor_end_to_page_end"] if rec else None}
        if t["Adesc"] == ["R_16BF", 682, 256, 682] and t["transA"] == "OP_N":
            w3_bwd.append(row)
        else:
            other.append(row)
    gate_b["w3_backward_calls"] = w3_bwd
    gate_b["w3_backward_all_padded"] = bool(w3_bwd) and all(
        r["matched_padded_buffer"] for r in w3_bwd)
    gate_b["other_wide_calls"] = other
    gate_b["other_wide_all_unpadded"] = all(not r["matched_padded_buffer"]
                                            for r in other)
    gate_b["descriptor_pointer_matches_padded_buffer"] = w3_bwd
    gate_b["descriptor_pointer_matched"] = gate_b["w3_backward_all_padded"]

    gate_c = {
        "reference_recorded_losses": ref_losses,
        "reference_recorded_evals": ref_evals,
        "padded_recorded_losses": {k: v for k, v in pad_losses.items()
                                   if k in ref_losses},
        "padded_recorded_evals": pad_evals,
        "losses_equal": all(ref_losses[k] == pad_losses.get(k) for k in ref_losses),
        "evals_equal": (all(a == b for a, b in zip(ref_evals, pad_evals))
                        if ref_evals else None),
        "padded_iterations_recorded": len(pad_losses),
        "reference_precision": "4 decimals; gradient norms were not recorded in "
                               "any reference arm, so device-level bitwise gradient "
                               "comparison is not possible from the record",
        "cpu_bitwise_equivalent": "yes, fp32, bf16 and the mixed autocast shape",
    }

    reset_lines = pad_rec.get("reset_lines") or []
    gate_d = {
        "faulted": pad_rec.get("faulted"),
        "fault_mode": pad_rec.get("fault_mode"),
        "fault_address": pad_rec.get("fault_address"),
        "exit_code": pad_rec.get("exit_code"),
        "gemm_calls_logged": pad_calls,
        "gemm_calls_reference": ref_calls,
        "wide_calls": pad_rec.get("wide_calls"),
        "wide_min_gap": pad_rec.get("wide_min_gap"),
        "wide_descriptors": pad_rec.get("wide_descriptors"),
        "padded_allocs": pad_rec.get("padded_allocs"),
        "reset_lines": reset_lines[:3],
        "gpu_idle_after": pad_rec.get("gpu_idle_after"),
        "kernel_fault_addresses": pad_rec.get("kernel_fault_addresses"),
        "kernel_fault_records": pad_rec.get("kernel_fault_records"),
        "completed": "COMPLETED" in (open(pad_out, errors="replace").read()
                                     if os.path.exists(pad_out) else ""),
    }

    gates = {"gate_a_descriptors": gate_a, "gate_b_output_storage": gate_b,
             "gate_c_numerics": gate_c, "gate_d_integrity": gate_d,
             "reference_case": args.ref, "padded_case": args.pad}
    json.dump(gates, open(out_path, "w"), indent=2, sort_keys=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk.endswith(("_ok", "_equal", "matched", "faulted",
                                          "completed", "wide_calls", "wide_min_gap",
                                          "target_count", "slack_min"))}
                      for k, v in gates.items() if isinstance(v, dict)}, indent=2,
                     sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
