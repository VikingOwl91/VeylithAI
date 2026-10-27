"""PERF-5I analysis, CPU only.

Runs the PERF-5H gate analysis, then adds what 5I needs beyond it:

    full-precision iteration-0 loss (loop value, stock recompute, padded
    recompute), the 4-decimal anchor against the reference, the phase split of
    the hipBLASLt log (loop versus post-loop check), and producer attribution
    between w1 forward outputs and w3 backward input gradients.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5h_analyze as base  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5i-20261027")
CASE = os.path.join(SCRATCH, "case_H")
REF = os.path.expanduser("~/veylith-perf5g-20261027/case_A")
MIB = 2 << 20
STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
C_DESC = re.compile(r"C=(0x[0-9a-fA-F]+) Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) "
                    r"ld=(\d+)\]")
A_DESC = re.compile(r"A=(0x[0-9a-fA-F]+) Adesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
TRANS = re.compile(r"transA=(\S+) transB=(\S+)")
B_DESC = re.compile(r"B=(0x[0-9a-fA-F]+) Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
W3_BACKWARD = (682, 256, 682)   # weight (256, 682) seen as a 682 x 256 operand
W1_FORWARD = (256, 682, 256)    # weight (682, 256) seen as a 256 x 682 operand


def phase_split(log_path, started_local):
    """Return (loop_lines, postcheck_lines, boundary_second_lines)."""
    loop, post, boundary = [], [], 0
    if not log_path:
        return loop, post, boundary
    if not started_local:
        # No marker means post_check never ran (fault or crash), so every line
        # belongs to the loop phase.
        loop = [ln for ln in open(log_path, errors="replace")
                if "rocblaslt_matmul" in ln]
        return loop, post, boundary
    for line in open(log_path, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        m = STAMP.match(line)
        stamp = m.group(1) if m else None
        if stamp is None:
            loop.append(line)
        elif stamp < started_local:
            loop.append(line)
        elif stamp > started_local:
            post.append(line)
        else:
            boundary += 1
            loop.append(line)
    return loop, post, boundary


def wide_producers(lines, padded_by_ptr):
    out = []
    for line in lines:
        c, a = C_DESC.search(line), A_DESC.search(line)
        if not c or not a or int(c.group(4)) != 3075:
            continue
        ptr = c.group(1)
        rows, cols, ld = int(c.group(3)), int(c.group(4)), int(c.group(5))
        adesc = (int(a.group(3)), int(a.group(4)), int(a.group(5)))
        trans = TRANS.search(line).groups()
        extent = rows * cols * 2
        end = int(ptr, 16) + extent
        rec = padded_by_ptr.get(ptr)
        same_shape = bool(rec) and rec["shape"] == [cols, rows]
        # The output width is what separates the two producers: the w3 backward
        # input gradient is (M, 682); w1 and w2 forward outputs are also (M, 682)
        # and share one descriptor, so they are not distinguishable from each
        # other in this log. Anything else with cols=3075 writes (M, 256).
        out_rows = rows
        if out_rows == 682 and adesc == W3_BACKWARD and trans[0] == "OP_N":
            producer = "w3_backward_input_grad"
        elif out_rows == 682 and adesc == W1_FORWARD and trans[0] == "OP_T":
            producer = "forward_output_mx682"
        else:
            producer = f"other_output_rows{out_rows}"
        out.append({
            "producer": producer, "Adesc": list(adesc),
            "transA": trans[0], "transB": trans[1],
            "C": ptr, "Cdesc": [rows, cols, ld],
            "padded_view": same_shape,
            "gap_to_next_2mib": ((end + MIB - 1) // MIB) * MIB - end,
            "gap_to_next_4kib": ((end + 4095) // 4096) * 4096 - end,
            "slack_after_tensor": rec["slack_after_tensor"] if same_shape else None,
            "bytes_to_page_rounded_storage_end":
                rec["bytes_from_tensor_end_to_page_end"] if same_shape else None,
        })
    return out


def _r4k(n):
    return ((n + 4095) // 4096) * 4096


def _r2mib(n):
    return ((n + MIB - 1) // MIB) * MIB


def fault_attribution(pages, log_path, rows, padded_by_ptr):
    """Which buffer's page-rounded end coincides with the fault page."""
    result = {"pages": list(pages or []), "page_exact_reads": [],
              "page_exact_stores": [], "block_exact_count": 0,
              "padded_containment": 0, "nearest_padded_gap_bytes": None}
    if not pages or not log_path:
        return result
    for pstr in pages:
        P = int(pstr, 16)
        for r in rows:
            sbase = int(r["storage_ptr"], 16)
            if sbase <= P < sbase + r["storage_bytes"]:
                result["padded_containment"] += 1
        gaps = []
        for r in rows:
            sbase = int(r["storage_ptr"], 16)
            send = sbase + r["storage_bytes"]
            gaps.append(0 if sbase <= P <= send else min(abs(P - sbase), abs(P - send)))
        if gaps:
            result["nearest_padded_gap_bytes"] = min(gaps)
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
                if _r2mib(end) == P:
                    result["block_exact_count"] += 1
                if _r4k(end) != P:
                    continue
                rec = padded_by_ptr.get(ptr)
                shape_match = bool(rec) and rec["shape"] == [cols, rws]
                entry = {"page": pstr, "role": role, "ptr": ptr,
                         "descriptor": [rws, cols, ld],
                         "tensor_end": hex(end), "padded_view": shape_match,
                         "slack_after_tensor": rec["slack_after_tensor"]
                         if shape_match else None,
                         "stamp": line[1:20]}
                (result["page_exact_reads"] if role in ("A", "B")
                 else result["page_exact_stores"]).append(entry)
    return result


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
    ref_losses, ref_evals = base.losses(os.path.join(args.ref, "stdout.txt"))
    pad_losses, pad_evals = base.losses(os.path.join(args.pad, "stdout.txt"))
    geo = json.load(open(os.path.join(args.pad, "geometry.json")))
    pc = geo.get("post_check") or {}
    log = sorted(f for f in os.listdir(args.pad) if f.startswith("hipblaslt-"))
    log_path = os.path.join(args.pad, log[0]) if log else None
    padded_by_ptr = {}
    for row in geo.get("buffers", []):
        padded_by_ptr.setdefault(row["data_ptr"], row)

    loop_lines, post_lines, boundary = phase_split(
        log_path, pc.get("postcheck_started_local"))
    wide = wide_producers(loop_lines, padded_by_ptr)
    post_wide = wide_producers(post_lines, padded_by_ptr)

    start_epoch = pc.get("postcheck_started_epoch")
    loop_allocs = [r for r in geo.get("buffers", [])
                   if start_epoch is None or r.get("t", 0) < start_epoch]
    target_rows = [r for r in loop_allocs if r["shape"] == [3075, 682]]

    gates["gate_a_numerics_full_precision"] = {
        "iter0_loss_loop": pc.get("iter0_loss_loop"),
        "iter0_loss_loop_4dp": pc.get("iter0_loss_loop_4dp"),
        "reference_recorded_4dp": "10.4816",
        "reference_display_losses": ref_losses,
        "padded_display_losses": pad_losses,
        "reference_display_evals": ref_evals,
        "padded_display_evals": pad_evals,
        "display_values_compared": len(set(ref_losses) & set(pad_losses)),
        "display_identity": bool(all(ref_losses[k] == pad_losses.get(k)
                                     for k in ref_losses)
                                 and all(a == b for a, b in zip(ref_evals, pad_evals))),
        "anchor_4dp_matches": (pad_losses.get(0) == ref_losses.get(0)
                              if pad_losses else None),
        "padded_loss_recomputed": pc.get("padded_loss_recomputed"),
        "stock_loss_recomputed": pc.get("stock_loss_recomputed"),
        "padded_loss_equals_loop_loss": pc.get("padded_loss_equals_loop_loss"),
        "stock_vs_padded_loss_bitwise": pc.get("stock_vs_padded_loss_bitwise"),
        "grad_names_equal": pc.get("grad_names_equal"),
        "grads_bitwise_equal": pc.get("grads_bitwise_equal"),
        "grad_norm_equal": pc.get("grad_norm_equal"),
        "grad_norm_experts_padded": pc.get("padded_grad_norm_experts"),
        "grad_norm_experts_stock": pc.get("stock_grad_norm_experts"),
        "grads_compared": pc.get("padded_grads_captured"),
        "model_state_unchanged_after_loop": pc.get("model_state_unchanged_after_loop"),
        "optimizer_state": "AdamW is constructed but --no_optimizer never steps, "
                           "so no optimizer state exists to compare",
        "reference_underlying_value": "never recorded by any reference arm",
        "full_precision_available_this_run": bool(pc.get("iter0_loss_loop")),
        "full_precision_note": "the loop faulted before the post-loop recompute, so "
                               "the captured full-precision loss was never persisted; "
                               "display identity is what this run can show",
        "errors": [pc[k] for k in ("padded_pass_error", "stock_pass_error", "error")
                   if k in pc],
    }

    gates["fault_attribution"] = fault_attribution(
        json.load(open(os.path.join(args.pad, "record.json"))).get("kernel_fault_addresses"),
        log_path, geo.get("buffers", []), padded_by_ptr)

    gates["producer_attribution"] = {
        "phase_split": {
            "postcheck_started_local": pc.get("postcheck_started_local"),
            "loop_gemm_lines": len(loop_lines),
            "postcheck_gemm_lines": len(post_lines),
            "boundary_second_lines": boundary,
            "loop_allocations": len(loop_allocs),
            "allocations_recorded": len(geo.get("buffers", [])),
        },
        "loop_wide_calls": wide,
        "postcheck_wide_calls": post_wide,
        "loop_w3_backward_all_padded": bool(wide) and all(
            r["padded_view"] for r in wide if r["producer"] == "w3_backward_input_grad"),
        "loop_stock_producer_min_gap_2mib": min(
            (r["gap_to_next_2mib"] for r in wide if not r["padded_view"]), default=None),
        "loop_any_stock_producer_within_64B": any(
            r["gap_to_next_2mib"] <= 64 for r in wide if not r["padded_view"]),
        "loop_any_w3_backward_within_64B": any(
            r["gap_to_next_2mib"] <= 64 for r in wide
            if r["producer"] == "w3_backward_input_grad"),
        "target_allocations_loop_phase": len(target_rows),
        "target_page_distances": [r["bytes_from_tensor_end_to_page_end"]
                                  for r in target_rows],
        "attribution_basis": "GEMM descriptor signature plus operation phase, with "
                             "the allocation record matched by pointer and shape",
    }

    ga = gates["gate_a_descriptors"]
    gb = gates["gate_b_output_storage"]
    gd = gates["gate_d_integrity"]
    an = gates["gate_a_numerics_full_precision"]
    numerics_ok = bool(an["display_identity"])
    descriptors_ok = bool(ga["w3_group_equal"])
    storage_ok = bool(gb["slack_ok"] and gb["all_offset_zero"]
                      and gb["all_pointer_preserved"] and gb["all_storage_preserved"]
                      and gb["any_resized_by_out"] is False
                      and gb["target_count"] > 0 and gb["target_page_gap_ok"]
                      and gb["w3_backward_all_padded"])
    producer_ok = bool(target_rows) and bool(gates["producer_attribution"]
                                            ["loop_w3_backward_all_padded"])
    if not numerics_ok:
        code, text = 3, "invalid due to numerical mismatch"
    elif not (descriptors_ok and storage_ok):
        code, text = 4, "invalid due to GEMM or allocation mismatch"
    elif not producer_ok:
        code, text = 5, "inconclusive due to insufficient producer attribution"
    elif gd["faulted"]:
        code, text = 2, "valid corrected padded execution, faulted"
    else:
        code, text = 1, "valid corrected padded execution, clean"
    gates["classification"] = {
        "code": code, "text": text,
        "numerics_ok": numerics_ok, "descriptors_ok": descriptors_ok,
        "storage_ok": storage_ok, "producer_attribution_ok": producer_ok,
        "faulted": gd["faulted"], "completed": gd["completed"],
        "numerics_scope": "display precision on every overlapping printed value; "
                          "full-precision comparison did not run because the loop "
                          "faulted first",
    }
    json.dump(gates, open(out_path, "w"), indent=2, sort_keys=True)
    print(json.dumps({"classification": gates["classification"],
                      "numerics": {k: an[k] for k in
                                   ("iter0_loss_loop", "anchor_4dp_matches",
                                    "stock_vs_padded_loss_bitwise",
                                    "grads_bitwise_equal", "grads_compared")},
                      "producers": {"loop_wide": len(wide),
                                    "stock_min_gap_2mib":
                                        gates["producer_attribution"]
                                        ["loop_stock_producer_min_gap_2mib"],
                                    "w3_backward_padded":
                                        gates["producer_attribution"]
                                        ["loop_w3_backward_all_padded"]}},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
