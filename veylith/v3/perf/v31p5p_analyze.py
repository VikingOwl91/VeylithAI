"""PERF-5P analysis: join tagged producers and allocator lifetimes to the reads.

CPU only, after the single GPU execution. Produces gates.json with the operand
census, the tail-4 attribution, lifetime confidence and the classification.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
import re

MIB = 2 << 20
PAGE = 4096
NB = 4194300
REF = os.path.expanduser("~/veylith-perf5g-20261027/case_A/stdout.txt")

A = re.compile(r"A=(0x[0-9a-fA-F]+) Adesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
B = re.compile(r"B=(0x[0-9a-fA-F]+) Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
C = re.compile(r"C=(0x[0-9a-fA-F]+) Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)\]")
TRANS = re.compile(r"transA=(\S+) transB=(\S+)")
LOSS = re.compile(r"iter\s+(\d+) \| loss (\S+)")
EVAL = re.compile(r"eval \| val loss (\S+)")


def losses(path):
    values, evals = {}, []
    for line in open(path, errors="replace"):
        m = LOSS.search(line)
        if m:
            values[int(m.group(1))] = m.group(2)
        e = EVAL.search(line)
        if e:
            evals.append(e.group(1))
    return values, evals


def read_events(case):
    path = os.path.join(case, "prov-events.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def intervals_from(snap_path):
    if not snap_path or not os.path.exists(snap_path):
        return [], "no snapshot"
    with open(snap_path, "rb") as fh:
        snap = pickle.load(fh)
    trace = snap.get("device_traces") or []
    if not trace:
        return [], "no trace"
    open_at, out = {}, []
    for e in trace[0]:
        if e.get("action") == "alloc":
            open_at.setdefault((e["addr"], e["size"]), []).append(e.get("time_us"))
        elif e.get("action") == "free_completed":
            lst = open_at.get((e["addr"], e["size"]))
            if lst:
                out.append({"addr": hex(e["addr"]), "block_size": e["size"],
                            "alloc_t": lst.pop(0), "free_t": e.get("time_us")})
    for (addr, size), lst in open_at.items():
        for t0 in lst:
            out.append({"addr": hex(addr), "block_size": size, "alloc_t": t0,
                        "free_t": None})
    blocks = [b for s in snap.get("segments", []) for b in s.get("blocks", [])
              if b.get("requested_size") == NB]
    return out, {"intervals": len(out), "blocks_4194300": len(blocks),
                 "block_sizes": sorted({b["size"] for b in blocks}),
                 "captured_frames_top": [f.get("name") for f in
                                         (blocks[0].get("frames") or [])[:3]] if blocks else []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=os.path.expanduser("~/veylith-perf5p-20261027/case_H"))
    args = ap.parse_args()
    case = args.case
    rec = json.load(open(os.path.join(case, "record.json")))
    prov = json.load(open(os.path.join(case, "prov.json")))
    ev = read_events(case)
    tags = {(e["ptr"], e["nbytes"]): e for e in ev if e.get("kind") == "producer"}
    log = glob.glob(os.path.join(case, "hipblaslt-*.log"))[0]

    reads = collections.defaultdict(list)
    for line in open(log, errors="replace"):
        if "rocblaslt_matmul" not in line:
            continue
        for role, rx in (("A", A), ("B", B)):
            m = rx.search(line)
            if not m:
                continue
            rws, cols, ld = int(m.group(3)), int(m.group(4)), int(m.group(5))
            if ld != rws:
                continue
            nb = rws * cols * 2
            if nb != NB:
                continue
            ptr = m.group(1)
            end = int(ptr, 16) + nb
            c, tr = C.search(line), TRANS.search(line)
            reads[ptr].append({
                "role": role,
                "page_dist": ((end + PAGE - 1) // PAGE) * PAGE - end,
                "grid_2mib_dist": ((end + MIB - 1) // MIB) * MIB - end,
                "Cdesc": [int(c.group(3)), int(c.group(4))] if c else None,
                "trans": list(tr.groups()) if tr else None,
                "stamp": line[1:20]})

    census = []
    for ptr, lst in reads.items():
        tag = tags.get((ptr, NB))
        census.append({
            "ptr": ptr, "reads": len(lst),
            "page_dist": lst[0]["page_dist"], "grid_2mib_dist": lst[0]["grid_2mib_dist"],
            "shapes": sorted({f"C{c['Cdesc']} {c['trans']}" for c in lst}),
            "op_n_reads": sum(1 for c in lst if c["trans"] and c["trans"][1] == "OP_N"),
            "op_t_reads": sum(1 for c in lst if c["trans"] and c["trans"][1] == "OP_T"),
            "producer_observed": tag["producer"] if tag else None,
            "producer_phase": tag["phase"] if tag else None,
            "provenance": "observed" if tag else "unknown",
        })
    tail4 = [c for c in census if c["page_dist"] == 4 and c["grid_2mib_dist"] == 4]

    snap_path = None
    snaps = [e for e in ev if e.get("kind") == "snapshot" and e.get("path")]
    if snaps:
        snap_path = snaps[0]["path"]          # earliest kept snapshot
    intervals, lifetime_meta = intervals_from(snap_path)
    for c in tail4:
        ivs = [i for i in intervals if i["addr"] == c["ptr"]]
        c["lifetime"] = {"intervals_at_addr": len(ivs),
                         "detail": ivs[:3],
                         "confidence": ("allocation observed; free not observed in "
                                        "the retained history" if ivs else
                                        "lifetime unknown: not retained in the "
                                        "bounded history")}

    ref_losses, ref_evals = losses(REF)
    pad_losses, pad_evals = losses(os.path.join(case, "stdout.txt"))
    numerics = {
        "display_identity": bool(all(ref_losses[k] == pad_losses.get(k)
                                     for k in ref_losses)
                                 and all(a == b for a, b in zip(ref_evals, pad_evals))),
        "reference_losses": ref_losses, "padded_losses": pad_losses,
        "reference_evals": ref_evals, "padded_evals": pad_evals,
        "first_loss_full_precision": prov.get("first_loss"),
        "first_loss_reference_expectation": 10.481635093688965,
        "first_loss_matches_expectation":
            prov.get("first_loss") == 10.481635093688965,
        "first_loss_note": "value captured at iteration 0 and compared against the "
                           "reference expectation recorded in earlier stages",
    }
    overhead = {
        "events": len(ev), "tagged_producers": len(tags),
        "snapshots": len([e for e in ev if e.get("kind") == "snapshot"]),
        "snapshot_bytes": sorted(os.path.getsize(os.path.join(case, f))
                                 for f in os.listdir(case)
                                 if f.startswith("prov-snap")),
        "alloc_history": prov.get("alloc_history"), "truncated": prov.get("truncated"),
        "get_batch_calls": prov.get("get_batch_calls"),
    }

    if not numerics["display_identity"]:
        code, text = 5, "invalid: numerical drift"
    elif rec.get("faulted"):
        established = [c for c in tail4 if c["provenance"] == "observed"
                       and c["lifetime"]["intervals_at_addr"] > 0]
        code, text = ((1, "fault reproduced; producer and lifetime identified")
                      if established else
                      (2, "fault reproduced; provenance ambiguous"))
    elif tail4:
        code, text = 3, ("no fault, with the critical geometry present and read "
                         "in this run")
    else:
        code, text = 3, "no fault and no critical exposure in this run"
    gates = {
        "meta": {"experiment": "v3.1-PERF-5P stock BF16 operand provenance",
                 "gpu_executions": 1, "classification": code,
                 "classification_text": text},
        "execution": rec,
        "numerics": numerics,
        "overhead": overhead,
        "census": census,
        "tail4_candidates": tail4,
        "lifetime_history": lifetime_meta,
        "tagged_producers": sorted({(t["producer"], t["phase"]) for t in tags.values()}),
        "classification": {"code": code, "text": text, "faulted": rec.get("faulted"),
                           "tail4_present": bool(tail4)},
    }
    out = os.path.join(case, "gates.json")
    json.dump(gates, open(out, "w"), indent=2, sort_keys=True)
    print(json.dumps({"classification": gates["classification"],
                      "numerics": {k: numerics[k] for k in
                                   ("display_identity", "first_loss_full_precision")},
                      "overhead": overhead,
                      "tail4": [{k: c[k] for k in
                                 ("ptr", "reads", "op_n_reads", "op_t_reads",
                                  "producer_observed", "producer_phase",
                                  "provenance")} for c in tail4],
                      "lifetime_history": lifetime_meta},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
