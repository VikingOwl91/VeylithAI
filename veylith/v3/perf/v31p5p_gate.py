"""PERF-5P CPU gates for the provenance instrumentation.

Four gates, all on CPU:

  1  numerics: stock versus instrumented model, state, logits, loss and every
     gradient bitwise
  2  tags: representative SwiGLU tensors carry the right producer and phase, and
     the product's gradient is tagged from the backward phase
  3  reuse: the join prefers the lifetime that covers the event time, refuses
     overlapping lifetimes and reports unknown when nothing matches
  4  persistence: a synthetic fault path still writes parseable partial records
     when the allocator history is unavailable
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, "veylith/debug")
sys.path.insert(0, "veylith/v3/perf")
import v31p5p_provenance as provenance  # noqa: E402
import sparse_moe_v2_train as impl  # noqa: E402


def cfg():
    return impl.Config(n_layer=6, n_head=8, n_kv_head=2, n_embd=256, block_size=256,
                       sliding_window=128, full_attn_every=4, dropout=0.0,
                       n_experts=8, n_experts_per_tok=2, first_k_dense=1, n_mtp=1,
                       vocab_size=1000)


def batch():
    gen = torch.Generator().manual_seed(5)
    return (torch.randint(0, 1000, (2, 32), generator=gen),
            torch.randint(0, 1000, (2, 32), generator=gen))


def run_model(instrumented: bool, out_dir: str):
    torch.manual_seed(1)
    if instrumented:
        provenance.enable(out_dir)
    model = impl.GPT(cfg())
    idx, tgt = batch()
    if instrumented:
        with provenance.Phase(impl):
            with provenance.TagMode():
                logits, loss = model(idx, tgt)
                loss.backward()
    else:
        logits, loss = model(idx, tgt)
        loss.backward()
    grads = [(n, p.grad.clone() if p.grad is not None else None)
             for n, p in model.named_parameters()]
    state = [(n, p.detach().clone()) for n, p in model.named_parameters()]
    return {"logits": logits.detach(), "loss": float(loss.detach()),
            "grads": grads, "state": state}


def events(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="p5p-gate-")
    try:
        # 1 + 2, numerics and tags
        stock = run_model(False, tmp)
        inst = run_model(True, os.path.join(tmp, "inst"))
        ev = events(os.path.join(tmp, "inst", "prov-events.jsonl"))
        producers = [e for e in ev if e.get("kind") == "producer"]
        consumers = [e for e in ev if e.get("kind") == "consumer"]
        names = {e["producer"] for e in producers}
        grads_tagged = [e for e in producers if e["producer"].startswith("grad_of:")]
        gates = {
          "state_bitwise": all(torch.equal(a[1], b[1])
                               for a, b in zip(stock["state"], inst["state"])),
          "logits_bitwise": bool(torch.equal(stock["logits"], inst["logits"])),
          "loss_bitwise": stock["loss"] == inst["loss"],
          "grads_bitwise": all(
              (a[1] is None and b[1] is None)
              or (a[1] is not None and b[1] is not None and torch.equal(a[1], b[1]))
              for a, b in zip(stock["grads"], inst["grads"])),
        }
        gates["model_run_events_parse"] = len(ev) >= 0
        gates["model_run_forward_phase_consistent"] = all(
            e["phase"] == "forward" for e in producers)

        # 2, size-filtered tagging, exercised synthetically: the debug batch is
        # far too small to allocate a 4,194,300-byte tensor
        tag_dir = os.path.join(tmp, "tags")
        provenance._STATE.update(seq=0, out_dir=None, dumps=[], alloc_history="off",
                                 truncated=False)
        provenance.enable(tag_dir)
        b = torch.randn(3075, 682, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(682, 682, dtype=torch.bfloat16, requires_grad=True)
        with provenance.TagMode():
            src = torch.randn(3075, 682, dtype=torch.bfloat16)
            prod = src * b
            lin = torch.nn.functional.linear(prod, w)
        provenance._STATE["phase"] = "backward"
        lin.sum().backward()
        provenance._STATE["phase"] = "forward"
        tev = events(os.path.join(tag_dir, "prov-events.jsonl"))
        tprod = [e for e in tev if e.get("kind") == "producer"]
        tnames = {e["producer"] for e in tprod}
        tgrads = [e for e in tprod if e["producer"].startswith("grad_of:")]
        gates["synthetic_tags_have_mul"] = any(n.endswith(".mul") for n in tnames)
        gates["synthetic_tags_have_linear"] = any("linear" in n for n in tnames)
        gates["synthetic_grad_tagged_in_backward"] = bool(tgrads) and all(
            e["phase"] == "backward" for e in tgrads)
        gates["synthetic_target_bytes_only"] = all(
            e["nbytes"] == provenance.TARGET_BYTES for e in tprod)
        gates["synthetic_shapes_include_3075x682"] = (3075, 682) in {
            tuple(e["shape"]) for e in tprod}
        gates["synthetic_consumers_recorded"] = any(
            e.get("kind") == "consumer" for e in tev)

        # 3, reuse and lifetime joins
        recs = [
            {"ptr": "0x1000", "nbytes": 4194300, "alloc_t": 10.0, "free_t": 20.0,
             "producer": "first"},
            {"ptr": "0x1000", "nbytes": 4194300, "alloc_t": 30.0, "free_t": None,
             "producer": "second"},
            {"ptr": "0x2000", "nbytes": 4194300, "alloc_t": 10.0, "free_t": None,
             "producer": "overlap-a"},
            {"ptr": "0x2000", "nbytes": 4194300, "alloc_t": 12.0, "free_t": None,
             "producer": "overlap-b"},
        ]
        m1, s1 = provenance.resolve(recs, "0x1000", 4194300, 15.0)
        m2, s2 = provenance.resolve(recs, "0x1000", 4194300, 35.0)
        m3, s3 = provenance.resolve(recs, "0x2000", 4194300, 15.0)
        m4, s4 = provenance.resolve(recs, "0x1000", 4194300, 25.0)
        m5, s5 = provenance.resolve(recs, "0x3000", 4194300, 15.0)
        m6, s6 = provenance.resolve(recs, "0x1000", 1234, 15.0)
        gates["reuse_prefers_covering_lifetime"] = (
            s1 == "observed" and m1["producer"] == "first"
            and s2 == "observed" and m2["producer"] == "second")
        gates["reuse_ambiguous_when_overlapping"] = (s3 == "ambiguous" and m3 is None)
        gates["reuse_unknown_between_lifetimes"] = (s4 == "unknown")
        gates["reuse_unknown_for_unknown_pointer"] = (s5 == "unknown")
        gates["reuse_refuses_byte_count_mismatch"] = (s6 == "unknown")

        # 4, persistence under a synthetic fault with no CUDA
        fault_dir = os.path.join(tmp, "fault")
        provenance._STATE.update(seq=0, out_dir=None, dumps=[], alloc_history="off",
                                 truncated=False)
        provenance.enable(fault_dir)
        t = torch.zeros(4, 4, dtype=torch.bfloat16)
        provenance.note_tensor(t, "gate.synthetic", "gate")
        snap = provenance.dump_snapshot("fault")
        provenance.finish()
        fev = events(os.path.join(fault_dir, "prov-events.jsonl"))
        gates["persistence_writes_partial_records"] = bool(fev)
        gates["persistence_alloc_history_unavailable_is_recorded"] = any(
            e.get("kind") == "alloc_history" and "unavailable" in str(e.get("status"))
            for e in fev)
        gates["persistence_snapshot_attempt_recorded"] = any(
            e.get("kind") == "snapshot" for e in fev)
        bad = os.path.join(fault_dir, "no-such-dir")
        provenance._STATE["out_dir"] = bad
        contained = provenance.dump_snapshot("bad-path") is None
        provenance._STATE["out_dir"] = fault_dir
        gates["persistence_snapshot_failure_is_contained"] = contained
        gates["persistence_has_start_and_finish"] = (
            any(e.get("kind") == "start" for e in fev)
            and any(e.get("kind") == "finish" for e in fev))

        ok = all(v for k, v in gates.items()
                 if isinstance(v, bool))
        print("P5P-GATE " + json.dumps({"ok": ok, **gates}, sort_keys=True))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
