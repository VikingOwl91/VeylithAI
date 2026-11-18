"""Replace a top-ranked expert 3 at punctuation pieces in frozen block 1."""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece
import torch

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from analysis.routing_study import history, load_inputs, loss_tokens, paired_stats, summarize_rows
from training.data import to_tensors


def replace_expert3(logits, punctuation):
    """Transfer E3's top weight to the best unused expert on eligible rows."""
    if logits.ndim != 2 or logits.shape[0] != punctuation.numel() or logits.shape[1] <= 3:
        raise ValueError("gate shape differs from the routing mask or lacks expert 3")
    scores, ids = logits.float().topk(2)
    eligible = punctuation.flatten() & (ids[:, 0] == 3)
    original = ids[eligible]
    available = logits[eligible].clone().scatter_(1, original, -torch.inf)
    substitute = available.argmax(dim=1)
    new_ids = torch.stack((substitute, original[:, 1]), dim=1)
    changed = torch.full_like(available, -torch.inf).scatter_(1, new_ids, scores[eligible])
    # Compare effective weights by expert ID, including equal-weight ties.
    new_scores, selected = changed.topk(2)
    actual = torch.zeros_like(changed).scatter_(1, selected, new_scores.softmax(1))
    expected = torch.zeros_like(changed).scatter_(1, new_ids, scores[eligible].softmax(1))
    if not torch.equal(actual, expected):
        raise ValueError("substitution changed the prescribed mixture weights")
    result = logits.clone()
    result[eligible] = changed
    return result, eligible.reshape_as(punctuation), substitute


@torch.inference_mode()
def evaluate_expert3(model, batches, sp):
    if (model.training or model.cfg.n_experts_per_tok != 2 or model.cfg.moe_impl != "dense"
            or not model.blocks[1].is_moe):
        raise ValueError("requires dense-combined top-2 evaluation in block 1")
    gate = model.blocks[1].mlp.gate
    if gate.weight.device.type != "cpu" or gate.weight.dtype != torch.float32:
        raise ValueError("requires a CPU float32 gate")
    _, pieces = history._piece_tables(sp)
    names = ("all", "sentence_end", "replaced", "first_replaced_in_sequence", "unaffected_prefix")
    counts, base_sums, delta_sums, sources = [], [], [], []
    replacements = torch.zeros(model.cfg.n_experts, dtype=torch.long)
    for number, (source, batch) in enumerate(batches, start=1):
        x, y, valid = to_tensors(batch, "cpu")
        punctuation = torch.from_numpy(pieces["sent_end"][x.numpy()]) & valid
        def ce():
            logits, _ = model(x)
            return loss_tokens(logits, y)

        baseline = ce()
        routes = []
        def intervene(module, args, logits):
            changed, eligible, substitute = replace_expert3(logits, punctuation)
            routes.append((eligible, substitute))
            return changed

        with gate.register_forward_hook(intervene):
            treated = ce()
        (eligible, substitute), = routes
        replacements += substitute.bincount(minlength=model.cfg.n_experts)
        seen = eligible.int().cumsum(dim=1)
        prefix = valid & (seen == 0)
        if not torch.isfinite(treated[valid]).all() or not torch.isfinite(baseline[valid]).all():
            raise ValueError(f"nonfinite loss on batch {number}")
        if not torch.equal(treated[prefix], baseline[prefix]):
            raise ValueError(f"loss changed before the first intervention on batch {number}")
        masks = (valid, punctuation, eligible, eligible & (seen == 1), prefix)
        delta = treated.double() - baseline.double()
        counts.append(np.column_stack([m.sum(1).numpy() for m in masks]))
        for values, rows in ((baseline, base_sums), (delta, delta_sums)):
            rows.append(np.column_stack([(values * m).sum(1, dtype=torch.float64).numpy() for m in masks]))
        sources.extend([source] * len(x))
        if number == 1 or number % 5 == 0:
            print(f"Batch {number}/{len(batches)} complete", flush=True)

    counts = np.concatenate(counts)
    baseline_means = summarize_rows(np.concatenate(base_sums), counts)
    estimates = paired_stats(np.concatenate(delta_sums), counts, np.asarray(sources))
    return {"sequences": len(sources), "substitute_counts": replacements.tolist(),
            "unaffected_prefix_ce_exact": True,
            "outcomes": {k: {"targets": int(counts[:, i].sum()),
                             "sequences": int((counts[:, i] > 0).sum()),
                             "baseline_ce": float(baseline_means[i]) if counts[:, i].any() else None,
                             **estimates[i]} for i, k in enumerate(names)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fresh-sample", action="store_true")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output filename")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    model, batches, sp, hashes = load_inputs(args.data_root, args.checkpoint, args.fresh_sample)
    suffix = "-confirmation" if args.fresh_sample else ""
    anchor = json.loads((V3 / "docs" / f"routing-sentence-end-update500{suffix}.json").read_text())
    if anchor["sha256"] != hashes or anchor["config"] != model.cfg.to_dict():
        raise ValueError("input hashes or model config differ from the reference")
    files = ("analysis/expert3_study.py", "analysis/routing_study.py", "analysis/v31n_expert_behavior.py",
             "architecture/model.py", "architecture/config.py", "training/data.py")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(V3), *args], text=True).strip()

    report = {"experiment": "v31-expert3-rank1-to-best-unused", "schema_version": 1, "update": 500,
              "layer": 1, "split": "validation", "sample_batch_offset_per_source": anchor["sample_batch_offset_per_source"],
              "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
              "config": model.cfg.to_dict(), "sha256": hashes,
              "source_sha256": {f: history.sha256_file(V3 / f) for f in files},
              "reproduction_command": 'python veylith/v3/analysis/expert3_study.py --data-root "$DATA" '
                                      '--checkpoint "$CKPT" --out "$OUT" '
                                      + f'--threads {args.threads}' + (' --fresh-sample' if args.fresh_sample else ''),
              "runtime": {"device": "cpu", "dtype": "float32", "python": platform.python_version(),
                          "torch": str(torch.__version__), "numpy": np.__version__, "sentencepiece": sentencepiece.__version__,
                          "threads": args.threads, "interop_threads": 1},
              "primary": "replaced", "delta": "substituted minus ordinary next-token CE",
              "bootstrap": {"unit": "sequence", "strata": "source", "draws": 1000, "seed": 20261118}}
    report.update(evaluate_expert3(model, batches, sp))
    for name in ("all", "sentence_end"):
        actual, expected = report["outcomes"][name], anchor["baseline"][name]
        if actual["baseline_ce"] != expected["ce"] or actual["targets"] != expected["targets"]:
            raise ValueError(f"{name} baseline differs from the reference")
    report["baseline_reference_exact"] = True
    report["elapsed_seconds"] = time.monotonic() - start
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as f:
        f.write(payload)


if __name__ == "__main__":
    main()
