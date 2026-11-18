"""CPU next-token evaluation on the frozen v3.1-N validation sample."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from analysis import v31n_expert_behavior as history
from architecture.config import ModelConfig
from architecture.model import GPT
from training.data import iter_source_batches, to_tensors


def verified_hash(path, expected):
    actual = history.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path.name} SHA-256 differs from the frozen record")
    return actual


def load_inputs(root, checkpoint, fresh_sample=False):
    docs = V3 / "docs"
    reference = json.loads((docs / "v31n-expert-behavior-summary.json").read_text())
    corpus = json.loads((docs / "dense-run-v31l-2026-10-27-summary.json").read_text())["dataset"]
    manifest = root / "corpus" / "manifest.json"
    hashes = {"manifest": verified_hash(manifest, corpus["manifest_sha256"]),
              "checkpoint": verified_hash(checkpoint, reference["checkpoint_integrity"]["500"]["sha256"])}
    manifest_data = json.loads(manifest.read_text())
    tokenizer = manifest.parent / manifest_data["tokenization"]["tokenizer"]["model"]
    hashes["tokenizer"] = verified_hash(tokenizer, corpus["tokenizer_sha256"])
    sample = reference["sample"]
    batches = []
    digest = hashlib.sha256()
    reference_digest = hashlib.sha256()
    reference_rows, selected_rows = set(), set()
    n_batches = history.BATCHES_PER_SOURCE
    for source in sample["sources"]:
        source_batches = list(iter_source_batches(str(root), "validation", source, history.SEQ_LEN,
                                                  history.BATCH, n_batches * (2 if fresh_sample else 1)))
        for batch in source_batches[:n_batches]:
            reference_digest.update(batch["tokens"].tobytes())
            if fresh_sample:
                reference_rows.update(hashlib.sha256(row.tobytes()).digest() for row in batch["tokens"])
        if fresh_sample:
            source_batches = source_batches[n_batches:]
        if sum(len(b["lengths"]) for b in source_batches) != sample["sequences_per_source"]:
            raise ValueError(f"unexpected sequence count for {source}")
        for batch in source_batches:
            digest.update(batch["tokens"].tobytes())
            if fresh_sample:
                selected_rows.update(hashlib.sha256(row.tobytes()).digest() for row in batch["tokens"])
            if any(length != sample["seq_len"] for length in batch["lengths"]):
                raise ValueError("sequence length differs from the frozen sample")
            batches.append((source, batch))
    hashes["sample_tokens"] = digest.hexdigest()
    if reference_digest.hexdigest() != sample["digest_sha256"]:
        raise ValueError("sample token SHA-256 differs from the frozen record")

    if fresh_sample and reference_rows & selected_rows:
        raise ValueError("follow-up sample repeats a reference sequence")

    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = GPT(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    return model, batches, sp, hashes


def loss_tokens(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)


def summarize_rows(rows, counts):
    n = counts.sum(axis=0)
    return np.divide(rows.sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)


def paired_stats(deltas, counts, groups, draws=1000):
    """Stratified sequence-paired bootstrap of token-weighted mean deltas."""
    rng = np.random.default_rng(20261118)
    strata = [np.flatnonzero(groups == source) for source in np.unique(groups)]
    estimates = []
    for _ in range(draws):
        sample = np.concatenate([rng.choice(rows, size=rows.size, replace=True) for rows in strata])
        estimates.append(summarize_rows(deltas[sample], counts[sample]))
    estimates = np.asarray(estimates)
    result = []
    for col, mean in enumerate(summarize_rows(deltas, counts)):
        finite = estimates[np.isfinite(estimates[:, col]), col]
        result.append({"delta_ce": float(mean) if np.isfinite(mean) else None,
                       "sequence_bootstrap_ci95": np.quantile(finite, [.025, .975]).tolist()
                       if len(finite) else None, "bootstrap_nonempty": len(finite)})
    return result


@torch.inference_mode()
def evaluate(model, batches, sp, layers, seeds, compare_sentence_end=False):
    if layers:
        from analysis.routing_control import routing_control
    _, table = history._piece_tables(sp)
    keys = ["all", "eos", "numeric", "punct_only", "sentence_end", "alpha"]
    baseline, counts, sources = [], [], []
    differences = {(layer, seed): [] for layer in layers for seed in seeds}
    grouped_differences = {(layer, seed): [] for layer in layers for seed in seeds}
    batch_sizes = []
    for number, (source, batch) in enumerate(batches):
        x, y, valid = to_tensors(batch, "cpu")
        ids = x.numpy()
        categories = [np.ones_like(ids, dtype=bool), ids == sp.eos_id(),
                      table["all_digit"][ids], table["no_alnum"][ids],
                      table["sent_end"][ids], table["all_alpha"][ids]]
        masks = [torch.from_numpy(mask) & valid for mask in categories]
        counts.append(np.column_stack([mask.sum(1).numpy() for mask in masks]))
        sources.extend([source] * len(x))
        batch_sizes.append(len(x))
        inputs = {}

        def capture(layer):
            def hook(block, args):
                inputs[layer] = args[0]
            return hook

        with ExitStack() as hooks:
            for layer in layers:
                hooks.callback(model.blocks[layer].register_forward_pre_hook(capture(layer)).remove)
            base_logits, _ = model(x)
        base_ce = loss_tokens(base_logits, y)
        del base_logits
        if not torch.isfinite(base_ce[valid]).all():
            raise ValueError("nonfinite baseline CE")

        def sequence_sums(values):
            return np.column_stack([(values * mask).sum(1, dtype=torch.float64).numpy() for mask in masks])

        baseline.append(sequence_sums(base_ce))
        for layer in layers:
            def suffix_ce(batch_seed, group_mask=None):
                hidden = inputs[layer]
                with routing_control(model, layer, valid, batch_seed, group_mask):
                    for block in model.blocks[layer:]:
                        hidden, _, _ = block(hidden)
                    return loss_tokens(model.head(model.ln_f(hidden)), y)

            if not torch.equal(suffix_ce(None)[valid], base_ce[valid]):
                raise ValueError(f"identity CE mismatch at layer {layer}, batch {number}")
            for seed in seeds:
                batch_seed = int(np.random.SeedSequence([seed, number]).generate_state(1)[0])
                def record(destination, group_mask=None):
                    ce = suffix_ce(batch_seed, group_mask)
                    if not torch.isfinite(ce[valid]).all():
                        raise ValueError(f"nonfinite shuffled CE at layer {layer}")
                    destination[layer, seed].append(sequence_sums(ce.double() - base_ce.double()))
                record(differences)
                if compare_sentence_end:
                    record(grouped_differences, torch.from_numpy(table["sent_end"][ids]))
        print(f"completed batch {number + 1}/{len(batches)} ({source})", flush=True)
    counts = np.concatenate(counts)
    baseline = np.concatenate(baseline)
    sources = np.asarray(sources)
    mean = summarize_rows(baseline, counts)
    result = {"baseline": {k: {"targets": int(counts[:, i].sum()),
                               "sequences_with_targets": int((counts[:, i] > 0).sum()),
                               "ce": float(mean[i]) if np.isfinite(mean[i]) else None}
                           for i, k in enumerate(keys)},
              "per_source_ce": {s: float(summarize_rows(baseline[sources == s],
                                                       counts[sources == s])[0]) for s in np.unique(sources)},
              "layers": {}}
    for layer in layers:
        values = [np.concatenate(differences[layer, seed]) for seed in seeds]
        result["layers"][str(layer)] = {
            "identity_ce_exact": True,
            "per_seed": {str(seed): dict(zip(keys, paired_stats(v, counts, sources)))
                         for seed, v in zip(seeds, values)},
            "mean_over_fixed_seeds": dict(zip(keys, paired_stats(np.mean(values, axis=0), counts, sources)))}
        if compare_sentence_end:
            grouped = [np.concatenate(grouped_differences[layer, seed]) for seed in seeds]
            paired = np.asarray(values) - np.asarray(grouped)
            # Routing rows move within batches. Resample these intact within source.
            starts = np.cumsum([0, *batch_sizes[:-1]])
            batch_counts = np.add.reduceat(counts, starts, axis=0)
            batch_deltas = np.add.reduceat(paired.mean(axis=0), starts, axis=0)
            intervals = paired_stats(batch_deltas, batch_counts, sources[starts])
            for interval in intervals:
                interval["batch_bootstrap_ci95"] = interval.pop("sequence_bootstrap_ci95")
            result["layers"][str(layer)]["sentence_end_comparison"] = {
                "grouped_minus_baseline": dict(zip(keys, paired_stats(
                    np.mean(grouped, axis=0), counts, sources))),
                "unrestricted_minus_grouped": dict(zip(keys, intervals)),
                "recovered_ce_per_seed": {
                    str(seed): dict(zip(keys, (float(x) if np.isfinite(x) else None
                                               for x in summarize_rows(rows, counts))))
                    for seed, rows in zip(seeds, paired)}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--compare-sentence-end", action="store_true",
                        help="compare unrestricted and sentence-end-preserving routing shuffles")
    parser.add_argument("--fresh-sample", action="store_true",
                        help="use the next ten validation batches from each source")
    args = parser.parse_args()
    if args.compare_sentence_end and not args.ablate:
        parser.error("--compare-sentence-end requires --ablate")
    if args.layers and not args.ablate:
        parser.error("--layers requires --ablate")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    model, batches, sp, hashes = load_inputs(args.data_root, args.checkpoint, args.fresh_sample)
    moe = [i for i, block in enumerate(model.blocks) if block.is_moe]
    layers = (args.layers or moe) if args.ablate else []
    if len(set(layers)) != len(layers) or set(layers) - set(moe):
        parser.error(f"choose distinct absolute MoE layer indices from {moe}")
    if args.ablate and model.cfg.moe_impl != "dense":
        parser.error("routing ablation requires the frozen dense-combined model")
    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(V3), *cmd], text=True).strip()
    command = ["python", "veylith/v3/analysis/routing_study.py", "--data-root", "$DATA",
               "--checkpoint", "$CKPT", "--out", "$OUT", "--threads", str(args.threads)]
    if layers:
        command.extend(["--ablate", "--layers", *map(str, layers), "--seeds", *map(str, args.seeds)])
    if args.compare_sentence_end:
        command.append("--compare-sentence-end")
    if args.fresh_sample:
        command.append("--fresh-sample")
    # The manifest uses relocatable artifact arguments. Keep shell expansion
    # for these three named paths, while quoting all fixed arguments.
    reproduction = " ".join('"' + arg + '"' if arg in {"$DATA", "$CKPT", "$OUT"} else shlex.quote(arg)
                            for arg in command)
    manifest = {"reproduction_command": reproduction, "schema_version": 1, "experiment": "v31-update500-routing-ce", "update": 500,
                "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "config": model.cfg.to_dict(), "sha256": hashes, "split": "validation",
                "sequences": sum(len(b["lengths"]) for _, b in batches),
                "runtime": {"device": "cpu", "dtype": "float32", "torch": str(torch.__version__),
                            "python": platform.python_version(), "threads": torch.get_num_threads(),
                            "interop_threads": torch.get_num_interop_threads(), "numpy": np.__version__,
                            "sentencepiece": spm.__version__},
                "shuffle_seeds": args.seeds if layers else [], "bootstrap_seed": 20261118,
                "bootstrap_draws": 1000, "mode": "ablation" if layers else "baseline"}
    if args.compare_sentence_end or args.fresh_sample:
        manifest["sample_batch_offset_per_source"] = history.BATCHES_PER_SOURCE if args.fresh_sample else 0
        manifest["analysis_files_sha256"] = {
            name: history.sha256_file(V3 / "analysis" / name)
            for name in ("routing_control.py", "routing_study.py")}
    started_eval = time.monotonic()
    manifest.update(evaluate(model, batches, sp, layers, args.seeds, args.compare_sentence_end))
    seconds = time.monotonic() - started_eval
    if manifest["baseline"]["all"]["targets"] != history.BATCH * history.BATCHES_PER_SOURCE * 3 * 511:
        raise ValueError("unexpected number of valid next-token targets")
    manifest.update(elapsed_seconds=time.monotonic() - start, evaluation_seconds=seconds,
                    peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Exclusive creation protects an earlier completed run from accidental replacement.
    payload = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(payload)


if __name__ == "__main__":
    main()
