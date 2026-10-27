"""v3.1-N: token/context routing profile of the frozen v3.1-L checkpoints.

CPU only, read-only. Loads step-0 and step-500 states, runs the exact frozen
forward pass on a bounded validation sample, and measures how top-1 expert
assignment depends on interpretable token features.

Usage:
    python v31n_expert_behavior.py --smoke --out /tmp/smoke.json
    python v31n_expert_behavior.py --out veylith/v3/docs/v31n-expert-behavior-summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

DATA = os.path.expanduser("~/veylith-v3-data/multi-100m-20261026")
RUN = os.path.expanduser("~/veylith-v31l-dense-20261027")
STATE0 = os.path.join(RUN, "ckpt_state_step0.pt")
STATE500 = os.path.join(RUN, "ckpt_state_step500.pt")
SOURCES = ["cosmopedia-v2", "fineweb-edu", "wikipedia"]
SEQ_LEN = 512
BATCH = 8
BATCHES_PER_SOURCE = 10          # 80 sequences per source, 240 total
N_PERMUTATIONS = 300
N_BOOTSTRAP = 300
SEED = 20261027
TOTAL_DEADLINE_S = 1150


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def js_divergence(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _piece_tables(sp):
    vocab = sp.get_piece_size()
    piece = [sp.id_to_piece(i) for i in range(vocab)]
    core = [p[1:] if p.startswith("\u2581") else p for p in piece]
    table = {
        "is_special": np.zeros(vocab, dtype=bool),
        "is_byte": np.zeros(vocab, dtype=bool),
        "all_alpha": np.zeros(vocab, dtype=bool),
        "all_digit": np.zeros(vocab, dtype=bool),
        "has_alpha": np.zeros(vocab, dtype=bool),
        "has_digit": np.zeros(vocab, dtype=bool),
        "no_alnum": np.zeros(vocab, dtype=bool),
        "upper": np.zeros(vocab, dtype=bool),
        "sent_end": np.zeros(vocab, dtype=bool),
        "word_start": np.zeros(vocab, dtype=bool),
        "whitespace": np.zeros(vocab, dtype=bool),
        "empty": np.zeros(vocab, dtype=bool),
    }
    for i, p in enumerate(piece):
        c = core[i]
        special = p.startswith("<") and p.endswith(">")
        table["is_special"][i] = special
        table["is_byte"][i] = special and p.startswith("<0x")
        table["empty"][i] = len(c) == 0
        table["all_alpha"][i] = bool(c) and all(ch.isalpha() for ch in c) and not special
        table["all_digit"][i] = bool(c) and all(ch.isdigit() for ch in c) and not special
        table["has_alpha"][i] = any(ch.isalpha() for ch in c)
        table["has_digit"][i] = any(ch.isdigit() for ch in c)
        table["no_alnum"][i] = bool(c) and not any(ch.isalnum() for ch in c) and not special
        table["upper"][i] = any(ch.isupper() for ch in c)
        table["sent_end"][i] = any(ch in ".!?" for ch in c) and not special
        table["word_start"][i] = p.startswith("\u2581")
        table["whitespace"][i] = bool(c) and c.strip() == ""
    return piece, table


def build_sample(sp, sources, batches_per_source):
    from data_pipeline.loader import plan_sequences  # noqa: F401
    from training.data import iter_source_batches

    tokens = []
    lengths = []
    source_ids = []
    order = []
    for source_index, source in enumerate(sources):
        batch_index = 0
        for batch in iter_source_batches(DATA, "validation", source, SEQ_LEN,
                                         BATCH, max_batches=batches_per_source):
            count = batch["tokens"].shape[0]
            tokens.append(batch["tokens"])
            lengths.extend(int(v) for v in batch["lengths"])
            source_ids.extend([source_index] * count)
            for row in range(count):
                order.append({"source": source, "batch": batch_index,
                              "row": row,
                              "length": int(batch["lengths"][row]),
                              "first_ids": [int(v) for v in batch["tokens"][row][:8]]})
            batch_index += 1
    tokens = np.concatenate(tokens, axis=0)
    lengths = np.asarray(lengths, dtype=np.int32)
    source_ids = np.asarray(source_ids, dtype=np.int8)
    digest = hashlib.sha256(tokens.tobytes()).hexdigest()
    return tokens, lengths, source_ids, order, digest


def build_features(tokens, lengths, source_ids, sp, table, counts_by_id):
    eos = sp.eos_id()
    pad = sp.pad_id()
    x = tokens[:, :-1]
    n_seq, positions = x.shape
    pos = np.broadcast_to(np.arange(positions)[None, :], (n_seq, positions))
    prev = np.full_like(x, pad)
    prev[:, 1:] = x[:, :-1]

    def lookup(key):
        return table[key][x]

    feat = {}
    feat["alpha"] = lookup("all_alpha")
    feat["numeric"] = lookup("all_digit")
    feat["punct_only"] = lookup("no_alnum")
    feat["word_start"] = lookup("word_start")
    feat["continuation"] = lookup("all_alpha") & ~lookup("word_start")
    feat["capitalized"] = lookup("upper")
    feat["sentence_end"] = lookup("sent_end")
    feat["byte_fallback"] = lookup("is_byte")
    feat["eos"] = x == eos
    feat["after_boundary"] = (pos == 0) | (prev == eos) | (prev == pad)
    feat["after_punct"] = table["no_alnum"][prev]
    feat["pos_early"] = pos < 128
    feat["pos_mid"] = (pos >= 128) & (pos < 384)
    feat["pos_late"] = pos >= 384

    total = counts_by_id.sum()
    order = np.argsort(-counts_by_id)
    cumulative = np.cumsum(counts_by_id[order])
    first = int(np.searchsorted(cumulative, total / 3.0))
    second = int(np.searchsorted(cumulative, 2 * total / 3.0))
    common = np.zeros(sp.get_piece_size(), dtype=bool)
    rare = np.zeros(sp.get_piece_size(), dtype=bool)
    common[order[:first + 1]] = True
    rare[order[second + 1:]] = True
    feat["freq_common"] = common[x]
    feat["freq_rare"] = rare[x]

    for name in feat:
        assert feat[name].shape == (n_seq, positions), name
    return feat, x


def validation_id_counts(sp):
    from data_pipeline import dataset as ds

    manifest = json.load(open(os.path.join(DATA, "corpus", "manifest.json"),
                              encoding="utf-8"))
    counts = np.zeros(sp.get_piece_size(), dtype=np.int64)
    for shard in manifest["tokenization"]["splits"]["validation"]["shards"]:
        array = np.load(ds.resolve(ds.corpus_dir(DATA), shard["path"]))
        counts += np.bincount(array.ravel(), minlength=sp.get_piece_size())
    return counts, int(counts.sum())


class Probe:
    def __init__(self, model):
        self.logits = {}
        self.handles = []
        index = 0
        for block in model.blocks:
            if not block.is_moe:
                continue
            self.handles.append(block.mlp.gate.register_forward_hook(
                self._hook(index)))
            index += 1

    def _hook(self, layer):
        def hook(module, inputs, output):
            self.logits[layer] = output.detach().float()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_model(torch, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    sys.path.insert(0, "veylith/v3")
    from architecture.config import ModelConfig
    from architecture.model import GPT
    model = GPT(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def runtime_inference(torch, model, batches, deadline):
    started = time.monotonic()
    layer_count = sum(1 for block in model.blocks if block.is_moe)
    rows = sum(batch.shape[0] for batch in batches)
    positions = batches[0].shape[1] - 1
    top1 = np.zeros((layer_count, rows, positions), dtype=np.int8)
    p1 = np.zeros((layer_count, rows, positions), dtype=np.float32)
    p2 = np.zeros((layer_count, rows, positions), dtype=np.float32)
    entropy = np.zeros((layer_count, rows, positions), dtype=np.float32)
    maxprob = np.zeros((layer_count, rows, positions), dtype=np.float32)
    done = 0
    truncated = False
    probe = Probe(model)
    try:
        with torch.no_grad():
            for batch in batches:
                if time.monotonic() - started > deadline:
                    truncated = True
                    break
                index = torch.from_numpy(batch[:, :-1].astype(np.int64))
                model(index)
                size = batch.shape[0]
                for layer, logits in probe.logits.items():
                    probs = torch.softmax(logits, dim=-1)
                    values, indices = torch.topk(probs, 2, dim=-1)
                    top1[layer, done:done + size] = indices[:, 0].numpy().reshape(size, positions)
                    p1[layer, done:done + size] = values[:, 0].numpy().reshape(size, positions)
                    p2[layer, done:done + size] = values[:, 1].numpy().reshape(size, positions)
                    maxprob[layer, done:done + size] = values[:, 0].numpy().reshape(size, positions)
                    norm = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1) / math.log(8)
                    entropy[layer, done:done + size] = norm.numpy().reshape(size, positions)
                done += size
    finally:
        probe.close()
    return {
        "top1": top1[:, :done],
        "p1": p1[:, :done],
        "p2": p2[:, :done],
        "entropy": entropy[:, :done],
        "maxprob": maxprob[:, :done],
        "seconds": time.monotonic() - started,
        "sequences": done,
        "tokens": done * positions,
        "truncated": truncated,
        "layers": layer_count,
    }


def analyze_associations(top1, features, source_ids, n_perm, n_boot, rng):
    n_layer, n_seq, positions = top1.shape
    flat = top1.reshape(n_layer, n_seq * positions).astype(np.int64)
    names = list(features)
    feat_flat = {name: features[name].reshape(-1) for name in names}
    n_tokens = flat.shape[1]

    global_counts = np.stack([np.bincount(flat[layer], minlength=8)
                              for layer in range(n_layer)])
    global_p = global_counts / n_tokens
    seq_counts = np.zeros((n_layer, n_seq, 8), dtype=np.int64)
    for layer in range(n_layer):
        for seq in range(n_seq):
            seq_counts[layer, seq] = np.bincount(top1[layer, seq], minlength=8)

    feat_seq_counts = np.zeros((n_layer, n_seq, len(names), 8), dtype=np.int64)
    observed_counts = np.zeros((n_layer, len(names), 8), dtype=np.int64)
    for layer in range(n_layer):
        for name_index, name in enumerate(names):
            mask = feat_flat[name]
            observed_counts[layer, name_index] = np.bincount(flat[layer][mask],
                                                             minlength=8)
            for seq in range(n_seq):
                row_mask = features[name][seq]
                if row_mask.any():
                    feat_seq_counts[layer, seq, name_index] = np.bincount(
                        top1[layer, seq][row_mask], minlength=8)

    results = {}
    for name_index, name in enumerate(names):
        js_bits = []
        maxdev = []
        lift = []
        ci_lo = []
        ci_hi = []
        within_source = []
        within_position = []
        for layer in range(n_layer):
            counts = observed_counts[layer, name_index]
            n = int(counts.sum())
            if n == 0:
                js_bits.append(None)
                maxdev.append(None)
                lift.append(None)
                ci_lo.append(None)
                ci_hi.append(None)
                within_source.append(None)
                within_position.append(None)
                continue
            conditional = counts / n
            js_bits.append(js_divergence(conditional, global_p[layer]))
            deviation = conditional - global_p[layer]
            maxdev.append(float(np.abs(deviation).max() * 100))
            lift.append(float((conditional / np.maximum(global_p[layer], 1e-12)).max()))
            # sequence bootstrap over sequences that contain this feature
            eligible = np.array([seq for seq in range(n_seq)
                                 if features[name][seq].any()])
            weights = np.zeros((n_boot, n_seq))
            for iteration in range(n_boot):
                draw = rng.choice(eligible, size=eligible.size, replace=True)
                weights[iteration] = np.bincount(draw, minlength=n_seq)
            matrix = feat_seq_counts[layer][:, name_index, :].astype(float)
            sums = weights @ matrix
            totals = sums.sum(axis=1)
            samples = np.array([js_divergence(sums[i] / totals[i], global_p[layer])
                                for i in range(n_boot) if totals[i] > 0])
            if samples.size:
                ci_lo.append(float(np.percentile(samples, 2.5)))
                ci_hi.append(float(np.percentile(samples, 97.5)))
            else:
                ci_lo.append(None)
                ci_hi.append(None)
            # within-source stratified JS
            per_source = []
            weights_source = []
            for source_index in range(len(SOURCES)):
                sel = source_ids == source_index
                if not sel.any():
                    continue
                base_counts = seq_counts[layer][sel].sum(axis=0)
                cond_counts = feat_seq_counts[layer][sel, name_index, :].sum(axis=0)
                if cond_counts.sum() == 0:
                    continue
                per_source.append(js_divergence(cond_counts / cond_counts.sum(),
                                                base_counts / base_counts.sum()))
                weights_source.append(cond_counts.sum())
            within_source.append(
                float(np.average(per_source, weights=weights_source))
                if per_source else None)
            # within-position-bucket stratified JS
            bucket_stats = []
            bucket_weights = []
            for low, high in ((0, 128), (128, 384), (384, positions)):
                base_counts = np.bincount(
                    top1[layer, :, low:high].reshape(-1).astype(np.int64),
                    minlength=8)
                mask = features[name][:, low:high]
                cond = top1[layer, :, low:high][mask]
                if cond.size == 0 or base_counts.sum() == 0:
                    continue
                cond_counts = np.bincount(cond.astype(np.int64), minlength=8)
                bucket_stats.append(js_divergence(cond_counts / cond_counts.sum(),
                                                  base_counts / base_counts.sum()))
                bucket_weights.append(cond_counts.sum())
            within_position.append(
                float(np.average(bucket_stats, weights=bucket_weights))
                if bucket_stats else None)
        results[name] = {
            "n_tokens": int(observed_counts[0, name_index, :].sum()),
            "js_bits": js_bits,
            "js_ci95": [[lo, hi] for lo, hi in zip(ci_lo, ci_hi)],
            "maxdev_pp": maxdev,
            "lift": lift,
            "js_within_source": within_source,
            "js_within_position": within_position,
        }

    # within-sequence expert permutation null (equivalent to permuting labels)
    null = np.zeros((n_perm, n_layer, len(names)))
    for iteration in range(n_perm):
        index = np.argsort(rng.random((n_seq, positions)), axis=1)
        index_full = np.broadcast_to(index[None, :, :], top1.shape)
        permuted = np.take_along_axis(top1, index_full, axis=2)
        flat_perm = permuted.reshape(n_layer, -1).astype(np.int64)
        for layer in range(n_layer):
            for name_index, name in enumerate(names):
                counts = np.bincount(flat_perm[layer][feat_flat[name]], minlength=8)
                total = counts.sum()
                if total:
                    null[iteration, layer, name_index] = js_divergence(
                        counts / total, global_p[layer])
    null95 = np.percentile(null, 95, axis=0)

    # p-values from the permutation null
    p_values = np.ones((n_layer, len(names)))
    for layer in range(n_layer):
        for name_index in range(len(names)):
            observed = results[names[name_index]]["js_bits"][layer]
            if observed is None:
                continue
            exceed = int(np.sum(null[:, layer, name_index] >= observed))
            p_values[layer, name_index] = (1 + exceed) / (1 + n_perm)
    q_values = benjamini_hochberg(p_values.reshape(-1)).reshape(p_values.shape)
    for name_index, name in enumerate(names):
        results[name]["p_value"] = [float(p_values[layer, name_index])
                                    for layer in range(n_layer)]
        results[name]["q_value"] = [float(q_values[layer, name_index])
                                    for layer in range(n_layer)]
        results[name]["null95_bits"] = [float(null95[layer, name_index])
                                        for layer in range(n_layer)]

    # source test at sequence level
    source_result = {"per_source_js": {source: [] for source in SOURCES},
                     "p_value": [], "q_value": [], "null95_bits": []}
    source_observed = np.zeros((n_layer, len(SOURCES)))
    for layer in range(n_layer):
        for source_index in range(len(SOURCES)):
            sel = source_ids == source_index
            counts = seq_counts[layer][sel].sum(axis=0)
            source_observed[layer, source_index] = js_divergence(
                counts / counts.sum(), global_p[layer])
    source_null = np.zeros((n_perm, n_layer, len(SOURCES)))
    for iteration in range(n_perm):
        labels = rng.permutation(source_ids)
        for layer in range(n_layer):
            total = seq_counts[layer].sum(axis=0)
            for source_index in range(len(SOURCES)):
                sel = labels == source_index
                counts = seq_counts[layer][sel].sum(axis=0)
                source_null[iteration, layer, source_index] = js_divergence(
                    counts / counts.sum(), global_p[layer])
    source_p = np.ones((n_layer, len(SOURCES)))
    for layer in range(n_layer):
        for source_index in range(len(SOURCES)):
            exceed = int(np.sum(source_null[:, layer, source_index]
                                >= source_observed[layer, source_index]))
            source_p[layer, source_index] = (1 + exceed) / (1 + n_perm)
    source_q = benjamini_hochberg(source_p.reshape(-1)).reshape(source_p.shape)
    for layer in range(n_layer):
        for source_index, source in enumerate(SOURCES):
            source_result["per_source_js"][source].append(
                float(source_observed[layer, source_index]))
            source_result["p_value"].append(float(source_p[layer, source_index]))
            source_result["q_value"].append(float(source_q[layer, source_index]))
            source_result["null95_bits"].append(
                float(np.percentile(source_null[:, layer, source_index], 95)))

    return {
        "global": {
            "shares": [global_p[layer].tolist() for layer in range(n_layer)],
            "entropy": [float(-(global_p[layer] * np.log2(
                np.clip(global_p[layer], 1e-12, None))).sum() / math.log2(8))
                for layer in range(n_layer)],
        },
        "features": results,
        "source": source_result,
        "null_median_bits": float(np.median(null)),
        "null95_median_bits": float(np.median(null95)),
        "permutation": n_perm,
        "bootstrap": n_boot,
        "tokens": n_tokens,
        "sequences": n_seq,
    }


def expert_correspondence(torch, states):
    sys.path.insert(0, "veylith/v3")
    matrices = []
    for layer in range(7):
        block = layer + 1
        stacks = []
        for state in states:
            model = state["model"]
            stack = []
            for expert in range(8):
                parts = [model[f"blocks.{block}.mlp.experts.{expert}.{name}.weight"]
                         for name in ("w1", "w2", "w3")]
                stack.append(torch.cat([part.reshape(-1) for part in parts]).float())
            stacks.append(torch.stack(stack))
        unit = [stack / stack.norm(dim=1, keepdim=True) for stack in stacks]
        matrix = (unit[0] @ unit[1].T).numpy()
        diagonal = np.diag(matrix)
        off = matrix.copy()
        np.fill_diagonal(off, -1.0)
        matrices.append({
            "layer": layer + 1,
            "diagonal": diagonal.tolist(),
            "diagonal_min": float(diagonal.min()),
            "offdiagonal_max": float(off.max()),
            "argmax_is_diagonal": bool(np.all(matrix.argmax(axis=1) == np.arange(8))),
            "matrix": [[round(float(v), 4) for v in row] for row in matrix],
        })
    return matrices


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequences-per-source", type=int, default=None)
    args = parser.parse_args()
    if args.smoke:
        args.sequences_per_source = 8
    batches_per_source = (args.sequences_per_source // BATCH
                          if args.sequences_per_source else BATCHES_PER_SOURCE)
    n_perm = 40 if args.smoke else N_PERMUTATIONS
    n_boot = 40 if args.smoke else N_BOOTSTRAP

    sys.path.insert(0, "veylith/v3")
    import sentencepiece as spm
    import torch
    from data_pipeline import dataset as ds

    torch.set_num_threads(torch.get_num_threads())
    rng = np.random.default_rng(SEED)

    manifest = json.load(open(os.path.join(DATA, "corpus", "manifest.json"),
                              encoding="utf-8"))
    tokenization = manifest["tokenization"]
    tokenizer_path = ds.resolve(ds.corpus_dir(DATA), tokenization["tokenizer"]["model"])
    tokenizer_sha = sha256_file(tokenizer_path)
    assert tokenizer_sha == tokenization["tokenizer"]["sha256"], "tokenizer hash mismatch"
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    piece, table = _piece_tables(sp)

    tokens, lengths, source_ids, order, sample_digest = build_sample(
        sp, SOURCES, batches_per_source)
    counts_by_id, validation_tokens = validation_id_counts(sp)
    features, x = build_features(tokens, lengths, source_ids, sp, table, counts_by_id)

    recorded = {row["step"]: row for row in
                (json.loads(line) for line in
                 open(os.path.join(RUN, "checkpoints.jsonl"), encoding="utf-8"))}
    integrity = {}
    for step, path in ((0, STATE0), (500, STATE500)):
        digest = sha256_file(path)
        integrity[step] = {
            "path": path,
            "sha256": digest,
            "recorded_sha256": recorded[step]["sha256"],
            "match": digest == recorded[step]["sha256"],
            "bytes": os.path.getsize(path),
            "recorded_bytes": recorded[step]["bytes"],
            "bytes_match": os.path.getsize(path) == recorded[step]["bytes"],
        }
        assert integrity[step]["match"], f"checkpoint hash mismatch at step {step}"

    model0, state0 = load_model(torch, STATE0)
    model500, state500 = load_model(torch, STATE500)
    correspondence = expert_correspondence(torch, (state0, state500))
    del state0, state500

    batches = [tokens[index:index + BATCH] for index in range(0, tokens.shape[0], BATCH)]

    # instrumentation equivalence on the first batch
    check_batch = batches[0]
    index = torch.from_numpy(check_batch[:, :-1].astype(np.int64))
    with torch.no_grad():
        plain, _ = model500(index)
        probe = Probe(model500)
        try:
            instrumented, _, _, router = model500(index, return_metrics=True,
                                                  return_router=True)
        finally:
            probe.close()
    instrumentation = {
        "max_abs_logit_difference": float((plain - instrumented).abs().max()),
        "tolerance": 1e-5,
        "within_tolerance": bool(float((plain - instrumented).abs().max()) <= 1e-5),
        "probe_matches_forward_router": bool(all(
            torch.equal(probe_top.reshape(-1), forward_top.reshape(-1))
            for probe_top, forward_top in zip(
                [probe.logits[layer].argmax(dim=-1) for layer in range(7)],
                [router[layer][:, 0] for layer in range(7)]))),
    }
    del plain, instrumented, router

    results = {}
    inference_budget = {"elapsed": 0.0, "limit": 480.0}

    def run_model(label, model):
        remaining = max(30.0, inference_budget["limit"] - inference_budget["elapsed"])
        inference = runtime_inference(torch, model, batches, remaining)
        inference_budget["elapsed"] += inference["seconds"]
        used = inference["sequences"]
        analysis = analyze_associations(
            inference["top1"],
            {name: values[:used] for name, values in features.items()},
            source_ids[:used], n_perm, n_boot, rng)
        analysis["router"] = {
            "layer": [layer + 1 for layer in range(inference["layers"])],
            "softmax_entropy": [float(inference["entropy"][layer].mean())
                                for layer in range(inference["layers"])],
            "max_probability": [float(inference["maxprob"][layer].mean())
                                for layer in range(inference["layers"])],
            "top2_margin": [float((inference["p1"][layer]
                                   - inference["p2"][layer]).mean())
                            for layer in range(inference["layers"])],
        }
        analysis["inference"] = {key: inference[key] for key in
                                 ("seconds", "sequences", "tokens", "truncated", "layers")}
        analysis["samples"] = {
            "top1": inference["top1"].astype(np.int8),
            "p1": inference["p1"],
            "softmax_entropy": inference["entropy"],
        }
        return analysis

    results["step0"] = run_model("step0", model0)
    del model0
    results["step500"] = run_model("step500", model500)
    del model500
    for step in (100, 250):
        if inference_budget["elapsed"] >= inference_budget["limit"]:
            break
        if time.monotonic() - started > TOTAL_DEADLINE_S - 240:
            break
        path = recorded[step]["path"]
        digest = sha256_file(path)
        integrity[step] = {
            "path": path,
            "sha256": digest,
            "recorded_sha256": recorded[step]["sha256"],
            "match": digest == recorded[step]["sha256"],
            "bytes": os.path.getsize(path),
            "recorded_bytes": recorded[step]["bytes"],
            "bytes_match": os.path.getsize(path) == recorded[step]["bytes"],
        }
        assert integrity[step]["match"], f"checkpoint hash mismatch at step {step}"
        model, state = load_model(torch, path)
        results[f"step{step}"] = run_model(f"step{step}", model)
        del model, state

    # step comparison on the same tokens
    used = min(results["step0"]["inference"]["sequences"],
               results["step500"]["inference"]["sequences"])
    comparison = {"agreement": [], "chance_agreement": [], "layer": []}
    for layer in range(results["step0"]["inference"]["layers"]):
        first = results["step0"]["samples"]["top1"][layer][:used].astype(np.int64)
        second = results["step500"]["samples"]["top1"][layer][:used].astype(np.int64)
        p0 = np.bincount(first.reshape(-1), minlength=8) / first.size
        p5 = np.bincount(second.reshape(-1), minlength=8) / second.size
        comparison["layer"].append(layer + 1)
        comparison["agreement"].append(float((first == second).mean()))
        comparison["chance_agreement"].append(float((p0 * p5).sum()))
    comparison["feature_js"] = {
        name: {
            "step0": results["step0"]["features"][name]["js_bits"],
            "step500": results["step500"]["features"][name]["js_bits"],
            "delta": [None if (a is None or b is None) else b - a
                      for a, b in zip(results["step0"]["features"][name]["js_bits"],
                                      results["step500"]["features"][name]["js_bits"])],
        }
        for name in results["step0"]["features"]
    }

    # literal feature-label permutation control (step 500, small count)
    control_perm = 40 if not args.smoke else 10
    control = {"feature_label_null_median_bits": None,
               "expert_permutation_null_median_bits": results["step500"]["null_median_bits"]}
    top1_500 = results["step500"]["samples"]["top1"]
    n_layer, n_seq, positions = top1_500.shape
    global_p = np.array(results["step500"]["global"]["shares"])
    samples = []
    for iteration in range(control_perm):
        index_perm = np.argsort(rng.random((n_seq, positions)), axis=1)
        for layer in range(n_layer):
            for name in features:
                permuted = np.take_along_axis(features[name][:n_seq], index_perm, axis=1)
                counts = np.bincount(top1_500[layer][permuted].astype(np.int64),
                                     minlength=8)
                samples.append(js_divergence(counts / counts.sum(), global_p[layer]))
    control["feature_label_null_median_bits"] = float(np.median(samples))

    # global frequency-matched random expert control
    global_shuffle = []
    per_feature_shuffle = {name: [] for name in features}
    for iteration in range(20):
        shuffled = rng.permutation(top1_500.reshape(-1)).reshape(top1_500.shape)
        values = []
        for layer in range(n_layer):
            for name in features:
                counts = np.bincount(shuffled[layer][features[name]].astype(np.int64),
                                     minlength=8)
                value = js_divergence(counts / counts.sum(), global_p[layer])
                values.append(value)
                per_feature_shuffle[name].append(value)
        global_shuffle.append(float(np.median(values)))
    control["global_shuffle_median_bits"] = float(np.median(global_shuffle))
    control["global_shuffle_max_bits"] = float(np.max(global_shuffle))
    control["global_shuffle_median_bits_by_feature"] = {
        name: float(np.median(values)) for name, values in per_feature_shuffle.items()}

    trajectory = {"steps": [], "softmax_entropy": {}, "max_probability": {},
                  "global_entropy": {}, "median_feature_js": {},
                  "significant_tests": {}, "inference_seconds": {}}
    for label in sorted(results, key=lambda name: int(name[4:])):
        step = int(label[4:])
        model_result = results[label]
        values = [value for feature in model_result["features"].values()
                  for value in feature["js_bits"] if value is not None]
        significant = 0
        for feature in model_result["features"].values():
            for layer in range(len(feature["js_bits"])):
                if (feature["js_bits"][layer] is not None
                        and feature["q_value"][layer] < 0.05
                        and feature["js_bits"][layer] > feature["null95_bits"][layer]):
                    significant += 1
        trajectory["steps"].append(step)
        trajectory["softmax_entropy"][step] = model_result["router"]["softmax_entropy"]
        trajectory["max_probability"][step] = model_result["router"]["max_probability"]
        trajectory["global_entropy"][step] = model_result["global"]["entropy"]
        trajectory["median_feature_js"][step] = float(np.median(values))
        trajectory["significant_tests"][step] = significant
        trajectory["inference_seconds"][step] = model_result["inference"]["seconds"]

    # compare the first 40 sequences per source with the v3.1-M sample
    baseline_path = "veylith/v3/docs/v31m-analysis-summary.json"
    baseline_reproduction = {"source": baseline_path}
    if os.path.exists(baseline_path):
        v31m = json.load(open(baseline_path, encoding="utf-8"))
        last = v31m["learning"]["validation"][-1]
        baseline_reproduction.update({
            "v31m_final_mean_entropy": last["mean_entropy"],
            "v31m_share_min": min(layer["min_share"] for layer in last["layers"]),
            "v31m_share_max": max(layer["max_share"] for layer in last["layers"]),
            "v31m_layers": len(last["layers"]),
            "v31m_experts": len(last["layers"][0]["share_pct"]),
        })
        subset_check = {}
        for label, key in (("step0", "init"), ("step500", "step500")):
            entropy = results[label]["samples"]["softmax_entropy"]
            maxprob = results[label]["samples"]["p1"]
            n_seq = entropy.shape[1]
            if n_seq >= 200 and tokens.shape[0] == 240:
                index = np.r_[0:40, 80:120, 160:200]
            else:
                index = np.arange(n_seq)
            measured_entropy = entropy[:, index].mean(axis=(1, 2))
            measured_maxprob = maxprob[:, index].mean(axis=(1, 2))
            stored_entropy = [v31m["specialization"]["softmax_and_outputs"][key][str(layer)]["softmax_entropy_norm_mean"]
                              for layer in range(1, 8)]
            stored_maxprob = [v31m["specialization"]["softmax_and_outputs"][key][str(layer)]["softmax_max_prob_mean"]
                              for layer in range(1, 8)]
            subset_check[label] = {
                "entropy": [float(v) for v in measured_entropy],
                "stored_entropy": stored_entropy,
                "max_abs_entropy_difference": float(np.max(np.abs(
                    measured_entropy - np.array(stored_entropy)))),
                "maxprob": [float(v) for v in measured_maxprob],
                "stored_maxprob": stored_maxprob,
                "max_abs_maxprob_difference": float(np.max(np.abs(
                    measured_maxprob - np.array(stored_maxprob)))),
            }
        baseline_reproduction["v31m_subset_check"] = subset_check

    for label in results:
        results[label].pop("samples")

    summary = {
        "meta": {
            "experiment": "v3.1-N expert behavior and routing profile",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "CPU-only, read-only",
            "seed": SEED,
            "permutations": n_perm,
            "bootstrap": n_boot,
            "elapsed_seconds": time.monotonic() - started,
            "smoke": bool(args.smoke),
        },
        "checkpoint_integrity": integrity,
        "tokenizer": {
            "path": tokenizer_path,
            "sha256": tokenizer_sha,
            "vocab": sp.get_piece_size(),
            "special_ids": {"bos": sp.bos_id(), "eos": sp.eos_id(),
                            "pad": sp.pad_id(), "unk": sp.unk_id()},
        },
        "sample": {
            "split": "validation",
            "sources": SOURCES,
            "sequences_per_source": tokens.shape[0] // len(SOURCES),
            "sequences": int(tokens.shape[0]),
            "tokens": int(tokens.shape[0] * (SEQ_LEN - 1)),
            "seq_len": SEQ_LEN,
            "deterministic": "first batches per source in loader order, no outcome-based selection",
            "digest_sha256": sample_digest,
            "all_lengths_512": bool(np.all(lengths == SEQ_LEN)),
            "validation_tokens_scanned_for_frequency": validation_tokens,
            "order_first_rows": order[:6],
        },
        "features": {
            "counts": {name: int(features[name].sum()) for name in features},
            "counts_by_source": {
                name: {source: int(features[name][source_ids == index].sum())
                       for index, source in enumerate(SOURCES)}
                for name in features},
            "frequency_basis": (
                "validation-split token counts, not training-corpus counts; "
                "tertiles selected by occurrence mass"),
        },
        "instrumentation": instrumentation,
        "expert_correspondence": correspondence,
        "baseline_reproduction": baseline_reproduction,
        "models": results,
        "trajectory": trajectory,
        "inference_total_seconds": inference_budget["elapsed"],
        "comparison": comparison,
        "controls": control,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("wrote", args.out, os.path.getsize(args.out), "bytes",
          "elapsed", round(time.monotonic() - started, 1), "s")


if __name__ == "__main__":
    main()
