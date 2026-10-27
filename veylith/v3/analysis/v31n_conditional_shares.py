"""v3.1-N supplement: conditional expert-share tables.

Runs the same deterministic sample and the same probe as v31n_expert_behavior.py
for step 0 and step 500 only, and records the full conditional expert-share
vector per feature and layer. The main analysis already computes the tests; this
pass exists so the distribution itself is inspectable.

CPU only, read-only.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "veylith/v3")

from v31n_expert_behavior import (  # noqa: E402
    BATCH, BATCHES_PER_SOURCE, DATA, Probe, RUN, SEQ_LEN, SOURCES,
    STATE0, STATE500, build_features, build_sample, load_model, sha256_file,
    validation_id_counts, _piece_tables,
)

OUT = "veylith/v3/docs/v31n-conditional-shares.json"


def counts_for(top1, features, source_ids):
    n_layer, n_seq, _ = top1.shape
    flat = top1.reshape(n_layer, -1).astype(np.int64)
    global_counts = np.stack([np.bincount(flat[layer], minlength=8)
                              for layer in range(n_layer)])
    result = {"global_shares": (global_counts / flat.shape[1]).round(5).tolist(),
              "features": {}}
    for name, values in features.items():
        per_layer = []
        for layer in range(n_layer):
            counts = np.bincount(flat[layer][values.reshape(-1)], minlength=8)
            n = int(counts.sum())
            shares = counts / n if n else np.full(8, np.nan)
            order = np.argsort(-shares) if n else np.array([], dtype=int)
            per_layer.append({
                "n": n,
                "shares": [round(float(v), 5) for v in shares],
                "top_expert": int(order[0]) if n else None,
                "top_share": round(float(shares[order[0]]), 5) if n else None,
            })
        result["features"][name] = per_layer
    result["source_shares"] = {}
    for index, source in enumerate(SOURCES):
        sel = source_ids == index
        per_layer = []
        for layer in range(n_layer):
            counts = np.bincount(top1[layer][sel].reshape(-1).astype(np.int64),
                                 minlength=8)
            per_layer.append([round(float(v), 5) for v in counts / counts.sum()])
        result["source_shares"][source] = per_layer
    return result


def main():
    started = time.monotonic()
    import sentencepiece as spm
    import torch
    from data_pipeline import dataset as ds

    torch.set_num_threads(torch.get_num_threads())
    manifest = json.load(open(os.path.join(DATA, "corpus", "manifest.json"),
                              encoding="utf-8"))
    tokenization = manifest["tokenization"]
    tokenizer_path = ds.resolve(ds.corpus_dir(DATA),
                                tokenization["tokenizer"]["model"])
    tokenizer_sha = sha256_file(tokenizer_path)
    assert tokenizer_sha == tokenization["tokenizer"]["sha256"]
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    _piece, table = _piece_tables(sp)

    tokens, lengths, source_ids, _order, digest = build_sample(
        sp, SOURCES, BATCHES_PER_SOURCE)
    counts_by_id, _validation_tokens = validation_id_counts(sp)
    features, _x = build_features(tokens, lengths, source_ids, sp, table,
                                  counts_by_id)
    recorded = {row["step"]: row for row in
                (json.loads(line) for line in
                 open(os.path.join(RUN, "checkpoints.jsonl"), encoding="utf-8"))}
    batches = [tokens[index:index + BATCH]
               for index in range(0, tokens.shape[0], BATCH)]

    result = {"sample_digest_sha256": digest,
              "sequences": int(tokens.shape[0]),
              "tokens": int(tokens.shape[0] * (SEQ_LEN - 1)),
              "features": list(features),
              "models": {}}
    checkpoints = (("step0", STATE0, 0), ("step250", recorded[250]["path"], 250),
                   ("step500", STATE500, 500))
    total_inference = 0.0
    for label, path, step in checkpoints:
        digest_state = sha256_file(path)
        assert digest_state == recorded[step]["sha256"], "checkpoint hash mismatch"
        model, state = load_model(torch, path)
        del state
        top1 = np.zeros((7, tokens.shape[0], SEQ_LEN - 1), dtype=np.int8)
        done = 0
        probe = Probe(model)
        started_inference = time.monotonic()
        try:
            with torch.no_grad():
                for batch in batches:
                    index = torch.from_numpy(batch[:, :-1].astype(np.int64))
                    model(index)
                    size = batch.shape[0]
                    for layer, logits in probe.logits.items():
                        top1[layer, done:done + size] = torch.topk(
                            logits, 2, dim=-1).indices[:, 0].numpy().reshape(
                                size, SEQ_LEN - 1)
                    done += size
        finally:
            probe.close()
        elapsed = time.monotonic() - started_inference
        total_inference += elapsed
        result["models"][label] = {
            "checkpoint_sha256": digest_state,
            "inference_seconds": elapsed,
            "sequences": done,
            **counts_for(top1[:, :done], {name: values[:done]
                                         for name, values in features.items()},
                         source_ids[:done]),
        }
        del model, top1
    result["inference_total_seconds"] = total_inference
    result["elapsed_seconds"] = time.monotonic() - started

    if "step250" in result["models"] and "step500" in result["models"]:
        stability = {"top_expert_agreement": {}, "mean_total_variation": {},
                     "per_feature_tv": {}}
        for first, second in (("step0", "step500"), ("step250", "step500")):
            same = 0
            cells = 0
            deviations = []
            per_feature = {}
            for name in result["features"]:
                feature_tv = []
                for layer in range(7):
                    a = result["models"][first]["features"][name][layer]
                    b = result["models"][second]["features"][name][layer]
                    same += int(a["top_expert"] == b["top_expert"])
                    cells += 1
                    tv = 0.5 * sum(abs(x - y) for x, y in zip(a["shares"], b["shares"]))
                    deviations.append(tv)
                    feature_tv.append(round(float(tv), 4))
                per_feature[name] = feature_tv
            stability["top_expert_agreement"][f"{first}_vs_{second}"] = same / cells
            stability["mean_total_variation"][f"{first}_vs_{second}"] = float(np.mean(deviations))
            stability["per_feature_tv"][f"{first}_vs_{second}"] = per_feature
        result["stability"] = stability

    # sanity check against the main analysis: recompute JS for one feature
    main_path = "veylith/v3/docs/v31n-expert-behavior-summary.json"
    checks = {}
    if os.path.exists(main_path):
        main_summary = json.load(open(main_path, encoding="utf-8"))

        def js(p, q):
            p = np.asarray(p, dtype=float)
            q = np.asarray(q, dtype=float)
            m = 0.5 * (p + q)
            return float(0.5 * np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0]))
                         + 0.5 * np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0])))

        for label in result["models"]:
            model_shares = result["models"][label]
            for name in ("numeric", "punct_only", "eos", "pos_late"):
                recomputed = [js(model_shares["features"][name][layer]["shares"],
                                 model_shares["global_shares"][layer])
                              for layer in range(7)]
                stored = main_summary["models"][label]["features"][name]["js_bits"]
                checks[f"{label}:{name}"] = float(np.max(np.abs(
                    np.array(recomputed) - np.array(stored))))
    result["js_consistency_max_abs_difference"] = checks
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("wrote", OUT, os.path.getsize(OUT), "bytes",
          "inference", round(total_inference, 1), "s",
          "elapsed", round(time.monotonic() - started, 1), "s")


if __name__ == "__main__":
    main()
