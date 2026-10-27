"""Evaluation on the frozen v3 splits: per-source CE, bits/char, router health."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import iter_source_batches, load_manifest, source_names, source_tensors
from .precision import autocast_context


@torch.no_grad()
def evaluate(model, cfg, split: str, *, max_batches: int | None = None) -> dict:
    """Evaluate one split per source. Validation is for selection; test is final."""
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    manifest = load_manifest(cfg.data_root)
    if split == "test" and cfg.select_metric not in ("macro_main_ce", "token_main_ce"):
        raise ValueError("test evaluation is final, not a selection signal")
    device = cfg.device
    sources = source_names(manifest)
    was_training = model.training
    model.eval()
    per_source = {}
    router_counts = None
    total_nats = total_tokens = total_chars = 0.0
    with autocast_context(device, cfg.precision):
        for source in sources:
            nats = tokens = chars = aux_weighted = 0.0
            for batch in iter_source_batches(
                    cfg.data_root, split, source, cfg.seq_len,
                    cfg.eval_batch_size, max_batches=max_batches):
                x, y, valid, batch_chars = source_tensors(batch, device)
                logits, _, metrics, router = model(
                    x, return_metrics=True, return_router=True)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    reduction="none").reshape(y.shape)
                mask = valid
                nats += float(ce[mask].sum())
                tokens += int(mask.sum())
                chars += float(batch_chars.sum())
                aux = float(metrics["aux_loss"])
                aux_weighted += aux * int(mask.sum())
                if router_counts is None:
                    router_counts = [torch.zeros(cfg.model.n_experts, dtype=torch.long)
                                     for _ in router]
                for layer, top_idx in enumerate(router):
                    counts = torch.bincount(top_idx.reshape(-1).cpu(),
                                            minlength=cfg.model.n_experts)
                    router_counts[layer] += counts
            if tokens == 0:
                per_source[source] = {"main_ce": None, "bits_per_char": None,
                                      "tokens": 0, "chars": 0, "nats": 0.0,
                                      "aux_loss": None}
                continue
            main_ce = nats / tokens
            per_source[source] = {
                "main_ce": main_ce,
                "bits_per_char": nats / (chars * math.log(2)) if chars else None,
                "tokens": int(tokens),
                "chars": int(chars),
                "nats": nats,
                "aux_loss": aux_weighted / tokens,
            }
            total_nats += nats
            total_tokens += tokens
            total_chars += chars
    model.train(was_training)

    valid_sources = [entry for entry in per_source.values() if entry["tokens"]]
    macro = (sum(entry["main_ce"] for entry in valid_sources) / len(valid_sources)
             if valid_sources else None)
    overall = {
        "split": split,
        "main_ce": total_nats / total_tokens if total_tokens else None,
        "macro_main_ce": macro,
        "bits_per_char": (total_nats / (total_chars * math.log(2))
                          if total_chars else None),
        "tokens": int(total_tokens),
        "chars": int(total_chars),
        "nats": total_nats,
        "aux_loss": (sum(entry["aux_loss"] * entry["tokens"] for entry in valid_sources)
                     / total_tokens if total_tokens else None),
    }
    router = _router_stats(router_counts)
    return {"split": split, "sources": per_source, "overall": overall,
            "router": router, "max_batches_per_source": max_batches}


def _router_stats(counts) -> dict | None:
    if counts is None:
        return None
    layers = []
    entropies = []
    for layer, counts_tensor in enumerate(counts):
        total = int(counts_tensor.sum())
        if total == 0:
            continue
        probs = counts_tensor.float() / total
        entropy = float(-(probs * (probs.clamp(min=1e-12).log())).sum())
        norm = entropy / math.log(len(probs))
        entropies.append(norm)
        layers.append({
            "layer": layer,
            "entropy_norm": norm,
            "share_pct": [round(float(value) * 100, 2) for value in probs],
            "unused_experts": [index for index, value in enumerate(probs)
                               if float(value) == 0.0],
        })
    return {"layers": layers,
            "mean_entropy_norm": sum(entropies) / len(entropies) if entropies else None}
