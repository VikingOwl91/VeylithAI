#!/usr/bin/env python3
"""CPU-only training audit for the veylith checkpoints.

Answers four questions with measurements instead of guesses:

- train/validation main-CE gap of every checkpoint
- how many parameters are active per token in the main forward pass
- how the composite loss splits into main CE, MTP and MoE aux, numerically
  and in terms of the gradient direction
- what the attention layout actually is

Reuses the model loading and section code from evaluate.py. Never touches a
GPU and never updates a weight: the gradient analysis only calls backward.

Usage:
    python3 veylith/audit.py --out veylith/evaluation/audit.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import argparse
import json
import math
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate as E  # noqa: E402


TRAIN_SECTION_START = 50_000


def make_train_sections(corpus, count=6, width=512, gap=128):
    """Fixed training-text sections, all before the char and BPE val splits."""
    limit = min(E.boundaries(corpus)["char_model_start"],
                E.boundaries(corpus)["bpe_model_start_char"])
    sections = []
    for index in range(count):
        begin = TRAIN_SECTION_START + index * (width + gap)
        text = corpus[begin:begin + width]
        if len(text) < width or begin + width > limit:
            break
        sections.append({
            "index": index,
            "start": begin,
            "end": begin + len(text),
            "chars": len(text),
            "sha256": E.sha256_text(text),
            "text": text,
        })
    if len(sections) != count:
        raise ValueError("not enough training text before the validation split")
    return sections


def strip_text(sections):
    return [{key: section[key] for key in ("index", "start", "end", "chars", "sha256")}
            for section in sections]


def parameter_accounting(record):
    """Trainable parameters by role and active parameters for one token.

    Conventions: dense matrices and the full embedding tables count as active
    (not just the touched row); exactly top_k routed experts are active per
    token; the router, shared expert, norms, attention and dense FFN are always
    active. MTP has no extra parameters: it reuses the trunk and the output
    head, so the active count is identical for main-only and MTP training.
    """
    model, cfg, family = record["_model"], record["_cfg"], record["family"]
    categories = {
        "token_embedding": 0,
        "position_embedding": 0,
        "attention": 0,
        "norms": 0,
        "dense_ffn": 0,
        "routed_experts": 0,
        "shared_expert": 0,
        "router": 0,
        "output_head": 0,
        "other": 0,
    }
    routed_per_layer = {}
    for name, param in model.named_parameters():
        count = param.numel()
        if name.startswith("tok_emb"):
            categories["token_embedding"] += count
        elif name.startswith("pos_emb"):
            categories["position_embedding"] += count
        elif name.startswith("head"):
            categories["output_head"] += count
        elif name.startswith("ln_f"):
            categories["norms"] += count
        elif name.startswith("blocks."):
            parts = name.split(".")
            layer = int(parts[1])
            rest = ".".join(parts[2:])
            if rest.startswith("attn."):
                categories["attention"] += count
            elif rest.startswith("ln"):
                categories["norms"] += count
            elif rest.startswith("mlp."):
                sub = rest[len("mlp."):]
                if sub.startswith("experts."):
                    categories["routed_experts"] += count
                    routed_per_layer[layer] = routed_per_layer.get(layer, 0) + count
                elif sub.startswith("shared."):
                    categories["shared_expert"] += count
                elif sub.startswith("gate."):
                    categories["router"] += count
                else:
                    categories["dense_ffn"] += count
            else:
                categories["other"] += count
        else:
            categories["other"] += count

    total = sum(categories.values())
    assert total == sum(p.numel() for p in model.parameters())

    n_moe = len(routed_per_layer)
    n_experts = getattr(cfg, "n_experts", None)
    top_k = getattr(cfg, "n_experts_per_tok", None)
    expert_size = None
    if n_moe:
        per_layer = {layer: value // n_experts for layer, value in routed_per_layer.items()}
        assert len(set(per_layer.values())) == 1, per_layer
        expert_size = next(iter(per_layer.values()))
        routed_total = categories["routed_experts"]
        always_active = total - routed_total
        active = always_active + n_moe * top_k * expert_size
    else:
        always_active = total
        active = total

    return {
        "total_trainable": total,
        "categories": categories,
        "moe_layers": n_moe,
        "experts_per_layer": n_experts,
        "top_k": top_k,
        "routed_expert_size": expert_size,
        "always_active": always_active,
        "active_per_token_main": active,
        "inactive_per_token": total - active,
        "total_to_active_ratio": round(total / active, 6),
        "mtp_extra_parameters": 0,
        "convention": "full dense matrices and full embedding tables count as active; "
                      "top_k routed experts active per token; MTP reuses trunk and head",
    }


def training_budget(record, train_tokens, selection_step=None):
    args = record["checkpoint"]["args"]
    cfg = record["_cfg"]
    steps = args.get("max_iters")
    batch = args.get("batch_size")
    block = cfg.block_size
    if not steps or not batch:
        return {"status": "not determinable"}
    per_step = batch * block
    total = steps * per_step
    possible_starts = train_tokens - block + 1
    windows = steps * batch
    mean_draws = windows / possible_starts
    budget = {
        "status": "reconstructed from stored args and config",
        "steps": steps,
        "batch_size": batch,
        "block_size": block,
        "tokens_per_step": per_step,
        "token_positions": total,
        "train_tokens": train_tokens,
        "corpus_passes": round(total / train_tokens, 3),
        "windows_drawn": windows,
        "possible_window_starts": possible_starts,
        "mean_draws_per_start": round(mean_draws, 4),
        "expected_unseen_fraction": round(math.exp(-mean_draws), 4),
        "sampling": "uniform random windows, not epochs",
    }
    if selection_step is not None:
        exposure = selection_step * per_step
        budget["selected_step"] = selection_step
        budget["selected_token_positions"] = exposure
        budget["selected_corpus_passes"] = round(exposure / train_tokens, 3)
    return budget


def tensor_norm(vector):
    return float(vector.norm())


def cosine(left, right):
    denominator = left.norm() * right.norm()
    if denominator == 0:
        return None
    return float(torch.dot(left, right) / denominator)


def gradient_components(record, chunks, max_batches=2):
    """Separate backward passes per loss term; no optimizer step."""
    model, cfg = record["_model"], record["_cfg"]
    if getattr(cfg, "n_mtp", 0) < 1:
        return {"status": "model has no MTP term"}
    weights_before = sum(float(param.sum()) for param in model.parameters())
    batch_results = []
    for x, y in chunks[:max_batches]:
        logits, loss, metrics = model(x, y, return_metrics=True)

        def flat_grad():
            parts = []
            missing = 0
            for param in model.parameters():
                if param.grad is None:
                    missing += 1
                    parts.append(torch.zeros_like(param).reshape(-1))
                else:
                    parts.append(param.grad.detach().reshape(-1))
            return torch.cat(parts), missing

        model.zero_grad(set_to_none=True)
        metrics["main_ce"].backward(retain_graph=True)
        g_main, missing_main = flat_grad()

        model.zero_grad(set_to_none=True)
        metrics["mtp_ce"].backward(retain_graph=True)
        g_mtp, missing_mtp = flat_grad()

        model.zero_grad(set_to_none=True)
        metrics["aux_loss"].backward(retain_graph=True)
        g_aux, missing_aux = flat_grad()
        model.zero_grad(set_to_none=True)

        composite = g_main + cfg.mtp_weight * g_mtp + cfg.moe_aux_weight * g_aux
        main_mtp = cosine(g_main, g_mtp)
        main_aux = cosine(g_main, g_aux)
        batch_results.append({
            "tokens": int(x.shape[1]),
            "measured": {
                "grad_norm_main": round(tensor_norm(g_main), 6),
                "grad_norm_mtp": round(tensor_norm(g_mtp), 6),
                "grad_norm_aux": round(tensor_norm(g_aux), 6),
                "grad_norm_mtp_weighted": round(cfg.mtp_weight * tensor_norm(g_mtp), 6),
                "grad_norm_aux_weighted": round(cfg.moe_aux_weight * tensor_norm(g_aux), 6),
                "grad_norm_composite": round(tensor_norm(composite), 6),
                "params_without_grad": {"main": missing_main, "mtp": missing_mtp,
                                        "aux": missing_aux},
            },
            "derived": {
                "cosine_main_mtp": None if main_mtp is None else round(main_mtp, 6),
                "cosine_main_aux": None if main_aux is None else round(main_aux, 6),
                "cosine_composite_main": round(cosine(composite, g_main), 6),
                "mtp_share_of_composite_norm": round(
                    cfg.mtp_weight * tensor_norm(g_mtp) / tensor_norm(composite), 6),
            },
        })
        del g_main, g_mtp, g_aux, composite
    weights_after = sum(float(param.sum()) for param in model.parameters())
    return {
        "status": "measured",
        "weights_unchanged": weights_before == weights_after,
        "batches": batch_results,
    }


def loss_decomposition(record, chunks):
    model, cfg = record["_model"], record["_cfg"]
    main_nats = mtp_nats = 0.0
    main_tokens = mtp_tokens = 0
    aux_values = []
    with torch.no_grad():
        for x, y in chunks:
            _, _, metrics = model(x, y, return_metrics=True)
            take = int(x.shape[1])
            main_nats += float(metrics["main_ce"]) * take
            main_tokens += take
            if metrics["mtp_ce"] is not None:
                mtp_nats += float(metrics["mtp_ce"]) * (take - 1)
                mtp_tokens += take - 1
            aux_values.append(float(metrics["aux_loss"]))
    main_ce = main_nats / main_tokens
    mtp_ce = mtp_nats / mtp_tokens if mtp_tokens else None
    aux = sum(aux_values) / len(aux_values)
    composite = main_ce
    if mtp_ce is not None:
        composite += cfg.mtp_weight * mtp_ce
    composite += cfg.moe_aux_weight * aux
    terms = {"main_ce": main_ce,
             "mtp_weighted": (cfg.mtp_weight * mtp_ce) if mtp_ce is not None else None,
             "aux_weighted": cfg.moe_aux_weight * aux}
    shares = {key: (round(100.0 * value / composite, 6) if value is not None else None)
              for key, value in terms.items()}
    return {
        "measured": {
            "main_ce": round(main_ce, 6),
            "mtp_ce": round(mtp_ce, 6) if mtp_ce is not None else None,
            "aux_loss_raw": round(aux, 6),
            "mtp_weight": cfg.mtp_weight,
            "aux_weight": cfg.moe_aux_weight,
            "composite": round(composite, 6),
            "tokens": main_tokens,
        },
        "derived": {
            "weighted_terms": {key: (round(value, 6) if value is not None else None)
                               for key, value in terms.items()},
            "share_pct_of_composite": shares,
        },
    }


def documented_decomposition(chars_per_token=2.263192):
    """Values recorded in the v2 checkpoint commit message (347aa70).

    The message reports the main CE as 2.256 bits/char and the composite as
    6.0358. Aux is 5.0889 nats, measured later on the same checkpoint on 16
    fixed validation windows (weight 0.01, so it barely matters).
    """
    main_bits_per_char = 2.256
    composite = 6.0358
    aux = 5.0889
    main_nats_per_token = main_bits_per_char * math.log(2) * chars_per_token
    remainder = composite - main_nats_per_token - 0.01 * aux
    mtp_ce = remainder / 0.5
    weighted = {"main_ce": main_nats_per_token, "mtp_weighted": 0.5 * mtp_ce,
                "aux_weighted": 0.01 * aux}
    return {
        "source": "commit 347aa70 message (training-time eval, random windows)",
        "measured": {"main_bits_per_char": main_bits_per_char,
                     "composite": composite,
                     "main_nats_per_token_implied": round(main_nats_per_token, 6),
                     "mtp_ce_implied": round(mtp_ce, 6)},
        "derived": {"weighted_terms": {key: round(value, 6) for key, value in weighted.items()},
                    "share_pct_of_composite": {key: round(100.0 * value / composite, 6)
                                               for key, value in weighted.items()}},
    }


def attention_layout(cfg, family):
    if family == "v1":
        return {"full_layers": list(range(cfg.n_layer)), "swa_layers": [],
                "window": None, "note": "v1 uses full causal attention in every layer"}
    full = [index for index in range(cfg.n_layer)
            if (index + 1) % cfg.full_attn_every == 0]
    swa = [index for index in range(cfg.n_layer) if index not in full]
    return {
        "full_layers": full,
        "swa_layers": swa,
        "window": cfg.sliding_window,
        "full_attn_every": cfg.full_attn_every,
        "note": "SWA attends to the last `window` positions including the current one; "
                "the full layers see the whole prefix. With a full layer at depth 3, "
                "every earlier position can reach the final layer through it, so the "
                "theoretical reach is the full context; the empirical use is not measured.",
    }


def model_chunks(record, sections):
    model, encode = record["_model"], record["_encode"]
    cfg = record["_cfg"]
    block = cfg.block_size
    chunks = []
    for section in sections:
        ids = encode(section["text"])
        for start in range(0, len(ids) - 1, block):
            take = min(block, len(ids) - 1 - start)
            x = torch.tensor([ids[start:start + take]], dtype=torch.long)
            y = torch.tensor([ids[start + 1:start + take + 1]], dtype=torch.long)
            chunks.append((x, y))
    return chunks


SELECTION_STEPS = {"v2-bpe": 500}  # recorded in commit 347aa70


def build_audit(args):
    corpus = E.read_corpus()
    train_sections = make_train_sections(corpus, args.sections, args.section_chars, args.gap_chars)
    val_sections, marks = E.make_sections(corpus, args.sections, args.section_chars, args.gap_chars)
    entries = E.CHECKPOINTS + E.parse_extra_checkpoints(args.extra_checkpoint)
    records = [E.load_entry(entry) for entry in entries]
    loaded = [record for record in records if record["status"] == "ok"]
    val_chunks = {record["name"]: model_chunks(record, val_sections) for record in loaded}

    checkpoints, gaps, decompositions, gradients, attention = [], {}, {}, {}, {}
    for record in loaded:
        name = record["name"]
        inventory = E.inventory(record)
        inventory["accounting"] = parameter_accounting(record)
        train_result = E.evaluate_model(record, train_sections)
        val_result = E.evaluate_model(record, val_sections)
        train_tokens = (E.boundaries(corpus)["bpe_model_start_token"]
                        if record["tokenizer"] == "bpe" else marks["char_model_start"])
        inventory["training_budget"] = training_budget(record, train_tokens,
                                                       SELECTION_STEPS.get(name))
        checkpoints.append(inventory)

        gaps[name] = {
            "train": train_result,
            "val": val_result,
            "derived": {
                "ce_gap_nats": round(val_result["derived"]["main_ce"]
                                     - train_result["derived"]["main_ce"], 6),
                "bits_per_char_gap": round(val_result["derived"]["bits_per_char"]
                                           - train_result["derived"]["bits_per_char"], 6),
                "val_train_ratio": round(val_result["derived"]["main_ce"]
                                         / train_result["derived"]["main_ce"], 6),
            },
        }

        if record["family"] == "v2":
            decompositions[name] = loss_decomposition(record, val_chunks[name])
            gradients[name] = gradient_components(record, val_chunks[name], args.grad_batches)
            attention[name] = attention_layout(record["_cfg"], "v2")
        else:
            attention[name] = attention_layout(record["_cfg"], "v1")

    return {
        "generated_by": "veylith/audit.py",
        "device": "cpu",
        "torch": torch.__version__,
        "protocol": {
            "corpus": E.display_path(E.CORPUS_PATH),
            "corpus_sha256": E.sha256_text(corpus),
            "train_sections": strip_text(train_sections),
            "val_sections": strip_text(val_sections),
            "boundaries": marks,
            "evaluation": "main next-token CE only; same chunking as evaluate.py",
            "gradients": "one separate backward per loss term, no optimizer step",
        },
        "checkpoints": checkpoints,
        "gaps": gaps,
        "loss_decomposition": decompositions,
        "documented_decomposition": documented_decomposition(),
        "gradients": gradients,
        "attention": attention,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only veylith training audit")
    parser.add_argument("--out", default=os.path.join(HERE, "evaluation", "audit.json"))
    parser.add_argument("--sections", type=int, default=6)
    parser.add_argument("--section-chars", type=int, default=512)
    parser.add_argument("--gap-chars", type=int, default=128)
    parser.add_argument("--grad-batches", type=int, default=2)
    parser.add_argument("--extra-checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="also audit this checkpoint; repeatable")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payload = build_audit(args)
    E.V2.write_json(args.out, payload)
    print(f"wrote {args.out}")
    for name, entry in payload["gaps"].items():
        gap = entry["derived"]["ce_gap_nats"]
        print(f"  {name:<16} train {entry['train']['derived']['main_ce']:.4f} "
              f"val {entry['val']['derived']['main_ce']:.4f} gap {gap:+.4f}")
    for name, entry in payload["loss_decomposition"].items():
        print(f"  {name} shares {entry['derived']['share_pct_of_composite']}")
    for name, entry in payload["gradients"].items():
        first = entry["batches"][0]["derived"]
        print(f"  {name} grad cosine main/mtp {first['cosine_main_mtp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
