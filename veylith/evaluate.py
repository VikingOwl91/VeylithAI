#!/usr/bin/env python3
"""CPU-only evaluation for veylith checkpoints.

The script never initializes CUDA/ROCm: the device environment is emptied
before torch is imported, and every tensor stays on the CPU.

It produces:
- an inventory of the checkpoints that actually exist
- main next-token cross-entropy on shared validation sections, with
  perplexity and bits per evaluated character
- deterministic text generation (greedy plus sampled with fixed seeds)
- MoE router statistics for the v2 checkpoints

Usage:
    python3 veylith/evaluate.py --out veylith/evaluation/results.json
"""

import os

# Hard CPU-only guarantee: this must happen before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import importlib.util
import json
import math
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))          # veylith/
REPO_ROOT = os.path.dirname(ROOT)
V1_DIR = os.path.join(ROOT, "v1")
V2_DIR = os.path.join(ROOT, "v2")
CORPUS_PATH = os.path.join(V2_DIR, "data", "tinyshakespeare.txt")
BPE_MODEL_PATH = os.path.join(V2_DIR, "data", "tinyshakespeare_bpe1000.model")
BPE_IDS_PATH = os.path.join(V2_DIR, "data", "tinyshakespeare_bpe1000.model.ids.pt")

CHECKPOINTS = [
    {"name": "v1-char", "path": os.path.join(V1_DIR, "checkpoints", "ckpt_char.pt")},
    {"name": "v1-bpe", "path": os.path.join(V1_DIR, "checkpoints", "ckpt.pt")},
    {"name": "v1-bpe-5000it", "path": os.path.join(V1_DIR, "checkpoints", "ckpt_bpe_5000it.pt")},
    {"name": "v2-bpe", "path": os.path.join(V2_DIR, "checkpoints", "ckpt.pt")},
    {"name": "v2-p1-1-temp", "path": "/tmp/opencode/p1-1-phaseb/ckpt.pt", "temporary": True},
]

PROMPTS = [
    "ROMEO:",
    "JULIET:",
    "To be, or not to be",
    "The king",
    "the the the",
    "?!3 $&',-.:;",
]
SEEDS = [1337, 42, 7]
TEMPERATURE = 0.8
TOP_K = 40


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = _load_module("veylith_v1_train", os.path.join(V1_DIR, "train.py"))
V2 = _load_module("veylith_v2_train", os.path.join(V2_DIR, "train.py"))


# ----------------------------------------------------------------- helpers

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path):
    absolute = os.path.abspath(path)
    if absolute.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(absolute, REPO_ROOT)
    return absolute


def read_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as handle:
        return handle.read()


def _bpe_processor():
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=BPE_MODEL_PATH)


def boundaries(corpus, bpe_vocab_size=1000):
    """Validation start offsets for the char and BPE tokenizers.

    v1 char splits at int(0.9 * chars); the BPE models split at
    int(0.9 * tokens). The shared sections must start after both.
    """
    char_start = int(0.9 * len(corpus))
    if os.path.exists(BPE_IDS_PATH):
        ids = torch.load(BPE_IDS_PATH, weights_only=True).tolist()
    else:
        sp = _bpe_processor()
        ids = sp.encode(corpus, out_type=int)
    sp = _bpe_processor()
    bpe_start_token = int(0.9 * len(ids))
    bpe_start_char = len(sp.decode(ids[:bpe_start_token]))
    return {
        "char_model_start": char_start,
        "bpe_model_start_token": bpe_start_token,
        "bpe_model_start_char": bpe_start_char,
        "shared_start": max(char_start, bpe_start_char),
    }


def make_sections(corpus, count=6, width=512, gap=128):
    marks = boundaries(corpus)
    start = marks["shared_start"]
    sections = []
    for index in range(count):
        begin = start + index * (width + gap)
        text = corpus[begin:begin + width]
        if len(text) < width:
            break
        sections.append({
            "index": index,
            "start": begin,
            "end": begin + len(text),
            "chars": len(text),
            "sha256": sha256_text(text),
            "text": text,
        })
    if not sections:
        raise ValueError("no section fits into the validation region")
    return sections, marks


def metrics_from_totals(nats, tokens, chars):
    """Main CE and its derived values; chars are the evaluated target chars."""
    if tokens <= 0 or chars <= 0:
        raise ValueError("tokens and chars must be positive")
    main_ce = nats / tokens
    return {
        "main_ce": round(main_ce, 6),
        "main_perplexity": round(math.exp(main_ce), 6),
        "bits_per_char": round(nats / chars / math.log(2), 6),
        "nats_per_char": round(nats / chars, 6),
    }


def repetition_ratio(ids, n):
    if len(ids) < n:
        return None
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


# -------------------------------------------------------------- checkpoints

def family_of(cfg):
    if "n_experts" in cfg:
        return "v2"
    if "n_layer" in cfg and "n_embd" in cfg and "n_kv_head" not in cfg:
        return "v1"
    raise ValueError("unknown config schema")


def build_tokenizer(ckpt, family_root, module):
    info = ckpt.get("tokenizer")
    if info is None and "stoi" in ckpt and "itos" in ckpt:
        info = {"type": "char", "stoi": ckpt["stoi"], "itos": ckpt["itos"]}
    if not isinstance(info, dict):
        return None
    info = dict(info)
    if info.get("type") == "bpe":
        model_path = info.get("model_path", "")
        if model_path and not os.path.isabs(model_path):
            info["model_path"] = os.path.join(family_root, model_path)
        if not os.path.exists(info["model_path"]):
            return None
    try:
        encode, decode = module.tokenizer_from_info(info)
    except Exception:
        return None
    return encode, decode, info.get("type", "unknown")


def load_entry(entry):
    record = {
        "name": entry["name"],
        "path": display_path(entry["path"]),
        "temporary": entry.get("temporary", False),
    }
    path = entry["path"]
    if not os.path.exists(path):
        record["status"] = "missing"
        return record
    record["sha256"] = sha256_file(path)
    record["size_bytes"] = os.path.getsize(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        record["status"] = "unreadable"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    cfg_dict = ckpt.get("config")
    if not isinstance(cfg_dict, dict) or "model" not in ckpt:
        record["status"] = "incompatible"
        record["error"] = "checkpoint has no config dict or no model state"
        return record
    try:
        family = family_of(cfg_dict)
        module = V1 if family == "v1" else V2
        cfg = module.Config(**cfg_dict)
        model = module.GPT(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
    except Exception as exc:
        record["status"] = "incompatible"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    tokenizer = build_tokenizer(ckpt, V1_DIR if family == "v1" else V2_DIR, module)
    if tokenizer is None:
        record["status"] = "incompatible"
        record["error"] = "tokenizer missing or unusable"
        return record
    encode, decode, tokenizer_type = tokenizer
    args = ckpt.get("args") or {}
    record.update({
        "status": "ok",
        "family": family,
        "tokenizer": tokenizer_type,
        "architecture": {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_kv_head": getattr(cfg, "n_kv_head", None),
            "n_embd": cfg.n_embd,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "n_experts": getattr(cfg, "n_experts", None),
            "n_experts_per_tok": getattr(cfg, "n_experts_per_tok", None),
            "first_k_dense": getattr(cfg, "first_k_dense", None),
            "n_mtp": getattr(cfg, "n_mtp", None),
            "mlp_ratio": getattr(cfg, "mlp_ratio", None),
        },
        "params": sum(p.numel() for p in model.parameters()),
        "buffers": sum(b.numel() for b in model.buffers()),
        "checkpoint": {
            "format": ckpt.get("format"),
            "kind": ckpt.get("kind"),
            "step": ckpt.get("step"),
            "val_loss": ckpt.get("val_loss"),
            "metrics": ckpt.get("metrics"),
            "selection": ckpt.get("selection"),
            "args": {key: args.get(key) for key in
                     ("tokenizer", "bpe_vocab_size", "max_iters", "seed", "batch_size")},
        },
        "_model": model,
        "_encode": encode,
        "_decode": decode,
        "_cfg": cfg,
    })
    return record


def inventory(record):
    return {key: value for key, value in record.items() if not key.startswith("_")}


# --------------------------------------------------------------- evaluation

def evaluate_model(record, sections):
    model, encode, decode = record["_model"], record["_encode"], record["_decode"]
    block = record["_cfg"].block_size
    total_nats, total_tokens, total_chars = 0.0, 0, 0
    per_section = []
    for section in sections:
        ids = encode(section["text"])
        if len(ids) < 2:
            per_section.append({"index": section["index"], "skipped": "fewer than 2 tokens"})
            continue
        if decode(ids) != section["text"]:
            raise ValueError(f"tokenizer roundtrip failed for section {section['index']} "
                             f"in {record['name']}")
        prefix_cache = {0: 0}

        def prefix_chars(k):
            if k not in prefix_cache:
                prefix_cache[k] = len(decode(ids[:k]))
            return prefix_cache[k]

        sec_nats, sec_tokens, sec_chars = 0.0, 0, 0
        for start in range(0, len(ids) - 1, block):
            take = min(block, len(ids) - 1 - start)
            x = torch.tensor([ids[start:start + take]], dtype=torch.long)
            y = torch.tensor([ids[start + 1:start + take + 1]], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model(x)
            nats = float(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                         reduction="sum"))
            chars = prefix_chars(start + take + 1) - prefix_chars(start + 1)
            if chars <= 0:
                raise ValueError(f"non-positive char span in section {section['index']}")
            sec_nats += nats
            sec_tokens += take
            sec_chars += chars
        per_section.append({
            "index": section["index"],
            "tokens": sec_tokens,
            "chars": sec_chars,
            "nats": round(sec_nats, 6),
        })
        total_nats += sec_nats
        total_tokens += sec_tokens
        total_chars += sec_chars
    return {
        "measured": {
            "tokens": total_tokens,
            "chars": total_chars,
            "nats": round(total_nats, 6),
            "sections": per_section,
        },
        "derived": metrics_from_totals(total_nats, total_tokens, total_chars),
    }


# --------------------------------------------------------------- generation

def generate_ids(model, prompt_ids, n_new_tokens, temperature, top_k, seed, greedy):
    generator = None if greedy else torch.Generator().manual_seed(seed)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    produced = []
    for _ in range(n_new_tokens):
        with torch.no_grad():
            logits, _ = model(ids[:, -model.block_size:])
        step_logits = logits[:, -1, :]
        if greedy:
            next_id = step_logits.argmax(dim=-1, keepdim=True)
        else:
            step_logits = step_logits / temperature
            values, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
            step_logits[step_logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(step_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        produced.append(int(next_id.item()))
        ids = torch.cat([ids, next_id], dim=1)
    return produced


def generation_runs(record, n_new_tokens, temperature, top_k, seeds):
    model, encode, decode = record["_model"], record["_encode"], record["_decode"]
    runs = []
    for prompt in PROMPTS:
        try:
            prompt_ids = encode(prompt)
        except KeyError as exc:
            runs.append({"model": record["name"], "prompt": prompt, "mode": "skipped",
                         "reason": f"tokenizer does not support {exc.args[0]!r}"})
            continue
        if not prompt_ids:
            runs.append({"model": record["name"], "prompt": prompt, "mode": "skipped",
                         "reason": "empty encoding"})
            continue
        modes = [("greedy", None)] + [("sample", seed) for seed in seeds]
        for mode, seed in modes:
            ids = generate_ids(model, prompt_ids, n_new_tokens, temperature, top_k,
                               seed, greedy=(mode == "greedy"))
            text = decode(ids)
            rep3 = repetition_ratio(ids, 3)
            rep4 = repetition_ratio(ids, 4)
            anomalies = []
            if not text:
                anomalies.append("empty output")
            if "\ufffd" in text:
                anomalies.append("replacement character")
            if any(ord(char) < 32 and char not in "\n\t" for char in text):
                anomalies.append("control character")
            if rep3 is not None and rep3 > 0.5:
                anomalies.append("degenerate repetition (rep3 > 0.5)")
            runs.append({
                "model": record["name"],
                "prompt": prompt,
                "mode": mode,
                "seed": seed,
                "measured": {
                    "text": text,
                    "n_new_tokens": len(ids),
                    "n_chars": len(text),
                },
                "derived": {
                    "rep3": None if rep3 is None else round(rep3, 6),
                    "rep4": None if rep4 is None else round(rep4, 6),
                    "unique_token_ratio": round(len(set(ids)) / len(ids), 6) if ids else 0.0,
                },
                "anomalies": anomalies,
            })
    return runs


# ------------------------------------------------------------------- router

def router_stats(record, sections):
    model, encode = record["_model"], record["_encode"]
    cfg = record["_cfg"]
    block = cfg.block_size
    n_experts = cfg.n_experts
    k = cfg.n_experts_per_tok
    gates = [(index, item.mlp.gate) for index, item in enumerate(model.blocks)
             if getattr(item, "is_moe", False)]
    if not gates:
        return None

    chunks = []
    section_tokens = []
    for section in sections:
        ids = encode(section["text"])
        spans = [(start, min(block, len(ids) - 1 - start))
                 for start in range(0, len(ids) - 1, block)]
        chunks.append((ids, spans))
        section_tokens.append(sum(take for _, take in spans))

    accumulators = {
        layer: {
            "counts": torch.zeros(n_experts, dtype=torch.float64),
            "prob_sum": torch.zeros(n_experts, dtype=torch.float64),
            "weight_sum": torch.zeros(n_experts, dtype=torch.float64),
            "entropy_sum": 0.0,
            "tokens": 0,
            "section_counts": [torch.zeros(n_experts, dtype=torch.float64)
                               for _ in sections],
        }
        for layer, _ in gates
    }
    cursor = {"section": 0}

    def make_hook(layer):
        def hook(module, inputs, output):
            state = accumulators[layer]
            probs = torch.softmax(output.detach().float(), dim=-1)
            _, top_idx = probs.topk(k, dim=-1)
            flat = top_idx.reshape(-1)
            counts = torch.bincount(flat, minlength=n_experts).double()
            state["counts"] += counts
            state["section_counts"][cursor["section"]] += counts
            state["prob_sum"] += probs.sum(0).double()
            state["entropy_sum"] += float(-(probs.clamp_min(1e-12).log() * probs).sum())
            weights = torch.softmax(output.detach().float().topk(k, dim=-1).values, dim=-1)
            state["weight_sum"].scatter_add_(0, flat, weights.reshape(-1).double())
            state["tokens"] += probs.shape[0]
        return hook

    handles = [gate.register_forward_hook(make_hook(layer)) for layer, gate in gates]
    try:
        for section_index, (ids, spans) in enumerate(chunks):
            cursor["section"] = section_index
            for start, take in spans:
                x = torch.tensor([ids[start:start + take]], dtype=torch.long)
                with torch.no_grad():
                    model(x)
    finally:
        for handle in handles:
            handle.remove()

    measured, derived = {}, {}
    uniform_slot_share = 100.0 / n_experts
    for layer, _ in gates:
        state = accumulators[layer]
        tokens = state["tokens"]
        slots = tokens * k
        counts = [int(value) for value in state["counts"].tolist()]
        shares = [100.0 * count / slots for count in counts]
        probs = [float(value) / tokens for value in state["prob_sum"]]
        weights = [float(weight) / count if count else 0.0
                   for weight, count in zip(state["weight_sum"], counts)]
        section_shares = []
        for index, section_count in enumerate(state["section_counts"]):
            section_slots = section_tokens[index] * k
            section_shares.append([100.0 * float(value) / section_slots
                                   for value in section_count])
        max_deviation = max(
            abs(section_shares[index][expert] - shares[expert])
            for index in range(len(section_shares))
            for expert in range(n_experts)
        )
        tv_distances = [0.5 * sum(abs(section_shares[index][expert] / 100.0 - shares[expert] / 100.0)
                                  for expert in range(n_experts))
                        for index in range(len(section_shares))]
        measured[layer] = {
            "tokens": tokens,
            "slots": slots,
            "counts": counts,
            "prob_sum": [round(float(value), 6) for value in state["prob_sum"].tolist()],
            "weight_sum": [round(float(value), 6) for value in state["weight_sum"].tolist()],
            "entropy_sum_nats": round(state["entropy_sum"], 6),
            "section_counts": [[int(value) for value in section.tolist()]
                               for section in state["section_counts"]],
            "section_tokens": section_tokens,
        }
        derived[layer] = {
            "share_pct_of_slots": [round(value, 6) for value in shares],
            "token_coverage_pct": [round(100.0 * count / tokens, 6) for count in counts],
            "mean_router_prob": [round(value, 6) for value in probs],
            "mean_topk_weight": [round(value, 6) for value in weights],
            "entropy_mean_nats": round(state["entropy_sum"] / tokens, 6),
            "entropy_norm": round(state["entropy_sum"] / tokens / math.log(n_experts), 6),
            "min_share_pct": round(min(shares), 6),
            "max_share_pct": round(max(shares), 6),
            "unused_experts": [expert for expert, count in enumerate(counts) if count == 0],
            "rare_experts": [expert for expert, share in enumerate(shares)
                             if share < 0.5 * uniform_slot_share],
            "uniform_slot_share_pct": round(uniform_slot_share, 6),
            "section_shares_pct": [[round(value, 6) for value in row]
                                   for row in section_shares],
            "section_max_abs_dev_pp": round(max_deviation, 6),
            "section_mean_tv": round(sum(tv_distances) / len(tv_distances), 6),
        }
    return {"measured": measured, "derived": derived}


# --------------------------------------------------------------------- main

def parse_extra_checkpoints(values):
    """Parse repeated NAME=PATH flags into checkpoint entries."""
    entries = []
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise SystemExit(f"--extra-checkpoint expects NAME=PATH, got {value!r}")
        entries.append({"name": name.strip(), "path": path.strip()})
    return entries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only veylith evaluation")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results.json"))
    parser.add_argument("--extra-checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="also evaluate this checkpoint; repeatable")
    parser.add_argument("--sections", type=int, default=6)
    parser.add_argument("--section-chars", type=int, default=512)
    parser.add_argument("--gap-chars", type=int, default=128)
    parser.add_argument("--gen-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-router", action="store_true")
    return parser.parse_args(argv)


def build_payload(args):
    corpus = read_corpus()
    sections, marks = make_sections(corpus, args.sections, args.section_chars, args.gap_chars)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    entries = CHECKPOINTS + parse_extra_checkpoints(args.extra_checkpoint)
    records = [load_entry(entry) for entry in entries]
    loaded = [record for record in records if record["status"] == "ok"]

    evaluation = {}
    for record in loaded:
        evaluation[record["name"]] = evaluate_model(record, sections)

    router = {}
    if not args.no_router:
        for record in loaded:
            if record["family"] == "v2":
                stats = router_stats(record, sections)
                if stats is not None:
                    router[record["name"]] = stats

    generation = None
    if not args.no_generation:
        generation = {"config": {
            "max_new_tokens": args.gen_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seeds": seeds,
            "greedy": True,
            "prompts": PROMPTS,
        }, "runs": []}
        for record in loaded:
            generation["runs"].extend(
                generation_runs(record, args.gen_tokens, args.temperature, args.top_k, seeds))

    return {
        "generated_by": "veylith/evaluate.py",
        "device": "cpu",
        "torch": torch.__version__,
        "protocol": {
            "corpus": display_path(CORPUS_PATH),
            "corpus_sha256": sha256_text(corpus),
            "corpus_chars": len(corpus),
            "corpus_ascii": corpus.isascii(),
            "boundaries": marks,
            "sections": [{key: section[key] for key in
                          ("index", "start", "end", "chars", "sha256")}
                         for section in sections],
            "evaluation": {
                "metric": "main next-token cross-entropy only (no MTP, no aux)",
                "chunking": "non-overlapping chunks of up to block_size tokens within a "
                            "section; the first token of a chunk has no left context; the "
                            "last token of a section is not predicted",
                "bits_per_char": "total nats over evaluated targets divided by the exact "
                                 "character span of those targets (prefix decodes)",
            },
            "generation": {
                "max_new_tokens": args.gen_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "seeds": seeds,
                "greedy": True,
                "prompts": PROMPTS,
                "note": "sampling uses a private CPU generator per run; no EOS handling, "
                        "so every run fills the token budget",
            },
            "router": {
                "probability": "softmax over all experts before top-k",
                "selection": "top-k of the same distribution",
                "weights": "softmax over the top-k logits",
                "uniform_slot_share_pct": 12.5,
                "uniform_token_coverage_pct": 25.0,
            },
        },
        "checkpoints": [inventory(record) for record in records],
        "evaluation": evaluation,
        "router": router,
        "generation": generation,
    }


def main(argv=None):
    args = parse_args(argv)
    payload = build_payload(args)
    V2.write_json(args.out, payload)
    print(f"wrote {args.out}")
    for name, entry in payload["evaluation"].items():
        derived = entry["derived"]
        print(f"  {name:<16} tokens={entry['measured']['tokens']:5d} "
              f"chars={entry['measured']['chars']:5d} "
              f"main_ce={derived['main_ce']:.4f} ppl={derived['main_perplexity']:.2f} "
              f"bits/char={derived['bits_per_char']:.4f}")
    for name, entry in payload["router"].items():
        for layer, derived in entry["derived"].items():
            print(f"  router {name} layer {layer}: entropy_norm={derived['entropy_norm']:.3f} "
                  f"share_min={derived['min_share_pct']:.1f}% "
                  f"share_max={derived['max_share_pct']:.1f}% "
                  f"unused={derived['unused_experts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
