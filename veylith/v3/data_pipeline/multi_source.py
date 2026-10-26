"""Multi-source preparation, merge and reporting.

The single-source pipeline stays unchanged. Each source is prepared into
``<root>/sources/<name>`` with the existing resumable staging format, then
``merge_sources`` performs the cross-source steps in this documented order:

1. per-source streaming, normalization, filtering, per-source exact/near dedup
2. cross-source exact dedup (sha256 of normalized text)
3. cross-source near dedup (SimHash, Hamming <= 3)
4. deterministic split assignment (pure function of ``doc_key``)
5. deterministic quota filling on the train split (hash-ordered documents,
   per-source character budget derived from the target tokens)
6. deterministic bucket shuffle across sources for shard writing

Quotas are applied after dedup and split assignment, so duplicates never
consume budget. A source that cannot fill its quota is reported; other
sources are not increased to compensate (no artificial backfill).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

from . import dataset as ds
from . import prepare
from . import sources
from .prepare import FilterConfig, assign_splits_and_dedup, simhash64, split_for

PILOT_NOTE = "quota is a train-token target; val/test are kept outside the quota"


def _shuffle_key(seed: int, doc_key: str) -> str:
    return hashlib.sha256(f"{seed}:{doc_key}".encode("utf-8")).hexdigest()


def _plan_filter(plan_entry: dict, spec) -> FilterConfig:
    base = {
        "min_chars": 200,
        "max_chars": 20000,
        "max_replacement_ratio": 0.01,
        "min_letter_ratio": 0.5,
        "require_language": "en",
        "min_language_score": 0.65,
        "min_int_score": 3,
    }
    base.update(spec.filter_overrides or {})
    for key in base:
        if key in plan_entry:
            base[key] = plan_entry[key]
    return FilterConfig(**base)


def _char_budget(train_tokens: int, est_chars_per_token: float,
                 val_permille: int, test_permille: int,
                 margin: float = 1.05) -> int:
    kept_fraction = 1.0 - (val_permille + test_permille) / 1000.0
    return int(train_tokens * est_chars_per_token / kept_fraction * margin)


def load_plan(path: str) -> dict:
    plan = ds.read_json(path)
    if "sources" not in plan or not plan["sources"]:
        raise ValueError("plan needs a non-empty 'sources' list")
    names = [entry["name"] for entry in plan["sources"]]
    if len(names) != len(set(names)):
        raise ValueError("plan has duplicate source names")
    for entry in plan["sources"]:
        if "name" not in entry or "train_tokens" not in entry:
            raise ValueError("each plan source needs name and train_tokens")
        if int(entry["train_tokens"]) <= 0:
            raise ValueError(f"train_tokens must be positive for {entry['name']}")
    return plan


def prepare_multi(plan: dict, root: str, *, resume: bool = False,
                  page_delay: float = 0.0, shard_docs: int = 500,
                  log=print) -> dict:
    """Prepare every source into ``<root>/sources`` and merge the result."""
    val_permille = int(plan.get("val_permille", 10))
    test_permille = int(plan.get("test_permille", 10))
    seed = int(plan.get("seed", 0))
    quotas = {}
    for entry in plan["sources"]:
        name = entry["name"]
        spec = sources.source_spec(name)
        train_tokens = int(entry["train_tokens"])
        est = float(entry.get("est_chars_per_token", 4.0))
        max_docs = int(entry.get("max_docs", 200000))
        budget = _char_budget(train_tokens, est, val_permille, test_permille,
                              float(entry.get("margin", 1.05)))
        source = sources.open_source(name, config=entry.get("config", spec.config),
                                     revision=entry.get("revision"),
                                     backend="datasets", page_delay=page_delay)
        filter_config = _plan_filter(entry, spec)
        source_root = os.path.join(root, "sources", name)
        log(f"prepare {name}: target {train_tokens:,} train tokens, "
            f"char budget {budget:,}, max docs {max_docs:,}")
        manifest = prepare.run_prepare(
            source, source_root, limit_docs=max_docs, limit_chars=budget,
            filter_config=filter_config, val_permille=val_permille,
            test_permille=test_permille, shard_docs=shard_docs,
            resume=resume, max_rows=int(entry.get("max_rows", max_docs * 5)),
            seed=seed, log=log)
        quotas[name] = {
            "train_tokens_target": train_tokens,
            "est_chars_per_token": est,
            "train_chars_budget": budget,
            "documents_available": manifest["counts"]["documents"]["train"],
        }
        log(f"  {name}: {manifest['counts']['documents']} docs, "
            f"digest {manifest['digest'][:16]}")
    blocked = plan.get("blocked") or []
    for entry in blocked:
        log(f"blocked {entry.get('name')}: {entry.get('reason', 'no reason recorded')}")
    return merge_sources(root, quotas, seed=seed, shard_docs=shard_docs,
                         val_permille=val_permille, test_permille=test_permille,
                         log=log)


def _load_entries(root: str, source_names: list[str]):
    entries = []
    manifests = {}
    for name in source_names:
        source_root = os.path.join(root, "sources", name)
        manifest_path = ds.manifest_path(source_root)
        if not os.path.exists(manifest_path):
            raise ValueError(f"missing per-source manifest: {manifest_path}")
        manifest = ds.read_json(manifest_path)
        problems = ds.validate_manifest(manifest)
        if problems:
            raise ValueError(f"invalid per-source manifest for {name}: {problems}")
        manifests[name] = manifest
        corpus = ds.corpus_dir(source_root)
        for split in ds.SPLITS:
            for shard in manifest["documents"][split]:
                path = ds.resolve(corpus, shard["path"])
                for line, record in enumerate(ds.iter_jsonl(path)):
                    text = record["text"]
                    entries.append({
                        "doc_key": record["doc_key"],
                        "source": name,
                        "split": split,
                        "chars": len(text),
                        "exact_hash": ds.sha256_text(text),
                        "simhash": simhash64(text),
                        "shard": shard["path"],
                        "line": line,
                    })
    return entries, manifests


def merge_sources(root: str, quotas: dict, *, seed: int = 0,
                  shard_docs: int = 500, buckets: int = 64,
                  val_permille: int = 10, test_permille: int = 10,
                  log=print) -> dict:
    source_names = sorted(quotas)
    entries, manifests = _load_entries(root, source_names)
    if not entries:
        raise ValueError("no documents to merge")
    first = manifests[source_names[0]]
    if (first["split_policy"]["val_permille"] != val_permille
            or first["split_policy"]["test_permille"] != test_permille):
        raise ValueError("split policy differs between plan and per-source manifest")
    for entry in entries:
        if entry["split"] != split_for(entry["doc_key"], val_permille, test_permille):
            raise ValueError(f"per-source split does not match policy for {entry['doc_key']}")

    result = assign_splits_and_dedup(entries, val_permille=val_permille,
                                     test_permille=test_permille,
                                     near_dup_hamming=3)
    decisions = result["decisions"]

    selected: dict[str, dict] = {}
    quota_stats = {}
    for name in source_names:
        quota = quotas.get(name)
        if quota is None:
            raise ValueError(f"missing quota for source {name}")
        target_chars = int(quota["train_tokens_target"] * quota["est_chars_per_token"])
        candidates = [entry for entry in entries
                      if entry["source"] == name
                      and decisions[entry["doc_key"]]["action"] == "keep"
                      and decisions[entry["doc_key"]]["split"] == "train"]
        candidates.sort(key=lambda entry: _shuffle_key(seed, entry["doc_key"]))
        chosen = []
        chosen_chars = 0
        for entry in candidates:
            if chosen_chars >= target_chars:
                break
            chosen.append(entry["doc_key"])
            chosen_chars += entry["chars"]
        chosen_set = set(chosen)
        for entry in entries:
            if entry["source"] != name:
                continue
            decision = decisions[entry["doc_key"]]
            if decision["action"] != "keep":
                continue
            if decision["split"] == "train":
                if entry["doc_key"] in chosen_set:
                    selected[entry["doc_key"]] = entry
            else:
                selected[entry["doc_key"]] = entry
        quota_stats[name] = {
            **quota,
            "train_documents_selected": len(chosen),
            "train_chars_selected": chosen_chars,
            "estimated_train_tokens": round(chosen_chars / quota["est_chars_per_token"], 1),
            "shortfall_chars": max(0, target_chars - chosen_chars),
            "candidates_train": len(candidates),
        }
        log(f"  quota {name}: {len(chosen)} train docs, "
            f"{chosen_chars:,} chars, est "
            f"{chosen_chars / quota['est_chars_per_token']:,.0f} tokens")

    if not selected:
        raise ValueError("quota selection is empty")

    # bucket shuffle: bounded memory, deterministic order
    work = os.path.join(root, "work", "buckets")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    handles = [open(os.path.join(work, f"bucket-{index:03d}.jsonl"), "w",
                    encoding="utf-8") for index in range(buckets)]
    written_keys = set()
    try:
        for name in source_names:
            corpus = ds.corpus_dir(os.path.join(root, "sources", name))
            manifest = manifests[name]
            for split in ds.SPLITS:
                for shard in manifest["documents"][split]:
                    for record in ds.iter_jsonl(ds.resolve(corpus, shard["path"])):
                        entry = selected.get(record["doc_key"])
                        if entry is None:
                            continue
                        bucket = int(_shuffle_key(seed, record["doc_key"])[:8], 16) % buckets
                        handles[bucket].write(json.dumps({
                            "doc_key": record["doc_key"],
                            "source": name,
                            "text": record["text"],
                            "meta": record.get("meta") or {},
                        }, sort_keys=True, ensure_ascii=False) + "\n")
                        written_keys.add(record["doc_key"])
    finally:
        for handle in handles:
            handle.close()
    if written_keys != set(selected):
        raise ValueError(
            f"bucket pass wrote {len(written_keys)} docs, selected {len(selected)}")

    documents_new = os.path.join(ds.corpus_dir(root), "documents.new")
    if os.path.exists(documents_new):
        shutil.rmtree(documents_new)
    os.makedirs(documents_new)
    buffers = {split: [] for split in ds.SPLITS}
    shard_counts = {split: 0 for split in ds.SPLITS}
    manifest_docs = {split: [] for split in ds.SPLITS}
    chars = {split: 0 for split in ds.SPLITS}
    by_source = {name: {"documents": {split: 0 for split in ds.SPLITS},
                        "chars": {split: 0 for split in ds.SPLITS}}
                 for name in source_names}

    def flush(split: str) -> None:
        records = buffers[split]
        if not records:
            return
        index = shard_counts[split]
        name = f"{split}-{index:05d}.jsonl"
        path = os.path.join(documents_new, split, name)
        entry = ds.write_jsonl_atomic(path, records)
        entry["path"] = f"documents/{split}/{name}"
        manifest_docs[split].append(entry)
        shard_counts[split] = index + 1
        buffers[split] = []

    for bucket in range(buckets):
        path = os.path.join(work, f"bucket-{bucket:03d}.jsonl")
        for record in ds.iter_jsonl(path):
            split = decisions[record["doc_key"]]["split"]
            buffers[split].append(record)
            chars[split] += len(record["text"])
            by_source[record["source"]]["documents"][split] += 1
            by_source[record["source"]]["chars"][split] += len(record["text"])
            if len(buffers[split]) >= shard_docs:
                flush(split)
    for split in ds.SPLITS:
        flush(split)
    shutil.rmtree(work)

    documents_dir = os.path.join(ds.corpus_dir(root), "documents")
    old_dir = os.path.join(ds.corpus_dir(root), "documents.old")
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)
    if os.path.exists(documents_dir):
        os.replace(documents_dir, old_dir)
    os.replace(documents_new, documents_dir)
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)

    source_list = []
    for name in source_names:
        manifest = manifests[name]
        spec = sources.SOURCES.get(name)
        source_list.append({
            **manifest["source"],
            "attribution": getattr(spec, "attribution", ""),
            "provenance": getattr(spec, "provenance", ""),
        })
    counts = {
        "rows_seen": sum(int(manifests[name]["counts"]["rows_seen"]) for name in source_names),
        "accepted": sum(int(manifests[name]["counts"]["accepted"]) for name in source_names),
        "available": len(entries),
        "selected": len(selected),
        "rejected": {key: sum(int(manifests[name]["counts"]["rejected"].get(key, 0))
                              for name in source_names)
                     for key in {k for name in source_names
                                 for k in manifests[name]["counts"]["rejected"]}},
        "exact_duplicates": int(result["stats"]["exact_duplicates"]),
        "near_duplicates": int(result["stats"]["near_duplicates"]),
        "within_source_exact_duplicates": sum(
            int(manifests[name]["counts"]["exact_duplicates"]) for name in source_names),
        "within_source_near_duplicates": sum(
            int(manifests[name]["counts"]["near_duplicates"]) for name in source_names),
        "documents": {split: sum(by_source[name]["documents"][split] for name in source_names)
                      for split in ds.SPLITS},
        "chars": chars,
        "by_source": by_source,
    }
    manifest = ds.build_manifest(
        source=source_list[0],
        sources=source_list,
        seed=seed,
        split_policy=prepare.split_policy(val_permille, test_permille),
        filters={"by_source": {name: manifests[name]["filters"] for name in source_names}},
        dedup={**prepare.dedup_policy(3),
               "order": "global exact then global near, then split assignment, "
                        "then per-source quota fill on train"},
        counts=counts,
        documents=manifest_docs,
        quotas=quota_stats,
        shuffle={"seed": seed, "buckets": buckets,
                 "method": "bucket = int(sha256(seed:doc_key)[:8],16) % buckets; "
                           "buckets written in order, records in encounter order"},
        tokenization=None,
    )
    problems = ds.validate_manifest(manifest)
    if problems:
        raise ValueError(f"merged manifest is invalid: {problems}")
    ds.atomic_write_json(ds.manifest_path(root), manifest)
    log(f"merged {len(selected)} documents: " + ", ".join(
        f"{split}={counts['documents'][split]}" for split in ds.SPLITS))
    return manifest


def report_multi(root: str, out: str | None = None) -> dict:
    """Per-source counts and held-out fertility for the merged corpus."""
    import sentencepiece as spm

    manifest = ds.read_json(ds.manifest_path(root))
    tokenization = manifest.get("tokenization")
    if not tokenization:
        raise ValueError("corpus is not tokenized")
    corpus = ds.corpus_dir(root)
    processor = spm.SentencePieceProcessor(
        model_file=ds.resolve(corpus, tokenization["tokenizer"]["model"]))
    heldout = {}
    for split in ("validation", "test"):
        for shard in manifest["documents"][split]:
            for record in ds.iter_jsonl(ds.resolve(corpus, shard["path"])):
                source = record.get("source", manifest["source"]["name"])
                bucket = heldout.setdefault(source, {"docs": 0, "chars": 0,
                                                     "words": 0, "tokens": 0})
                pieces = processor.encode(record["text"], out_type=int)
                bucket["docs"] += 1
                bucket["chars"] += len(record["text"])
                bucket["words"] += len(record["text"].split())
                bucket["tokens"] += len(pieces)
    by_source = tokenization.get("by_source", {})
    report = {"digest": manifest["digest"], "tokenizer": tokenization["tokenizer"],
              "total_train_tokens": tokenization["splits"]["train"]["tokens"],
              "sources": {}}
    for name in manifest.get("sources", [manifest["source"]]):
        source = name["name"]
        tokens = by_source.get(source, {})
        train_tokens = int(tokens.get("train", 0)) if isinstance(tokens, dict) else int(tokens or 0)
        held = heldout.get(source, {"docs": 0, "chars": 0, "words": 0, "tokens": 0})
        report["sources"][source] = {
            "train_tokens": train_tokens,
            "heldout": held,
            "chars_per_token_heldout": round(held["chars"] / held["tokens"], 4)
            if held["tokens"] else None,
            "fertility_tokens_per_word": round(held["tokens"] / held["words"], 4)
            if held["words"] else None,
            "quota": (manifest.get("quotas") or {}).get(source),
        }
    report["totals"] = {
        "train_tokens": int(tokenization["splits"]["train"]["tokens"]),
        "validation_tokens": int(tokenization["splits"]["validation"]["tokens"]),
        "test_tokens": int(tokenization["splits"]["test"]["tokens"]),
        "documents": manifest["counts"]["documents"],
        "chars": manifest["counts"]["chars"],
    }
    if out:
        ds.atomic_write_json(out, report)
    return report
