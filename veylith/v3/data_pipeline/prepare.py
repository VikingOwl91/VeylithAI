"""Normalization, filtering, split assignment and deduplication.

Determinism rules:

* the split of a kept document is a pure function of its ``doc_key``
* the kept document of a duplicate cluster is the one with the smallest
  ``doc_key``, independent of stream order
* filtering is stateless per document

The prepare step is a bounded streaming pass into staging shards, followed by
a deterministic finalize pass over the local staging files. Memory use is
proportional to the index metadata, not to the text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass

from . import FILTER_VERSION, PIPELINE_VERSION
from . import dataset as ds
from .sources import SourceError

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
BLANK_RE = re.compile(r"\n{3,}")
SHINGLES = 5
BANDS = 4
BAND_BITS = 16


@dataclass(frozen=True)
class FilterConfig:
    min_chars: int = 200
    max_chars: int = 20000
    max_replacement_ratio: float = 0.01
    min_letter_ratio: float = 0.5
    require_language: str | None = "en"
    min_language_score: float = 0.65
    min_int_score: int | None = 3


def normalize_text(text: str) -> str:
    """Lossless-for-content normalization, deterministic for the same input."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RE.sub("\n\n", text)
    return text.strip()


def validate_text(text: str, cfg: FilterConfig, meta: dict | None = None) -> tuple[bool, str]:
    meta = meta or {}
    if not text:
        return False, "empty"
    if len(text) < cfg.min_chars:
        return False, "too_short"
    if len(text) > cfg.max_chars:
        return False, "too_long"
    if text.count("\ufffd") / len(text) > cfg.max_replacement_ratio:
        return False, "replacement_chars"
    letters = sum(1 for ch in text if ch.isalpha())
    if letters / len(text) < cfg.min_letter_ratio:
        return False, "letter_ratio"
    if cfg.require_language is not None:
        language = meta.get("language")
        if language is not None and language != cfg.require_language:
            return False, "language"
    score = meta.get("language_score")
    if score is not None and float(score) < cfg.min_language_score:
        return False, "language_score"
    int_score = meta.get("int_score")
    if cfg.min_int_score is not None and int_score is not None:
        if int(int_score) < cfg.min_int_score:
            return False, "int_score"
    if not WORD_RE.search(text):
        return False, "no_words"
    return True, ""


def simhash64(text: str, shingles: int = SHINGLES) -> int:
    """64-bit SimHash over lower-cased word shingles."""
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0
    if len(words) < shingles:
        grams = [" ".join(words)]
    else:
        grams = [" ".join(words[i:i + shingles])
                 for i in range(len(words) - shingles + 1)]
    vector = [0] * 64
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit in range(64):
        if vector[bit] > 0:
            result |= 1 << bit
    return result


def hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def split_for(doc_key: str, val_permille: int = 5, test_permille: int = 5) -> str:
    """Pure split assignment: test first, then validation, then train."""
    value = int(hashlib.sha256(doc_key.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if value < test_permille:
        return "test"
    if value < test_permille + val_permille:
        return "validation"
    return "train"


def split_policy(val_permille: int, test_permille: int) -> dict:
    return {
        "val_permille": int(val_permille),
        "test_permille": int(test_permille),
        "assignment": "int(sha256(doc_key)[:8],16) % 1000; "
                      "test < test_permille, validation < test+val, else train",
    }


def dedup_policy(near_dup_hamming: int, max_bucket: int = 4096) -> dict:
    return {
        "exact": "sha256(normalized_text)",
        "near": f"simhash64 over word {SHINGLES}-grams, {BANDS} bands x "
                f"{BAND_BITS} bits, hamming <= {near_dup_hamming}",
        "canonical": "min doc_key of the cluster",
        "max_bucket": int(max_bucket),
        "horizon": "all documents staged in this run",
    }


class _UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if a < b:
            self.parent[b] = a
        else:
            self.parent[a] = b


def assign_splits_and_dedup(entries: list[dict], *, val_permille: int = 5,
                            test_permille: int = 5, near_dup_hamming: int = 3,
                            max_bucket: int = 4096) -> dict:
    """Deterministic decisions for every staged document.

    ``entries`` need ``doc_key``, ``exact_hash`` and ``simhash``. The result
    maps every key to ``keep``/``exact_duplicate``/``near_duplicate`` with the
    canonical key and, for kept documents, the split.
    """
    by_key = {entry["doc_key"]: entry for entry in entries}
    if len(by_key) != len(entries):
        raise ValueError("duplicate doc_key in the staging index")

    exact_groups: dict[str, list[str]] = {}
    for key in sorted(by_key):
        exact_groups.setdefault(by_key[key]["exact_hash"], []).append(key)
    exact_canon = {}
    for group in exact_groups.values():
        canonical = min(group)
        for key in group:
            exact_canon[key] = canonical
    exact_kept = sorted(set(exact_canon.values()))

    union = _UnionFind(exact_kept)
    buckets: dict[tuple[int, int], list[str]] = {}
    for key in exact_kept:
        value = int(by_key[key]["simhash"])
        for band in range(BANDS):
            signature = (value >> (BAND_BITS * band)) & ((1 << BAND_BITS) - 1)
            buckets.setdefault((band, signature), []).append(key)
    skipped_buckets = 0
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        if len(bucket) > max_bucket:
            skipped_buckets += 1
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                left, right = bucket[i], bucket[j]
                if hamming64(int(by_key[left]["simhash"]),
                             int(by_key[right]["simhash"])) <= near_dup_hamming:
                    union.union(left, right)

    clusters: dict[str, list[str]] = {}
    for key in exact_kept:
        clusters.setdefault(union.find(key), []).append(key)
    near_canon = {}
    for group in clusters.values():
        canonical = min(group)
        for key in group:
            near_canon[key] = canonical

    decisions = {}
    for key in sorted(by_key):
        canonical = exact_canon[key]
        if canonical != key:
            decisions[key] = {"action": "exact_duplicate", "canonical": canonical}
            continue
        near = near_canon[key]
        if near != key:
            decisions[key] = {"action": "near_duplicate", "canonical": near}
            continue
        decisions[key] = {"action": "keep", "canonical": key,
                          "split": split_for(key, val_permille, test_permille)}

    stats = {
        "kept": len(clusters),
        "clusters": len(clusters),
        "exact_duplicates": len(entries) - len(exact_kept),
        "near_duplicates": len(exact_kept) - len(clusters),
        "skipped_buckets": skipped_buckets,
    }
    return {"decisions": decisions, "stats": stats}


# --------------------------------------------------------------- run

def _state_identity(source) -> dict:
    spec = source.spec
    license_name = spec.license_name
    if hasattr(source, "info"):
        info = source.info()
        license_name = info.get("license") or license_name
    return {
        "name": source.name,
        "dataset_id": spec.dataset_id,
        "config": getattr(source, "config", spec.config),
        "split": spec.split,
        "revision": source.resolved_revision()
        if hasattr(source, "resolved_revision") else source.revision,
        "backend": source.backend_name() if hasattr(source, "backend_name") else "local",
        "license": license_name,
        "license_url": spec.license_url,
        "access": spec.access,
    }


def _load_index(staging: str) -> list[dict]:
    entries = []
    for name in sorted(os.listdir(staging)):
        if name.startswith("index-") and name.endswith(".jsonl"):
            entries.extend(ds.iter_jsonl(os.path.join(staging, name)))
    return entries


def _load_state(path: str):
    return ds.read_json(path) if os.path.exists(path) else None


def run_prepare(source, root: str, *, limit_docs: int,
                filter_config: FilterConfig | None = None,
                val_permille: int = 5, test_permille: int = 5,
                shard_docs: int = 1000, resume: bool = False,
                max_rows: int | None = None, near_dup_hamming: int = 3,
                seed: int = 0, finalize: bool = True, log=print) -> dict:
    """Stream, stage and finalize a corpus.

    ``finalize=False`` stops after staging and is used by resume tests; the
    CLI always finalizes. Resume continues from the last flushed staging
    shard and skips documents already present in the index.
    """
    if limit_docs <= 0:
        raise ValueError("limit_docs must be positive")
    if shard_docs <= 0:
        raise ValueError("shard_docs must be positive")
    cfg = filter_config or FilterConfig()
    staging = ds.staging_dir(root)
    corpus = ds.corpus_dir(root)
    state_path = os.path.join(staging, "state.json")

    if not resume:
        if os.path.exists(root):
            with os.scandir(root) as scan:
                if any(scan):
                    raise ValueError(f"{root} is not empty; use --resume or a new root")
    identity = _state_identity(source)
    revision = identity["revision"]

    os.makedirs(staging, exist_ok=True)
    state = _load_state(state_path) if resume else None
    if resume and state is None:
        raise ValueError(f"no staging state at {state_path}; cannot resume")
    if state is not None:
        for key in ("name", "dataset_id", "config", "split", "revision", "backend"):
            if state.get("source", {}).get(key) != identity[key]:
                raise ValueError(
                    f"resume source mismatch for {key}: "
                    f"{state.get('source', {}).get(key)!r} != {identity[key]!r}")
        if state.get("filter_version") != FILTER_VERSION:
            raise ValueError("resume filter version mismatch")
        if state.get("filters") != {**asdict(cfg), "filter_version": FILTER_VERSION}:
            raise ValueError("resume filter configuration mismatch")
        index_entries = _load_index(staging)
        staged_keys = {entry["doc_key"] for entry in index_entries}
        staged = int(state["staged"])
        shard_index = int(state["shard_index"])
        next_offset = int(state["next_offset"])
        rejected = dict(state.get("rejected") or {})
        rows_seen = int(state.get("rows_seen") or 0)
        if staged != len(index_entries):
            raise ValueError("staging index does not match the state count")
    else:
        staged_keys = set()
        staged = 0
        shard_index = 0
        next_offset = 0
        rejected = {}
        rows_seen = 0
        index_entries = []
        state = {
            "pipeline_version": PIPELINE_VERSION,
            "filter_version": FILTER_VERSION,
            "source": identity,
            "filters": {**asdict(cfg), "filter_version": FILTER_VERSION},
            "next_offset": 0,
            "staged": 0,
            "shard_index": 0,
            "rejected": {},
            "rows_seen": 0,
            "backend": identity.get("backend"),
        }
        ds.atomic_write_json(state_path, state)

    buffer_records: list[dict] = []
    buffer_index: list[dict] = []
    buffer_start = next_offset
    stop = staged >= limit_docs
    max_rows = max_rows if max_rows is not None else max(1000, (limit_docs - staged) * 10 + 100)

    def flush(offset_after: int) -> None:
        nonlocal shard_index, staged, next_offset
        wrote = bool(buffer_records)
        shard_name = None
        if wrote:
            shard_name = f"shard-{shard_index:05d}.jsonl"
            index_name = f"index-{shard_index:05d}.jsonl"
            ds.write_jsonl_atomic(os.path.join(staging, shard_name), buffer_records)
            ds.write_jsonl_atomic(os.path.join(staging, index_name), buffer_index)
            staged += len(buffer_records)
            shard_index += 1
            buffer_records.clear()
            buffer_index.clear()
        next_offset = max(next_offset, offset_after)
        state.update({"next_offset": next_offset, "staged": staged,
                      "shard_index": shard_index, "rejected": dict(rejected),
                      "rows_seen": rows_seen})
        ds.atomic_write_json(state_path, state)
        if wrote:
            log(f"  staged {staged} docs ({shard_name}), rows seen {rows_seen}")

    if not stop:
        last_processed = next_offset - 1
        for doc in source.iter_documents(offset=next_offset, max_rows=max_rows):
            last_processed = int(doc.arrival_index)
            rows_seen += 1
            text = normalize_text(doc.text)
            ok, reason = validate_text(text, cfg, doc.meta)
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            if doc.doc_key in staged_keys:
                continue
            entry = {
                "doc_key": doc.doc_key,
                "arrival": int(doc.arrival_index),
                "exact_hash": ds.sha256_text(text),
                "simhash": simhash64(text),
                "chars": len(text),
                "shard": f"shard-{shard_index:05d}.jsonl",
                "line": len(buffer_records),
            }
            buffer_records.append({"doc_key": doc.doc_key, "text": text,
                                   "meta": doc.meta})
            buffer_index.append(entry)
            staged_keys.add(doc.doc_key)
            if staged + len(buffer_records) >= limit_docs:
                flush(int(doc.arrival_index) + 1)
                break
            if len(buffer_records) >= shard_docs:
                flush(int(doc.arrival_index) + 1)
        flush(last_processed + 1)
    else:
        flush(next_offset)

    if not finalize:
        return {"staged": staged, "state_path": state_path,
                "shard_index": shard_index, "rejected": rejected,
                "rows_seen": rows_seen}

    return finalize_corpus(source, root, filter_config=cfg,
                           val_permille=val_permille,
                           test_permille=test_permille, shard_docs=shard_docs,
                           near_dup_hamming=near_dup_hamming, seed=seed, log=log)


def finalize_corpus(source, root: str, *, filter_config: FilterConfig,
                    val_permille: int, test_permille: int, shard_docs: int,
                    near_dup_hamming: int, seed: int = 0, log=print) -> dict:
    staging = ds.staging_dir(root)
    corpus = ds.corpus_dir(root)
    state = _load_state(os.path.join(staging, "state.json"))
    if state is None:
        raise ValueError("no staging state; run prepare first")
    entries = _load_index(staging)
    if not entries:
        raise ValueError("staging is empty; nothing to finalize")

    result = assign_splits_and_dedup(entries, val_permille=val_permille,
                                     test_permille=test_permille,
                                     near_dup_hamming=near_dup_hamming)
    decisions = result["decisions"]

    documents_new = os.path.join(corpus, "documents.new")
    if os.path.exists(documents_new):
        shutil.rmtree(documents_new)
    os.makedirs(documents_new)
    shard_buffers: dict[str, list[dict]] = {split: [] for split in ds.SPLITS}
    shard_counts: dict[str, int] = {split: 0 for split in ds.SPLITS}
    manifest_docs: dict[str, list[dict]] = {split: [] for split in ds.SPLITS}
    chars: dict[str, int] = {split: 0 for split in ds.SPLITS}
    kept = 0

    shard_files = sorted(name for name in os.listdir(staging)
                         if name.startswith("shard-") and name.endswith(".jsonl"))
    index_files = sorted(name for name in os.listdir(staging)
                         if name.startswith("index-") and name.endswith(".jsonl"))
    if len(shard_files) != len(index_files):
        raise ValueError("staging shard and index files do not pair up")

    for shard_name, index_name in zip(shard_files, index_files):
        index_entries = ds.iter_jsonl(os.path.join(staging, index_name))
        records = ds.iter_jsonl(os.path.join(staging, shard_name))
        for entry, record in zip(index_entries, records):
            if entry["doc_key"] != record["doc_key"]:
                raise ValueError("staging shard does not match its index")
            decision = decisions[entry["doc_key"]]
            if decision["action"] != "keep":
                continue
            split = decision["split"]
            shard_buffers[split].append(record)
            kept += 1
            chars[split] += len(record["text"])
            if len(shard_buffers[split]) >= shard_docs:
                _flush_documents(documents_new, split, shard_counts, shard_buffers,
                                 manifest_docs, corpus)
        # drain the remaining buffer of every split at the end of each staging shard
        # only when it is full enough; keep small buffers across staging shards
    for split in ds.SPLITS:
        _flush_documents(documents_new, split, shard_counts, shard_buffers,
                         manifest_docs, corpus)

    documents_dir = os.path.join(corpus, "documents")
    old_dir = os.path.join(corpus, "documents.old")
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)
    if os.path.exists(documents_dir):
        os.replace(documents_dir, old_dir)
    os.replace(documents_new, documents_dir)
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)

    counts = {
        "rows_seen": int(state.get("rows_seen", 0)),
        "accepted": len(entries),
        "rejected": dict(state.get("rejected") or {}),
        "exact_duplicates": int(result["stats"]["exact_duplicates"]),
        "near_duplicates": int(result["stats"]["near_duplicates"]),
        "documents": {split: sum(shard["docs"] for shard in manifest_docs[split])
                      for split in ds.SPLITS},
        "chars": chars,
    }

    manifest = ds.build_manifest(
        source=state["source"],
        seed=int(seed),
        split_policy=split_policy(val_permille, test_permille),
        filters={**asdict(filter_config), "filter_version": FILTER_VERSION},
        dedup=dedup_policy(near_dup_hamming),
        counts=counts,
        documents=manifest_docs,
        tokenization=None,
    )
    problems = ds.validate_manifest(manifest)
    if problems:
        raise ValueError(f"manifest is invalid: {problems}")
    ds.atomic_write_json(ds.manifest_path(root), manifest)
    log(f"  finalized {kept} documents: " + ", ".join(
        f"{split}={counts['documents'][split]}" for split in ds.SPLITS))
    return manifest


def _flush_documents(documents_new: str, split: str, shard_counts: dict,
                     buffers: dict, manifest_docs: dict, corpus: str) -> None:
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
