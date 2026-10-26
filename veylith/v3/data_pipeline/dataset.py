"""Document shards, manifests, atomic writes and integrity verification.

Layout under a corpus root (default ``veylith/v3/data``):

    staging/state.json          resumable source pass
    staging/index.jsonl         one line per accepted document (hashes, counts)
    staging/shard-00000.jsonl   normalized documents, arrival order
    corpus/manifest.json        the single commit point
    corpus/documents/<split>/   final document shards (JSONL)
    corpus/tokenizer/           BPE model and its training text
    corpus/tokens/<split>/      uint16 token shards plus per-doc metadata

Every write is temp-file-plus-rename. The manifest carries a digest over all
fields; ``verify_corpus`` recomputes every hash it can.
"""

from __future__ import annotations

import hashlib
import json
import os

from . import FILTER_VERSION, FORMAT_VERSION, PIPELINE_VERSION

SPLITS = ("train", "validation", "test")
DIGEST_LEN = 64


# ------------------------------------------------------------------- basics

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def manifest_digest(manifest: dict) -> str:
    body = {key: value for key, value in manifest.items() if key != "digest"}
    return sha256_bytes(canonical_json(body))


# ------------------------------------------------------------ atomic writes

def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str, data: bytes) -> str:
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = f"{target}.tmp.{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    _fsync_dir(os.path.dirname(target))
    return target


def atomic_write_text(path: str, text: str) -> str:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str, payload) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return atomic_write_text(path, data)


def read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_atomic(path: str, records) -> dict:
    """Write JSON lines atomically; return path, sha256, bytes, docs, chars."""
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = f"{target}.tmp.{os.getpid()}"
    digest = hashlib.sha256()
    size = docs = chars = 0
    with open(tmp, "wb") as handle:
        for record in records:
            line = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            size += len(line)
            docs += 1
            chars += len(record.get("text") or "")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    _fsync_dir(os.path.dirname(target))
    return {"path": target, "sha256": digest.hexdigest(), "bytes": size,
            "docs": docs, "chars": chars}


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


# ------------------------------------------------------------------ layout

def corpus_dir(root: str) -> str:
    return os.path.join(root, "corpus")


def staging_dir(root: str) -> str:
    return os.path.join(root, "staging")


def manifest_path(root: str) -> str:
    return os.path.join(corpus_dir(root), "manifest.json")


def document_shard_relpath(split: str, index: int) -> str:
    return f"documents/{split}/{split}-{index:05d}.jsonl"


def token_shard_relpath(split: str, index: int) -> str:
    return f"tokens/{split}/{split}-{index:05d}.tokens.npy"


def token_meta_relpath(split: str, index: int) -> str:
    return f"tokens/{split}/{split}-{index:05d}.meta.json"


def relpath_to_root(root: str, path: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace(os.sep, "/")


def resolve(root: str, relative: str) -> str:
    return os.path.join(root, relative.replace("/", os.sep))


# ---------------------------------------------------------------- manifest

def build_manifest(*, source: dict, seed: int, split_policy: dict, filters: dict,
                   dedup: dict, counts: dict, documents: dict,
                   tokenization=None) -> dict:
    manifest = {
        "format_version": FORMAT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "filter_version": FILTER_VERSION,
        "source": source,
        "seed": int(seed),
        "split_policy": split_policy,
        "filters": filters,
        "dedup": dedup,
        "counts": counts,
        "documents": documents,
        "tokenization": tokenization,
    }
    manifest["digest"] = manifest_digest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> list[str]:
    """Return a list of problems; empty means structurally valid."""
    problems: list[str] = []
    required = ("format_version", "pipeline_version", "filter_version", "source",
                "seed", "split_policy", "filters", "dedup", "counts",
                "documents", "tokenization", "digest")
    for key in required:
        if key not in manifest:
            problems.append(f"missing manifest field: {key}")
    if problems:
        return problems
    if manifest["format_version"] != FORMAT_VERSION:
        problems.append(f"format_version {manifest['format_version']} != {FORMAT_VERSION}")
    if manifest.get("filter_version") != FILTER_VERSION:
        problems.append(f"filter_version {manifest.get('filter_version')} != {FILTER_VERSION}")
    digest = manifest.get("digest")
    if not isinstance(digest, str) or len(digest) != DIGEST_LEN:
        problems.append("digest is not a sha256 hex string")
    elif manifest_digest(manifest) != digest:
        problems.append("manifest digest mismatch")

    source = manifest["source"]
    for key in ("name", "dataset_id", "config", "split", "revision", "license", "access"):
        if key not in source:
            problems.append(f"missing source.{key}")
    for key in ("val_permille", "test_permille", "assignment"):
        if key not in manifest["split_policy"]:
            problems.append(f"missing split_policy.{key}")

    counts = manifest["counts"]
    for key in ("rows_seen", "accepted", "exact_duplicates", "near_duplicates",
                "documents", "chars"):
        if key not in counts:
            problems.append(f"missing counts.{key}")

    documents = manifest["documents"]
    for split in SPLITS:
        if split not in documents:
            problems.append(f"missing documents.{split}")
            continue
        total_docs = 0
        total_chars = 0
        for shard in documents[split]:
            for key in ("path", "sha256", "bytes", "docs", "chars"):
                if key not in shard:
                    problems.append(f"documents.{split} shard missing {key}")
            if not str(shard.get("path", "")).startswith(f"documents/{split}/"):
                problems.append(f"documents.{split} shard path outside its split")
            if len(str(shard.get("sha256", ""))) != DIGEST_LEN:
                problems.append(f"documents.{split} shard sha256 malformed")
            total_docs += int(shard.get("docs", 0))
            total_chars += int(shard.get("chars", 0))
        if isinstance(counts.get("documents"), dict):
            if int(counts["documents"].get(split, -1)) != total_docs:
                problems.append(f"counts.documents.{split} does not match shards")
        if isinstance(counts.get("chars"), dict):
            if int(counts["chars"].get(split, -1)) != total_chars:
                problems.append(f"counts.chars.{split} does not match shards")

    tokenization = manifest["tokenization"]
    if tokenization is not None:
        for key in ("vocab_size", "special_ids", "tokenizer", "splits", "total_tokens"):
            if key not in tokenization:
                problems.append(f"missing tokenization.{key}")
        tokenizer = tokenization.get("tokenizer", {})
        for key in ("model", "sha256", "vocab", "vocab_sha256", "train_text_sha256"):
            if key not in tokenizer:
                problems.append(f"missing tokenization.tokenizer.{key}")
        for split in SPLITS:
            entry = (tokenization.get("splits") or {}).get(split)
            if not isinstance(entry, dict):
                problems.append(f"missing tokenization.splits.{split}")
    return problems


# --------------------------------------------------------------- verification

def verify_corpus(root: str, deep: bool = False) -> dict:
    """Check manifests, shard hashes, split isolation and tokenizer links."""
    from .prepare import split_for

    result = {"ok": False, "problems": [], "counts": {}, "digest": None,
              "deep": bool(deep)}
    path = manifest_path(root)
    if not os.path.exists(path):
        result["problems"].append(f"manifest missing: {path}")
        return result
    manifest = read_json(path)
    result["digest"] = manifest.get("digest")
    result["problems"].extend(validate_manifest(manifest))
    if result["problems"]:
        return result
    result["counts"] = manifest["counts"]

    corpus = corpus_dir(root)
    seen_keys: dict[str, str] = {}
    doc_by_key: dict[str, str] = {}
    doc_shards: dict[str, list[dict]] = {}

    for split in SPLITS:
        shards = manifest["documents"][split]
        doc_shards[split] = []
        for shard in shards:
            full = resolve(corpus, shard["path"])
            if not os.path.exists(full):
                result["problems"].append(f"missing shard: {shard['path']}")
                continue
            if os.path.getsize(full) != int(shard["bytes"]):
                result["problems"].append(f"size mismatch: {shard['path']}")
            if sha256_file(full) != shard["sha256"]:
                result["problems"].append(f"sha256 mismatch: {shard['path']}")
                continue
            records = list(iter_jsonl(full))
            if len(records) != int(shard["docs"]):
                result["problems"].append(f"doc count mismatch: {shard['path']}")
            chars = sum(len(record.get("text") or "") for record in records)
            if chars != int(shard["chars"]):
                result["problems"].append(f"char count mismatch: {shard['path']}")
            doc_shards[split].append({"entry": shard, "records": records})
            for record in records:
                key = record.get("doc_key")
                text = record.get("text")
                if not isinstance(key, str) or not isinstance(text, str):
                    result["problems"].append(f"malformed record in {shard['path']}")
                    continue
                if key in seen_keys:
                    result["problems"].append(
                        f"duplicate doc_key across splits: {key} in "
                        f"{seen_keys[key]} and {split}")
                seen_keys[key] = split
                doc_by_key[key] = sha256_text(text)
                expected = split_for(key, manifest["split_policy"]["val_permille"],
                                     manifest["split_policy"]["test_permille"])
                if expected != split:
                    result["problems"].append(
                        f"doc {key} is in {split}, split policy says {expected}")

    tokenization = manifest["tokenization"]
    if tokenization is not None:
        tokenizer = tokenization["tokenizer"]
        for key in ("model", "vocab"):
            full = resolve(corpus, tokenizer[key])
            if not os.path.exists(full):
                result["problems"].append(f"missing tokenizer file: {tokenizer[key]}")
        if os.path.exists(resolve(corpus, tokenizer["model"])):
            if sha256_file(resolve(corpus, tokenizer["model"])) != tokenizer["sha256"]:
                result["problems"].append("tokenizer model sha256 mismatch")
        if os.path.exists(resolve(corpus, tokenizer["vocab"])):
            if sha256_file(resolve(corpus, tokenizer["vocab"])) != tokenizer["vocab_sha256"]:
                result["problems"].append("tokenizer vocab sha256 mismatch")
        processor = None
        if deep:
            import sentencepiece as spm
            model_file = resolve(corpus, tokenizer["model"])
            if os.path.exists(model_file):
                processor = spm.SentencePieceProcessor(model_file=model_file)
        eos_id = tokenization["special_ids"]["eos"]
        for split in SPLITS:
            shards = tokenization["splits"][split].get("shards", [])
            total_tokens = 0
            for shard in shards:
                data_path = resolve(corpus, shard["path"])
                meta_path = resolve(corpus, shard["meta"])
                if not os.path.exists(data_path) or not os.path.exists(meta_path):
                    result["problems"].append(f"missing token shard: {shard['path']}")
                    continue
                if sha256_file(data_path) != shard["sha256"]:
                    result["problems"].append(f"token sha256 mismatch: {shard['path']}")
                meta = read_json(meta_path)
                total_tokens += int(meta["tokens"])
                docs = meta.get("docs", [])
                if len(docs) != int(shard["docs"]):
                    result["problems"].append(f"token doc count mismatch: {shard['path']}")
                import numpy as np
                tokens = None
                if deep:
                    tokens = np.load(data_path)
                for doc in docs:
                    key = doc["doc_key"]
                    if key not in doc_by_key:
                        result["problems"].append(
                            f"token shard references unknown doc: {key}")
                        continue
                    start, end = int(doc["start"]), int(doc["end"])
                    if not 0 <= start < end <= int(meta["tokens"]):
                        result["problems"].append(f"bad offsets for {key}")
                        continue
                    if doc["text_sha256"] != doc_by_key[key]:
                        result["problems"].append(f"text hash mismatch for {key}")
                    if tokens is not None:
                        if int(tokens[end - 1]) != eos_id:
                            result["problems"].append(f"missing eos for {key}")
                        text = processor.decode(tokens[start:end - 1].tolist())
                        if sha256_text(text) != doc["text_sha256"]:
                            result["problems"].append(f"decode mismatch for {key}")
            expected_tokens = int(tokenization["splits"][split].get("tokens", -1))
            if total_tokens != expected_tokens:
                result["problems"].append(
                    f"token count mismatch for {split}: {total_tokens} != {expected_tokens}")

    result["ok"] = not result["problems"]
    return result
