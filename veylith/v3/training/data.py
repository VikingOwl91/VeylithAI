"""Data access for training and evaluation on the frozen v3 token shards."""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from data_pipeline import dataset as ds
from data_pipeline import loader


def load_manifest(root: str) -> dict:
    return ds.read_json(ds.manifest_path(root))


def source_names(manifest: dict) -> list[str]:
    if manifest.get("sources"):
        return [item["name"] for item in manifest["sources"]]
    return [manifest["source"]["name"]]


def tokenizer_info(manifest: dict) -> dict:
    tokenization = manifest.get("tokenization")
    if not tokenization:
        raise ValueError("corpus is not tokenized")
    return tokenization["tokenizer"]


def _document_chars(root: str, manifest: dict) -> dict[str, int]:
    corpus = ds.corpus_dir(root)
    chars = {}
    for split in ds.SPLITS:
        for shard in manifest["documents"][split]:
            for record in ds.iter_jsonl(ds.resolve(corpus, shard["path"])):
                chars[record["doc_key"]] = len(record["text"])
    return chars


def iter_source_batches(root: str, split: str, source: str | None,
                        seq_len: int, batch_size: int,
                        max_batches: int | None = None):
    """Yield source-pure batches with per-sequence character counts.

    Documents are concatenated within a source (EOS already terminates each
    document); a sequence never mixes two sources. The character count of a
    sequence is the proportional share of its documents. That is the legacy
    estimate used only when exact prefix-decode characters are disabled;
    ``evaluate`` defaults to the exact tokenizer-based span. Validation/test
    data is only read here.
    """
    manifest = load_manifest(root)
    tokenization = manifest.get("tokenization")
    if not tokenization:
        raise ValueError("corpus is not tokenized")
    pad_id = int(tokenization["special_ids"]["pad"])
    need_chars = False
    for shard in tokenization["splits"][split]["shards"]:
        meta = ds.read_json(ds.resolve(ds.corpus_dir(root), shard["meta"]))
        if any("chars" not in doc for doc in meta["docs"]):
            need_chars = True
            break
    doc_chars = _document_chars(root, manifest) if need_chars else {}

    buffer_ids: list[int] = []
    buffer_chars = 0.0
    pending: list[tuple[list[int], float]] = []
    emitted = 0
    corpus = ds.corpus_dir(root)
    for shard in tokenization["splits"][split]["shards"]:
        tokens = np.load(ds.resolve(corpus, shard["path"]))
        meta = ds.read_json(ds.resolve(corpus, shard["meta"]))
        for doc in meta["docs"]:
            if source is not None and doc.get("source") != source:
                continue
            chars = doc.get("chars")
            if chars is None:
                chars = doc_chars[doc["doc_key"]]
            ids = tokens[int(doc["start"]):int(doc["end"])].tolist()
            total = len(ids)
            position = 0
            while position < total:
                take = min(total - position, seq_len - len(buffer_ids))
                buffer_ids.extend(ids[position:position + take])
                buffer_chars += float(chars) * take / total
                position += take
                if len(buffer_ids) == seq_len:
                    pending.append((buffer_ids, buffer_chars))
                    buffer_ids = []
                    buffer_chars = 0.0
                    if len(pending) >= batch_size:
                        yield _batch(pending, pad_id, seq_len)
                        pending = []
                        emitted += 1
                        if max_batches is not None and emitted >= max_batches:
                            return
    if buffer_ids:
        pending.append((buffer_ids, buffer_chars))
    if pending and (max_batches is None or emitted < max_batches):
        yield _batch(pending, pad_id, seq_len)


def _batch(pending: list[tuple[list[int], float]], pad_id: int,
           seq_len: int) -> dict:
    rows = len(pending)
    array = np.full((rows, seq_len), pad_id, dtype=np.uint16)
    lengths = []
    chars = []
    for row, (ids, char_count) in enumerate(pending):
        if len(ids) > seq_len:
            raise ValueError("sequence longer than seq_len")
        array[row, :len(ids)] = np.asarray(ids, dtype=np.uint16)
        lengths.append(len(ids))
        chars.append(char_count)
    return {"tokens": array, "lengths": lengths,
            "chars": np.asarray(chars, dtype=np.float32)}


def iter_train_batches(root: str, cfg, start_epoch: int, start_batch: int,
                       epochs: int):
    """Loader-backed training batches plus the epoch length for resume math."""
    plan = loader.plan_sequences(root, split="train", seq_len=cfg.seq_len,
                                 mode="concat")
    batches_per_epoch = math.ceil(len(plan["sequences"]) / cfg.batch_size)
    batches = loader.iter_batches(
        plan, root, batch_size=cfg.batch_size, epochs=epochs, seed=cfg.seed,
        start_epoch=start_epoch, start_batch=start_batch)
    return plan, batches_per_epoch, batches


def to_tensors(batch: dict, device: str):
    tokens = torch.from_numpy(np.asarray(batch["tokens"], dtype=np.int64)).to(device)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    lengths = torch.tensor(batch["lengths"], device=device)
    positions = torch.arange(x.size(1), device=device)[None, :]
    valid = positions < (lengths[:, None] - 1)
    return x, y, valid


def source_tensors(batch: dict, device: str):
    x, y, valid = to_tensors(batch, device)
    chars = torch.tensor(batch["chars"], dtype=torch.float32, device=device)
    return x, y, valid, chars
