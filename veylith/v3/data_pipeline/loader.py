"""Deterministic sequence packing and batch sampling.

Two packing modes:

``concat`` (default)
    Tokens are concatenated in shard and document order and cut into
    ``seq_len`` windows. A window may contain more than one document; EOS
    tokens mark every boundary and the window records its document keys. No
    token is used twice in one epoch, and no window overlaps another.

``document``
    Every document is cut into ``seq_len`` pieces on its own; the last piece
    is padded with the pad id. No window contains two documents.

Sampling is counter-based: epoch ``e`` uses
``numpy.random.SeedSequence([seed, e])``; the permutation is a pure function
of ``(seed, epoch, count)`` and its digest is returned with every batch.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

from . import dataset as ds


@dataclass(frozen=True)
class Sequence:
    index: int
    segments: tuple[tuple[str, int, int], ...]
    length: int
    doc_keys: tuple[str, ...]
    padded: bool = False


def _load_tokens(root: str):
    manifest = ds.read_json(ds.manifest_path(root))
    tokenization = manifest.get("tokenization")
    if not tokenization:
        raise ValueError("corpus is not tokenized; run tokenize first")
    return manifest, tokenization


def plan_sequences(root: str, split: str = "train", seq_len: int = 256,
                   mode: str = "concat", pad_id: int = 3) -> dict:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if mode not in ("concat", "document"):
        raise ValueError(f"unknown packing mode: {mode}")
    if split not in ds.SPLITS:
        raise ValueError(f"unknown split: {split}")
    _, tokenization = _load_tokens(root)
    corpus = ds.corpus_dir(root)
    shards = tokenization["splits"][split]["shards"]
    sequences: list[Sequence] = []
    index = 0

    buffer_segments: list[tuple[str, int, int]] = []
    buffer_length = 0
    buffer_docs: list[str] = []

    for shard in shards:
        meta = ds.read_json(ds.resolve(corpus, shard["meta"]))
        for doc in meta["docs"]:
            start = int(doc["start"])
            end = int(doc["end"])
            if mode == "document":
                position = start
                while position < end:
                    take = min(end - position, seq_len)
                    padded = take < seq_len
                    sequences.append(Sequence(
                        index=index,
                        segments=((shard["path"], position, take),),
                        length=take,
                        doc_keys=(doc["doc_key"],),
                        padded=padded,
                    ))
                    index += 1
                    position += take
                continue

            position = start
            while position < end:
                take = min(end - position, seq_len - buffer_length)
                buffer_segments.append((shard["path"], position, take))
                buffer_length += take
                if doc["doc_key"] not in buffer_docs:
                    buffer_docs.append(doc["doc_key"])
                position += take
                if buffer_length == seq_len:
                    sequences.append(Sequence(index, tuple(buffer_segments),
                                              buffer_length, tuple(buffer_docs)))
                    index += 1
                    buffer_segments = []
                    buffer_length = 0
                    buffer_docs = []

    if mode == "concat" and buffer_length:
        sequences.append(Sequence(index, tuple(buffer_segments),
                                  buffer_length, tuple(buffer_docs)))
        index += 1

    total = sum(sequence.length for sequence in sequences)
    return {
        "split": split,
        "seq_len": int(seq_len),
        "mode": mode,
        "pad_id": int(pad_id),
        "tokens": total,
        "sequences": sequences,
    }


def read_sequence(root: str, sequence: Sequence, cache: dict | None = None) -> np.ndarray:
    corpus = ds.corpus_dir(root)
    parts = []
    for path, start, take in sequence.segments:
        array = None if cache is None else cache.get(path)
        if array is None:
            array = np.load(ds.resolve(corpus, path))
            if cache is not None:
                cache[path] = array
        parts.append(array[start:start + take])
    if not parts:
        return np.zeros(0, dtype=np.uint16)
    return np.concatenate(parts)


class EpochSampler:
    """Pure function of (seed, epoch, count) to a permutation."""

    def __init__(self, count: int, seed: int = 0):
        if count < 0:
            raise ValueError("count must be non-negative")
        self.count = int(count)
        self.seed = int(seed)

    def order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(epoch)]))
        return rng.permutation(self.count)

    def digest(self, epoch: int) -> str:
        return hashlib.sha256(self.order(epoch).astype(np.int64).tobytes()).hexdigest()


def iter_batches(plan: dict, root: str, *, batch_size: int = 4, epochs: int = 1,
                 seed: int = 0, max_batches: int | None = None):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sequences = plan["sequences"]
    seq_len = plan["seq_len"]
    sampler = EpochSampler(len(sequences), seed)
    cache: dict = {}
    produced = 0
    for epoch in range(int(epochs)):
        order = sampler.order(epoch)
        digest = sampler.digest(epoch)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            tokens = np.full((len(indices), seq_len), plan["pad_id"], dtype=np.uint16)
            lengths = []
            for row, sequence_index in enumerate(indices):
                sequence = sequences[int(sequence_index)]
                values = read_sequence(root, sequence, cache)
                if len(values) > seq_len:
                    raise ValueError(
                        f"sequence {sequence.index} has {len(values)} tokens, "
                        f"seq_len is {seq_len}")
                tokens[row, :len(values)] = values
                lengths.append(len(values))
            yield {
                "epoch": epoch,
                "batch": start // batch_size,
                "indices": [int(value) for value in indices],
                "tokens": tokens,
                "lengths": lengths,
                "order_digest": digest,
            }
            produced += 1
            if max_batches is not None and produced >= max_batches:
                return


def verify_plan(plan: dict, expected_tokens: int | None = None,
                check_coverage: bool = True) -> list[str]:
    """Structural checks: indices, lengths, coverage, no token reuse."""
    problems = []
    sequences = plan["sequences"]
    if [sequence.index for sequence in sequences] != list(range(len(sequences))):
        problems.append("sequence indices are not 0..n-1")
    total = sum(sequence.length for sequence in sequences)
    if total != plan["tokens"]:
        problems.append("sum of sequence lengths does not match plan tokens")
    if expected_tokens is not None and total != expected_tokens:
        problems.append(f"plan covers {total} tokens, expected {expected_tokens}")
    if plan["mode"] == "document":
        for sequence in sequences:
            if not sequence.padded and sequence.length != plan["seq_len"]:
                problems.append(f"unpadded short sequence {sequence.index}")
            if sequence.padded and sequence.length >= plan["seq_len"]:
                problems.append(f"padded full sequence {sequence.index}")
    else:
        for sequence in sequences[:-1]:
            if sequence.length != plan["seq_len"]:
                problems.append(f"short window {sequence.index} before the end")
    if check_coverage:
        seen: dict[str, set[int]] = {}
        for sequence in sequences:
            for path, start, take in sequence.segments:
                bucket = seen.setdefault(path, set())
                for position in range(start, start + take):
                    if position in bucket:
                        problems.append(f"token reuse at {path}:{position}")
                    bucket.add(position)
        for path, positions in seen.items():
            if positions and sorted(positions) != list(range(min(positions),
                                                              max(positions) + 1)):
                problems.append(f"coverage gap in {path}")
    return problems
