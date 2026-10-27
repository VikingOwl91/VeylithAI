"""Exact character spans for the prefix-decode bits/char protocol."""

from __future__ import annotations

import numpy as np

from data_pipeline import dataset as ds


class PrefixCharCounter:
    """Measure the decoded character span of the next-token targets.

    For a token sequence ``ids[0:L]`` the evaluated targets are ``ids[1:L]``.
    The exact span is the length of the decoded sequence minus the length of
    the decoded first token. This is the telescoped prefix-decode definition
    from the evaluation protocol, so no per-target decode is required.
    """

    def __init__(self, data_root: str, manifest: dict):
        import sentencepiece as spm

        tokenization = manifest.get("tokenization")
        if not tokenization:
            raise ValueError("corpus is not tokenized")
        relative = tokenization["tokenizer"]["model"]
        self.model_path = ds.resolve(ds.corpus_dir(data_root), relative)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    def target_chars(self, tokens: np.ndarray, lengths) -> np.ndarray:
        rows = len(lengths)
        chars = np.zeros(rows, dtype=np.float64)
        for row, length in enumerate(lengths):
            length = int(length)
            if length <= 1:
                continue
            ids = [int(value) for value in tokens[row, :length]]
            full = len(self.processor.decode(ids))
            first = len(self.processor.decode(ids[:1]))
            if full < first:
                raise ValueError(
                    f"prefix decode shrank at row {row}: {first} -> {full}")
            chars[row] = float(full - first)
        return chars
