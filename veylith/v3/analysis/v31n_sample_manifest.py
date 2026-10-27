"""Write the v3.1-N validation sample manifest (no inference).

Sequence IDs are (source, batch in loader order, row). The sample is the first
ten source-pure batches per source, so it is fully determined by the corpus and
the loader. The token digest is checked against the analysis summary.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "veylith/v3")

from v31n_expert_behavior import (  # noqa: E402
    BATCHES_PER_SOURCE, DATA, SOURCES, build_sample, sha256_file,
)

OUT = "veylith/v3/docs/v31n-sample.csv"
SUMMARY = "veylith/v3/docs/v31n-expert-behavior-summary.json"


def main():
    import sentencepiece as spm
    from data_pipeline import dataset as ds

    manifest = json.load(open(os.path.join(DATA, "corpus", "manifest.json"),
                              encoding="utf-8"))
    tokenization = manifest["tokenization"]
    tokenizer_path = ds.resolve(ds.corpus_dir(DATA),
                                tokenization["tokenizer"]["model"])
    assert sha256_file(tokenizer_path) == tokenization["tokenizer"]["sha256"]
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    tokens, lengths, source_ids, order, digest = build_sample(
        sp, SOURCES, BATCHES_PER_SOURCE)
    summary = json.load(open(SUMMARY, encoding="utf-8"))
    assert digest == summary["sample"]["digest_sha256"], "sample digest changed"
    assert len(order) == tokens.shape[0] == 240

    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sequence_id", "source", "loader_batch", "row",
                         "length", "first_token_id", "token_digest_sha256"])
        for index, row in enumerate(order):
            writer.writerow([index, row["source"], row["batch"], row["row"],
                             row["length"], row["first_ids"][0], digest])
    print("wrote", OUT, os.path.getsize(OUT), "bytes;",
          "sequences", len(order), "digest", digest)


if __name__ == "__main__":
    main()
