"""Compare BPE vocabularies on the same train split.

Both tokenizers are trained by ``train_tokenizer`` on final train-split
documents only. Held-out validation and test documents are used for encoding
measurements, never for training. No model quality is claimed.

Each training runs in a child process so the parent can measure the child's
peak RSS with psutil; wall time includes the child start.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

from . import dataset as ds
from . import tokenize

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_EMBD = 256
V2_TOTAL_PARAMS = 25_602_304
V2_VOCAB = 1000


def train_worker(root: str, vocab: int) -> dict:
    """Child entry point: train one tokenizer and return its metadata."""
    return tokenize.train_tokenizer(root, vocab_size=vocab, log=lambda *_: None)


def _run_child(root: str, vocab: int) -> tuple[dict, float, float]:
    code = (
        "import json, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "from data_pipeline.tokenizer_compare import train_worker; "
        "print(json.dumps(train_worker(sys.argv[2], int(sys.argv[3]))))"
    )
    start = time.time()
    process = subprocess.Popen(
        [sys.executable, "-c", code, V3, root, str(vocab)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0
    try:
        import psutil
        handle = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, handle.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.05)
    except ImportError:
        pass
    out, err = process.communicate()
    wall = time.time() - start
    if process.returncode != 0:
        raise RuntimeError(f"tokenizer training failed for {vocab}: {err[-2000:]}")
    lines = [line for line in out.splitlines() if line.strip()]
    return json.loads(lines[-1]), wall, peak


def _heldout_documents(root: str):
    manifest = ds.read_json(ds.manifest_path(root))
    corpus = ds.corpus_dir(root)
    docs = []
    for split in ("validation", "test"):
        for shard in manifest["documents"][split]:
            for record in ds.iter_jsonl(ds.resolve(corpus, shard["path"])):
                docs.append((split, record["doc_key"], record["text"]))
    return manifest, docs


def _percentiles(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    return {
        "mean": round(float(np.mean(values)), 3),
        "median": float(statistics.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(max(values)),
    }


def build_comparison(root: str, vocabs=(4096, 8192), seq_len: int = 256,
                     probes=None) -> dict:
    import sentencepiece as spm

    probes = probes or {
        "empty": "",
        "unknown_token": "<unk>",
        "emoji": "🙂🙂 language model",
        "cjk": "日本語のテキスト",
        "combining": "e\u0301cole",
        "whitespace": "a\tb\nc\r\n",
    }
    manifest, heldout = _heldout_documents(root)
    heldout_chars = sum(len(text) for _, _, text in heldout)
    heldout_words = sum(len(text.split()) for _, _, text in heldout)

    results = {}
    for vocab in vocabs:
        metadata, train_seconds, peak_rss = _run_child(root, int(vocab))
        model_path = ds.resolve(ds.corpus_dir(root), metadata["tokenizer"]["model"])
        processor = spm.SentencePieceProcessor(model_file=model_path)

        token_counts = []
        failures = 0
        unk_usage = 0
        unk_id = int(metadata["special_ids"]["unk"])
        for _, _, text in heldout:
            pieces = processor.encode(text, out_type=int)
            token_counts.append(len(pieces))
            unk_usage += sum(1 for piece in pieces if piece == unk_id)
            if processor.decode(pieces) != text:
                failures += 1
        tokens = sum(token_counts)
        probe_results = {}
        for name, text in probes.items():
            pieces = processor.encode(text, out_type=int)
            probe_results[name] = {
                "tokens": len(pieces),
                "unk_used": int(metadata["special_ids"]["unk"]) in pieces,
                "roundtrip": processor.decode(pieces) == text,
            }
        roundtrip_failures = failures + sum(
            1 for item in probe_results.values() if not item["roundtrip"])
        document_chunks = int(sum(
            max(1, (length + seq_len - 1) // seq_len) for length in token_counts))
        concat_sequences = int((tokens + seq_len - 1) // seq_len) if tokens else 0
        vocab_size = int(vocab)
        embedding_head = 2 * vocab_size * N_EMBD
        total_params = V2_TOTAL_PARAMS - 2 * V2_VOCAB * N_EMBD + embedding_head
        results[str(vocab_size)] = {
            "model": metadata["tokenizer"]["model"],
            "model_sha256": metadata["tokenizer"]["sha256"],
            "vocab_sha256": metadata["tokenizer"]["vocab_sha256"],
            "train_text_sha256": metadata["tokenizer"]["train_text_sha256"],
            "train_seconds": round(train_seconds, 3),
            "peak_rss_bytes": int(peak_rss),
            "heldout_tokens": tokens,
            "chars_per_token": round(heldout_chars / tokens, 4) if tokens else None,
            "fertility_tokens_per_word": round(tokens / heldout_words, 4) if heldout_words else None,
            "tokens_per_doc": _percentiles(token_counts),
            "document_chunks_seq%d" % seq_len: document_chunks,
            "concat_sequences_seq%d" % seq_len: concat_sequences,
            "special_ids": metadata["special_ids"],
            "probes": probe_results,
            "roundtrip_failures": roundtrip_failures,
            "unk_usage_in_heldout": unk_usage,
            "params_embedding_and_head": embedding_head,
            "params_total_v2_shape": total_params,
        }

    ordered = sorted(results, key=int)
    best = max(ordered, key=lambda key: results[key]["chars_per_token"] or 0)
    return {
        "corpus_digest": manifest["digest"],
        "source_revision": manifest["source"]["revision"],
        "train_documents": manifest["counts"]["documents"]["train"],
        "train_chars": manifest["counts"]["chars"]["train"],
        "heldout_documents": len(heldout),
        "heldout_chars": heldout_chars,
        "heldout_words": heldout_words,
        "seq_len": seq_len,
        "vocabs": results,
        "preliminary_default": int(best),
        "note": "chars/token and fertility are measured on held-out text; "
                "they do not measure model quality",
    }


def write_comparison(root: str, out: str, vocabs=(4096, 8192)) -> dict:
    payload = build_comparison(root, vocabs=vocabs)
    ds.atomic_write_json(out, payload)
    return payload
