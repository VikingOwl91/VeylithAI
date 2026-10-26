"""BPE tokenizer training on the train split and token shards.

The tokenizer input is built only from final train-split documents, sorted by
``doc_key``. Every document ends with the EOS id; the per-document offsets are
stored next to the token array, so packing can respect or explicitly cross
document boundaries.
"""

from __future__ import annotations

import os

import numpy as np

from . import dataset as ds

SPECIAL_IDS = {"unk": 0, "bos": 1, "eos": 2, "pad": 3}


def _load_documents(root: str, split: str):
    manifest = ds.read_json(ds.manifest_path(root))
    corpus = ds.corpus_dir(root)
    docs = []
    for shard in manifest["documents"][split]:
        for record in ds.iter_jsonl(ds.resolve(corpus, shard["path"])):
            docs.append((record["doc_key"], record["text"]))
    return manifest, docs


def train_tokenizer(root: str, *, vocab_size: int = 8192, log=print) -> dict:
    """Train BPE on train-split documents only; return manifest metadata."""
    import sentencepiece as spm

    manifest, docs = _load_documents(root, "train")
    if not docs:
        raise ValueError("no train documents; run prepare first")
    docs.sort(key=lambda item: item[0])

    tokenizer_dir = os.path.join(ds.corpus_dir(root), "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    stem = f"bpe{vocab_size}"
    train_text_rel = f"tokenizer/{stem}.train.txt"
    train_text_path = ds.resolve(ds.corpus_dir(root), train_text_rel)

    chunks = [text for _, text in docs]
    text = "\n\n".join(chunks) + "\n"
    ds.atomic_write_text(train_text_path, text)

    model_rel = f"tokenizer/{stem}.model"
    vocab_rel = f"tokenizer/{stem}.vocab"
    model_path = ds.resolve(ds.corpus_dir(root), model_rel)
    vocab_path = ds.resolve(ds.corpus_dir(root), vocab_rel)
    tmp_model = model_path + f".tmp.{os.getpid()}"
    tmp_vocab = vocab_path + f".tmp.{os.getpid()}"
    try:
        # sentence_iterator plus model_writer keeps the training-text path out
        # of the model proto, so the model bytes are reproducible across roots.
        with open(tmp_model, "wb") as handle:
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(chunks),
                model_writer=handle,
                model_type="bpe",
                vocab_size=int(vocab_size),
                character_coverage=0.9995,
                byte_fallback=True,
                normalization_rule_name="identity",
                remove_extra_whitespaces=False,
                add_dummy_prefix=False,
                num_threads=1,
                shuffle_input_sentence=False,
                split_by_whitespace=True,
                unk_id=SPECIAL_IDS["unk"],
                bos_id=SPECIAL_IDS["bos"],
                eos_id=SPECIAL_IDS["eos"],
                pad_id=SPECIAL_IDS["pad"],
                unk_piece="<unk>",
                bos_piece="<s>",
                eos_piece="</s>",
                pad_piece="<pad>",
                train_extremely_large_corpus=False,
                minloglevel=2,
            )
        processor = spm.SentencePieceProcessor(model_file=tmp_model)
        with open(tmp_vocab, "w", encoding="utf-8") as handle:
            for piece_id in range(processor.get_piece_size()):
                handle.write(f"{processor.id_to_piece(piece_id)}\t"
                             f"{processor.get_score(piece_id)}\n")
        os.replace(tmp_model, model_path)
        os.replace(tmp_vocab, vocab_path)
    finally:
        for leftover in (tmp_model, tmp_vocab):
            if os.path.exists(leftover):
                os.remove(leftover)

    processor = spm.SentencePieceProcessor(model_file=model_path)
    if processor.get_piece_size() != int(vocab_size):
        raise ValueError(
            f"tokenizer produced {processor.get_piece_size()} pieces, "
            f"expected {vocab_size}; the train split is too small")
    log(f"  tokenizer: {vocab_size} pieces, train text {len(text):,} chars")

    return {
        "vocab_size": int(vocab_size),
        "special_ids": dict(SPECIAL_IDS),
        "tokenizer": {
            "model": model_rel,
            "sha256": ds.sha256_file(model_path),
            "vocab": vocab_rel,
            "vocab_sha256": ds.sha256_file(vocab_path),
            "train_text": train_text_rel,
            "train_text_sha256": ds.sha256_file(train_text_path),
        },
    }


def tokenize_corpus(root: str, *, vocab_size: int = 8192, log=print) -> dict:
    """Train the tokenizer and write token shards for all three splits."""
    import sentencepiece as spm

    manifest = ds.read_json(ds.manifest_path(root))
    if manifest.get("tokenization") is not None:
        raise ValueError("manifest already has tokenization; use a fresh root")
    if vocab_size > 65535:
        raise ValueError("uint16 token shards require vocab_size <= 65535")
    tokenizer = train_tokenizer(root, vocab_size=vocab_size, log=log)
    corpus = ds.corpus_dir(root)
    model_path = ds.resolve(corpus, tokenizer["tokenizer"]["model"])
    processor = spm.SentencePieceProcessor(model_file=model_path)
    eos_id = SPECIAL_IDS["eos"]
    pad_id = SPECIAL_IDS["pad"]

    splits = {}
    total_tokens = 0
    for split in ds.SPLITS:
        shard_meta = []
        split_tokens = 0
        split_docs = 0
        for shard in manifest["documents"][split]:
            name = os.path.basename(shard["path"])[:-len(".jsonl")]
            index = int(name.rsplit("-", 1)[1])
            records = list(ds.iter_jsonl(ds.resolve(corpus, shard["path"])))
            ids = []
            docs_meta = []
            for record in records:
                pieces = processor.encode(record["text"], out_type=int)
                start = len(ids)
                ids.extend(pieces)
                ids.append(eos_id)
                docs_meta.append({
                    "doc_key": record["doc_key"],
                    "start": start,
                    "end": len(ids),
                    "tokens": len(ids) - start,
                    "text_sha256": ds.sha256_text(record["text"]),
                })
            array = np.asarray(ids, dtype=np.uint16)
            data_rel = ds.token_shard_relpath(split, index)
            meta_rel = ds.token_meta_relpath(split, index)
            data_path = ds.resolve(corpus, data_rel)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            tmp = data_path + f".tmp.{os.getpid()}.npy"
            np.save(tmp, array)
            os.replace(tmp, data_path)
            meta = {
                "format_version": 1,
                "split": split,
                "tokens": int(array.size),
                "docs": docs_meta,
                "eos_id": eos_id,
                "pad_id": pad_id,
                "vocab_size": int(vocab_size),
            }
            ds.atomic_write_json(ds.resolve(corpus, meta_rel), meta)
            shard_meta.append({
                "path": data_rel,
                "meta": meta_rel,
                "sha256": ds.sha256_file(data_path),
                "bytes": os.path.getsize(data_path),
                "tokens": int(array.size),
                "docs": len(docs_meta),
            })
            split_tokens += int(array.size)
            split_docs += len(docs_meta)
        splits[split] = {"tokens": split_tokens, "docs": split_docs,
                         "shards": shard_meta}
        total_tokens += split_tokens
        log(f"  tokens {split}: {split_docs} docs, {split_tokens:,} tokens")

    tokenization = {**tokenizer, "splits": splits, "total_tokens": total_tokens}
    manifest["tokenization"] = tokenization
    manifest["digest"] = ds.manifest_digest(manifest)
    problems = ds.validate_manifest(manifest)
    if problems:
        raise ValueError(f"manifest is invalid after tokenization: {problems}")
    ds.atomic_write_json(ds.manifest_path(root), manifest)
    return tokenization


def load_processor(root: str):
    import sentencepiece as spm
    manifest = ds.read_json(ds.manifest_path(root))
    tokenization = manifest.get("tokenization")
    if not tokenization:
        raise ValueError("corpus is not tokenized")
    model_path = ds.resolve(ds.corpus_dir(root), tokenization["tokenizer"]["model"])
    return spm.SentencePieceProcessor(model_file=model_path)
