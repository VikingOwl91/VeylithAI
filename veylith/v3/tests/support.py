"""Helpers shared by the v3 pipeline tests."""

import json
import os

from data_pipeline import prepare, sources, tokenize

WORDS = (
    "river stone market garden letter winter summer window village engine "
    "silver copper forest island castle meadow harbor bridge tower candle "
    "hunter singer dancer writer painter sailor farmer baker merchant "
    "morning evening shadow thunder whisper journey promise memory silence "
    "orange purple yellow green crimson azure golden iron wooden paper "
    "question answer reason purpose fortune danger comfort sorrow joy"
).split()


def make_text(seed, words=120, marker=""):
    import random
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(words)) + f" {marker}".strip()


def make_docs(count=30):
    docs = []
    for index in range(count):
        docs.append({
            "id": f"doc-{index:03d}",
            "text": make_text(index, marker=f"marker{index:03d}"),
            "meta": {"language": "en", "language_score": 0.95, "int_score": 4},
        })
    return docs


def write_source(path, docs):
    with open(path, "w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return path


class FakeSource:
    """In-memory source with the same interface as the real adapters."""

    def __init__(self, docs, revision="fake-rev-1", license_name="unknown",
                 name="local-jsonl", dataset_id="local/jsonl"):
        self.name = name
        self.spec = sources.SourceSpec(
            name=name, dataset_id=dataset_id, config="-",
            split="train", license_name=license_name, access="local")
        self.revision = revision
        self.docs = docs

    def iter_documents(self, *, offset=0, max_rows=None, limit_rows=None):
        yielded = 0
        for index in range(offset, len(self.docs)):
            if max_rows is not None and index - offset >= max_rows:
                return
            if limit_rows is not None and yielded >= limit_rows:
                return
            doc = self.docs[index]
            yield sources.SourceDocument(
                doc_key=f"{self.spec.dataset_id}@{self.revision}:{doc['id']}",
                source=self.name,
                dataset_id=self.spec.dataset_id,
                revision=self.revision,
                source_id=doc["id"],
                text=doc["text"],
                meta={"row_index": index, **(doc.get("meta") or {})},
                arrival_index=index,
            )
            yielded += 1


def prepare_root(tmp_path, source, *, docs=30, name="run", **kwargs):
    root = str(tmp_path / name)
    manifest = prepare.run_prepare(
        source, root, limit_docs=docs,
        filter_config=kwargs.pop("filter_config", prepare.FilterConfig(min_chars=100)),
        shard_docs=kwargs.pop("shard_docs", 8),
        val_permille=kwargs.pop("val_permille", 200),
        test_permille=kwargs.pop("test_permille", 200),
        log=lambda *_: None, **kwargs)
    return root, manifest


def stage_and_tokenize(tmp_path, source, *, docs=30, vocab=512, name="run", **kwargs):
    root, manifest = prepare_root(tmp_path, source, docs=docs, name=name, **kwargs)
    tokenization = tokenize.tokenize_corpus(root, vocab_size=vocab,
                                            log=lambda *_: None)
    manifest = __import__("data_pipeline.dataset", fromlist=["read_json"]).read_json(
        __import__("data_pipeline.dataset", fromlist=["manifest_path"]).manifest_path(root))
    return root, manifest, tokenization
