"""Multi-source preparation, merge, quota and report tests."""

import json
import os

import pytest

from data_pipeline import dataset as ds
from data_pipeline import multi_source, prepare, tokenize
import support


def _docs():
    docs_a = support.make_docs(10)
    docs_b = [{"id": f"b-{index:03d}", "text": support.make_text(1000 + index),
               "meta": {"language": "en", "language_score": 0.95, "int_score": 4}}
              for index in range(8)]
    # one exact duplicate across sources, one near duplicate
    docs_b[0] = {"id": "b-000", "text": docs_a[0]["text"],
                 "meta": {"language": "en", "language_score": 0.95, "int_score": 4}}
    docs_b[1] = {"id": "b-001", "text": docs_a[1]["text"] + " " + docs_a[1]["text"],
                 "meta": {"language": "en", "language_score": 0.95, "int_score": 4}}
    return docs_a, docs_b


def _prepare_sources(tmp_path, *, val=0, test=0):
    root = str(tmp_path / "multi")
    docs_a, docs_b = _docs()
    for docs, name, dataset_id in ((docs_a, "src-a", "local/a"),
                                   (docs_b, "src-b", "local/b")):
        source = support.FakeSource(docs, name=name, dataset_id=dataset_id)
        prepare.run_prepare(
            source, os.path.join(root, "sources", name), limit_docs=20,
            filter_config=prepare.FilterConfig(min_chars=100), shard_docs=4,
            val_permille=val, test_permille=test, log=lambda *_: None)
    return root


def _quotas(target=120.0, est=10.0):
    return {"src-a": {"train_tokens_target": target, "est_chars_per_token": est},
            "src-b": {"train_tokens_target": target, "est_chars_per_token": est}}


def test_merge_dedups_across_sources_and_applies_quotas(tmp_path):
    root = _prepare_sources(tmp_path)
    manifest = multi_source.merge_sources(
        root, _quotas(), seed=0, shard_docs=5, val_permille=0, test_permille=0,
        log=lambda *_: None)
    assert manifest["counts"]["exact_duplicates"] >= 1
    assert manifest["counts"]["near_duplicates"] >= 1
    assert set(manifest["counts"]["by_source"]) == {"src-a", "src-b"}
    assert ds.verify_corpus(root)["ok"]

    keys = []
    sources_seen = set()
    for shard in manifest["documents"]["train"]:
        for record in ds.iter_jsonl(ds.resolve(ds.corpus_dir(root), shard["path"])):
            keys.append(record["doc_key"])
            sources_seen.add(record["source"])
    assert len(keys) == len(set(keys))
    assert sources_seen == {"src-a", "src-b"}

    digest = manifest["digest"]
    again = multi_source.merge_sources(
        root, _quotas(), seed=0, shard_docs=5, val_permille=0, test_permille=0,
        log=lambda *_: None)
    assert again["digest"] == digest


def test_quota_shortfall_is_reported_and_not_backfilled(tmp_path):
    root = _prepare_sources(tmp_path)
    quotas = _quotas(target=1000000.0)
    manifest = multi_source.merge_sources(
        root, quotas, seed=0, shard_docs=5, val_permille=0, test_permille=0,
        log=lambda *_: None)
    for name in ("src-a", "src-b"):
        quota = manifest["quotas"][name]
        assert quota["shortfall_chars"] > 0
        # every candidate left after cross-source dedup is selected; no truncation
        assert quota["train_documents_selected"] == quota["candidates_train"]


def test_multi_source_tokenize_and_report(tmp_path):
    root = _prepare_sources(tmp_path, val=300, test=300)
    multi_source.merge_sources(
        root, _quotas(), seed=0, shard_docs=5, val_permille=300, test_permille=300,
        log=lambda *_: None)
    tokenize.tokenize_corpus(root, vocab_size=320, log=lambda *_: None)
    out = str(tmp_path / "report.json")
    report = multi_source.report_multi(root, out=out)
    assert report["totals"]["train_tokens"] > 0
    for name in ("src-a", "src-b"):
        assert report["sources"][name]["train_tokens"] > 0
        assert report["sources"][name]["heldout"]["docs"] > 0
        assert report["sources"][name]["fertility_tokens_per_word"] is not None
    assert ds.verify_corpus(root, deep=True)["ok"]
    stored = json.load(open(out, encoding="utf-8"))
    assert stored["digest"] == report["digest"]
