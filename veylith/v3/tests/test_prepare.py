"""Tests for normalization, filtering, split assignment and deduplication."""

import json
import os

import pytest

from data_pipeline import dataset as ds
from data_pipeline import prepare
import support


def test_normalize_text_is_stable_and_content_preserving():
    raw = "  a\r\nb  \r\n\r\n\r\nc\x00d  "
    assert prepare.normalize_text(raw) == "a\nb\n\ncd"
    assert prepare.normalize_text("e\u0301") == "\u00e9"
    assert prepare.normalize_text(raw) == prepare.normalize_text(raw)


def test_validate_text_reports_the_first_reason():
    cfg = prepare.FilterConfig(min_chars=10, max_chars=200)
    assert prepare.validate_text("", cfg) == (False, "empty")
    assert prepare.validate_text("short", cfg) == (False, "too_short")
    assert prepare.validate_text("x" * 300, cfg) == (False, "too_long")
    assert prepare.validate_text("word " * 20 + "\ufffd" * 5, cfg) == (
        False, "replacement_chars")
    assert prepare.validate_text("12345 67890 12345", cfg) == (False, "letter_ratio")
    assert prepare.validate_text("word " * 20, cfg, {"language": "de"}) == (
        False, "language")
    assert prepare.validate_text("word " * 20, cfg, {"language_score": 0.1}) == (
        False, "language_score")
    assert prepare.validate_text("word " * 20, cfg, {"int_score": 1}) == (
        False, "int_score")
    assert prepare.validate_text("word " * 20, cfg, {"language": "en",
                                                     "int_score": 4})[0]


def test_split_for_is_pure_and_covers_all_splits():
    keys = [f"k-{index}" for index in range(2000)]
    splits = [prepare.split_for(key, 300, 300) for key in keys]
    assert set(splits) == {"train", "validation", "test"}
    assert all(prepare.split_for(key, 300, 300) == split
               for key, split in zip(keys, splits))
    assert prepare.split_for("k-1", 0, 0) == "train"


def test_exact_dedup_keeps_the_min_key():
    entries = [
        {"doc_key": "b", "exact_hash": "same", "simhash": 0},
        {"doc_key": "a", "exact_hash": "same", "simhash": 0},
        {"doc_key": "c", "exact_hash": "other", "simhash": 0x5555555555555555},
    ]
    result = prepare.assign_splits_and_dedup(entries, val_permille=0, test_permille=0)
    decisions = result["decisions"]
    assert decisions["a"]["action"] == "keep"
    assert decisions["b"]["action"] == "exact_duplicate"
    assert decisions["b"]["canonical"] == "a"
    assert decisions["c"]["action"] == "keep"
    assert result["stats"]["exact_duplicates"] == 1
    assert result["stats"]["near_duplicates"] == 0


def test_near_dedup_clusters_a_shared_text_and_keeps_the_min_key():
    base = " ".join(f"word{index}" for index in range(200))
    near = base + " " + base
    unrelated = " ".join(f"other{index}" for index in range(200))
    assert prepare.hamming64(prepare.simhash64(base),
                             prepare.simhash64(near)) <= 3
    entries = [
        {"doc_key": "b", "exact_hash": "h2", "simhash": prepare.simhash64(near)},
        {"doc_key": "a", "exact_hash": "h1", "simhash": prepare.simhash64(base)},
        {"doc_key": "c", "exact_hash": "h3", "simhash": prepare.simhash64(unrelated)},
    ]
    result = prepare.assign_splits_and_dedup(entries, val_permille=0, test_permille=0)
    decisions = result["decisions"]
    assert decisions["a"]["action"] == "keep"
    assert decisions["b"]["action"] == "near_duplicate"
    assert decisions["b"]["canonical"] == "a"
    assert decisions["c"]["action"] == "keep"


def test_prepare_is_deterministic(tmp_path):
    source = support.FakeSource(support.make_docs(40))
    root_a, manifest_a = support.prepare_root(tmp_path, source, docs=40, name="a")
    root_b, manifest_b = support.prepare_root(tmp_path, source, docs=40, name="b")
    assert manifest_a["digest"] == manifest_b["digest"]
    for split in ds.SPLITS:
        shards_a = [shard["sha256"] for shard in manifest_a["documents"][split]]
        shards_b = [shard["sha256"] for shard in manifest_b["documents"][split]]
        assert shards_a == shards_b
    assert ds.verify_corpus(root_a)["ok"]


def test_prepare_split_isolation_and_manifest_counts(tmp_path):
    source = support.FakeSource(support.make_docs(80))
    root, manifest = support.prepare_root(
        tmp_path, source, docs=80, name="run", shard_docs=10,
        val_permille=300, test_permille=300)
    counts = manifest["counts"]
    assert set(counts["documents"]) == set(ds.SPLITS)
    assert sum(counts["documents"].values()) == counts["accepted"]
    assert counts["documents"]["train"] > 0
    assert counts["documents"]["validation"] > 0
    assert counts["documents"]["test"] > 0
    assert counts["exact_duplicates"] == 0

    seen = {}
    for split in ds.SPLITS:
        for shard in manifest["documents"][split]:
            for record in ds.iter_jsonl(ds.resolve(ds.corpus_dir(root), shard["path"])):
                assert record["doc_key"] not in seen
                seen[record["doc_key"]] = split
                assert prepare.split_for(record["doc_key"], 300, 300) == split
    assert ds.verify_corpus(root)["ok"]


def test_prepare_stops_at_the_document_limit(tmp_path):
    source = support.FakeSource(support.make_docs(40))
    _, manifest = support.prepare_root(tmp_path, source, docs=10, shard_docs=4)
    assert manifest["counts"]["accepted"] == 10
    assert sum(manifest["counts"]["documents"].values()) == 10


def test_resume_matches_an_uninterrupted_run(tmp_path):
    source = support.FakeSource(support.make_docs(24))
    config = prepare.FilterConfig(min_chars=100)
    resumed = str(tmp_path / "resumed")
    prepare.run_prepare(source, resumed, limit_docs=8, filter_config=config,
                        shard_docs=5, val_permille=0, test_permille=0,
                        finalize=False, log=lambda *_: None)
    manifest_resumed = prepare.run_prepare(
        source, resumed, limit_docs=16, filter_config=config, shard_docs=5,
        val_permille=0, test_permille=0, resume=True, log=lambda *_: None)

    single = str(tmp_path / "single")
    manifest_single = prepare.run_prepare(
        source, single, limit_docs=16, filter_config=config, shard_docs=5,
        val_permille=0, test_permille=0, log=lambda *_: None)

    assert manifest_resumed["digest"] == manifest_single["digest"]
    for split in ds.SPLITS:
        assert ([shard["sha256"] for shard in manifest_resumed["documents"][split]]
                == [shard["sha256"] for shard in manifest_single["documents"][split]])
    assert ds.verify_corpus(resumed)["ok"]


def test_resume_refuses_changed_source_or_filter(tmp_path):
    docs = support.make_docs(16)
    source = support.FakeSource(docs)
    root = str(tmp_path / "run")
    prepare.run_prepare(source, root, limit_docs=8,
                        filter_config=prepare.FilterConfig(min_chars=100),
                        shard_docs=5, val_permille=0, test_permille=0,
                        finalize=False, log=lambda *_: None)

    changed_source = support.FakeSource(docs, revision="other-rev")
    with pytest.raises(ValueError) as err:
        prepare.run_prepare(changed_source, root, limit_docs=16,
                            filter_config=prepare.FilterConfig(min_chars=100),
                            shard_docs=5, val_permille=0, test_permille=0,
                            resume=True, log=lambda *_: None)
    assert "source mismatch" in str(err.value)

    with pytest.raises(ValueError) as err:
        prepare.run_prepare(source, root, limit_docs=16,
                            filter_config=prepare.FilterConfig(min_chars=300),
                            shard_docs=5, val_permille=0, test_permille=0,
                            resume=True, log=lambda *_: None)
    assert "filter" in str(err.value)


def test_tampered_document_shard_fails_verification(tmp_path):
    source = support.FakeSource(support.make_docs(20))
    root, manifest = support.prepare_root(tmp_path, source, docs=20)
    shard = manifest["documents"]["train"][0]
    path = ds.resolve(ds.corpus_dir(root), shard["path"])
    data = bytearray(open(path, "rb").read())
    data[len(data) // 2] ^= 0x01
    with open(path, "wb") as handle:
        handle.write(bytes(data))
    result = ds.verify_corpus(root)
    assert not result["ok"]
    assert any("sha256 mismatch" in problem for problem in result["problems"])


def test_manifest_validation_rejects_missing_fields_and_bad_digest(tmp_path):
    source = support.FakeSource(support.make_docs(12))
    root, manifest = support.prepare_root(tmp_path, source, docs=12)
    broken = json.loads(json.dumps(manifest))
    del broken["source"]["license"]
    problems = ds.validate_manifest(broken)
    assert any("source.license" in problem for problem in problems)

    broken = json.loads(json.dumps(manifest))
    broken["digest"] = "0" * 64
    assert "manifest digest mismatch" in ds.validate_manifest(broken)

    broken = json.loads(json.dumps(manifest))
    broken["counts"]["documents"]["train"] += 1
    assert any("counts.documents.train" in problem
               for problem in ds.validate_manifest(broken))


def test_manifest_records_revision_license_and_filters(tmp_path):
    source = support.FakeSource(support.make_docs(12), revision="rev-abc",
                                license_name="odc-by")
    _, manifest = support.prepare_root(tmp_path, source, docs=12)
    assert manifest["source"]["revision"] == "rev-abc"
    assert manifest["source"]["license"] == "odc-by"
    assert manifest["source"]["access"] == "local"
    assert manifest["filters"]["min_chars"] == 100
    assert manifest["split_policy"]["val_permille"] == 200
    assert manifest["dedup"]["canonical"] == "min doc_key of the cluster"
    assert len(manifest["digest"]) == 64


def test_near_duplicate_pairs_are_clean_after_dedup(tmp_path):
    docs = support.make_docs(12)
    shared = support.make_text(500)
    docs.append({"id": "near-a", "text": shared,
                 "meta": {"language": "en", "language_score": 0.95, "int_score": 4}})
    docs.append({"id": "near-b", "text": shared + " " + shared,
                 "meta": {"language": "en", "language_score": 0.95, "int_score": 4}})
    source = support.FakeSource(docs)
    root, manifest = support.prepare_root(tmp_path, source, docs=len(docs),
                                          val_permille=300, test_permille=300)
    assert manifest["counts"]["near_duplicates"] >= 1
    result = prepare.find_near_duplicate_pairs(root)
    assert result["pairs"] == []
    assert result["checked"] == sum(manifest["counts"]["documents"].values())


def test_prepare_stops_at_the_char_limit(tmp_path):
    source = support.FakeSource(support.make_docs(30))
    root, manifest = support.prepare_root(tmp_path, source, docs=30,
                                          name="chars", limit_chars=6000)
    assert manifest["counts"]["accepted"] < 30
    assert sum(manifest["counts"]["chars"].values()) >= 6000
