"""Tokenizer, token shards, packing and CLI tests for the v3 pipeline."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from data_pipeline import dataset as ds
from data_pipeline import loader, prepare, sources, tokenize
import support

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
PIPELINE = os.path.join(V3, "pipeline.py")


def test_tokenizer_is_reproducible_and_trained_on_train_only(tmp_path):
    docs = support.make_docs(60)
    revision = "fake-rev-1"
    val_id = next(candidate for candidate in (f"val-marker-{index}" for index in range(2000))
                  if prepare.split_for(f"local/jsonl@{revision}:{candidate}",
                                       200, 200) == "validation")
    marker = "zzuniquemarkerzz"
    docs.append({"id": val_id, "text": support.make_text(999, marker=marker),
                 "meta": {"language": "en", "language_score": 0.95, "int_score": 4}})
    source = support.FakeSource(docs, revision=revision)

    root_a, manifest_a, tokenization_a = support.stage_and_tokenize(
        tmp_path, source, docs=len(docs), name="a")
    root_b, manifest_b, tokenization_b = support.stage_and_tokenize(
        tmp_path, source, docs=len(docs), name="b")

    train_text = open(os.path.join(ds.corpus_dir(root_a), "tokenizer",
                                   "bpe512.train.txt"), encoding="utf-8").read()
    assert marker not in train_text
    assert tokenization_a["tokenizer"]["sha256"] == tokenization_b["tokenizer"]["sha256"]
    assert tokenization_a["total_tokens"] == tokenization_b["total_tokens"]
    for split in ds.SPLITS:
        assert ([shard["sha256"] for shard in tokenization_a["splits"][split]["shards"]]
                == [shard["sha256"] for shard in tokenization_b["splits"][split]["shards"]])
    assert manifest_a["digest"] == manifest_b["digest"]


def test_token_shards_roundtrip_document_boundaries(tmp_path):
    import sentencepiece as spm
    source = support.FakeSource(support.make_docs(24))
    root, manifest, tokenization = support.stage_and_tokenize(tmp_path, source, docs=24)
    processor = spm.SentencePieceProcessor(
        model_file=ds.resolve(ds.corpus_dir(root), tokenization["tokenizer"]["model"]))
    eos_id = tokenization["special_ids"]["eos"]

    text_by_key = {}
    for split in ds.SPLITS:
        for shard in manifest["documents"][split]:
            for record in ds.iter_jsonl(ds.resolve(ds.corpus_dir(root), shard["path"])):
                text_by_key[record["doc_key"]] = record["text"]

    checked = 0
    for split in ds.SPLITS:
        for shard in tokenization["splits"][split]["shards"]:
            tokens = np.load(ds.resolve(ds.corpus_dir(root), shard["path"]))
            meta = ds.read_json(ds.resolve(ds.corpus_dir(root), shard["meta"]))
            assert int(tokens.size) == meta["tokens"]
            previous_end = 0
            for doc in meta["docs"]:
                assert doc["start"] == previous_end
                previous_end = doc["end"]
                assert int(tokens[doc["end"] - 1]) == eos_id
                text = processor.decode(tokens[doc["start"]:doc["end"] - 1].tolist())
                assert text == text_by_key[doc["doc_key"]]
                assert ds.sha256_text(text) == doc["text_sha256"]
                checked += 1
    assert checked == sum(tokenization["splits"][s]["docs"] for s in ds.SPLITS)
    assert ds.verify_corpus(root, deep=True)["ok"]


def test_deep_verify_detects_token_tampering(tmp_path):
    source = support.FakeSource(support.make_docs(16))
    root, manifest, tokenization = support.stage_and_tokenize(tmp_path, source, docs=16)
    assert ds.verify_corpus(root, deep=True)["ok"]

    shard = tokenization["splits"]["train"]["shards"][0]
    path = ds.resolve(ds.corpus_dir(root), shard["path"])
    tokens = np.load(path)
    original = int(tokens[0])
    tokens[0] = np.uint16(10 if original != 10 else 11)
    np.save(path, tokens)

    manifest_path = ds.manifest_path(root)
    stored = ds.read_json(manifest_path)
    stored_shard = stored["tokenization"]["splits"]["train"]["shards"][0]
    stored_shard["sha256"] = ds.sha256_file(path)
    stored_shard["bytes"] = os.path.getsize(path)
    stored["digest"] = ds.manifest_digest(stored)
    ds.atomic_write_json(manifest_path, stored)

    assert ds.verify_corpus(root, deep=False)["ok"]
    deep = ds.verify_corpus(root, deep=True)
    assert not deep["ok"]
    assert any("decode mismatch" in problem for problem in deep["problems"])


def test_concat_packing_covers_every_token_once(tmp_path):
    source = support.FakeSource(support.make_docs(24))
    root, manifest, tokenization = support.stage_and_tokenize(tmp_path, source, docs=24)
    plan = loader.plan_sequences(root, split="train", seq_len=16, mode="concat")
    assert loader.verify_plan(plan) == []
    assert plan["tokens"] == tokenization["splits"]["train"]["tokens"]

    rebuilt = np.concatenate([loader.read_sequence(root, sequence)
                              for sequence in plan["sequences"]])
    shards = [np.load(ds.resolve(ds.corpus_dir(root), shard["path"]))
              for shard in tokenization["splits"]["train"]["shards"]]
    original = np.concatenate(shards)
    assert np.array_equal(rebuilt, original)
    assert len(rebuilt) == tokenization["splits"]["train"]["tokens"]


def test_document_packing_never_crosses_documents(tmp_path):
    source = support.FakeSource(support.make_docs(20))
    root, manifest, tokenization = support.stage_and_tokenize(tmp_path, source, docs=20)
    plan = loader.plan_sequences(root, split="train", seq_len=16, mode="document")
    assert loader.verify_plan(plan) == []
    for sequence in plan["sequences"]:
        assert len(sequence.doc_keys) == 1
        for path, start, take in sequence.segments:
            meta_path = path.replace(".tokens.npy", ".meta.json")
            meta = ds.read_json(ds.resolve(ds.corpus_dir(root), meta_path))
            doc = next(item for item in meta["docs"]
                       if item["doc_key"] == sequence.doc_keys[0])
            assert doc["start"] <= start
            assert start + take <= doc["end"]


def test_sampler_and_batches_are_deterministic(tmp_path):
    source = support.FakeSource(support.make_docs(20))
    root, manifest, tokenization = support.stage_and_tokenize(tmp_path, source, docs=20)
    plan = loader.plan_sequences(root, split="train", seq_len=16, mode="concat")
    sampler = loader.EpochSampler(len(plan["sequences"]), seed=7)
    first = sampler.order(0)
    assert np.array_equal(first, sampler.order(0))
    assert sorted(first.tolist()) == list(range(len(plan["sequences"])))
    assert sampler.digest(0) == sampler.digest(0)

    batches = list(loader.iter_batches(plan, root, batch_size=3, seed=7))
    flat = [index for batch in batches for index in batch["indices"]]
    assert flat == first.tolist()
    for batch in batches:
        assert batch["order_digest"] == sampler.digest(0)
        assert batch["tokens"].shape == (len(batch["indices"]), 16)
        for row, length in zip(batch["tokens"], batch["lengths"]):
            assert 0 < length <= 16
            if length < 16:
                assert np.all(row[length:] == plan["pad_id"])

    again = list(loader.iter_batches(plan, root, batch_size=3, seed=7))
    assert [batch["indices"] for batch in batches] == [batch["indices"] for batch in again]
    assert all(np.array_equal(left["tokens"], right["tokens"])
               for left, right in zip(batches, again))


def _run_cli(args, cwd):
    return subprocess.run([sys.executable, PIPELINE] + args, cwd=cwd,
                          capture_output=True, text=True, timeout=600)


def test_cli_end_to_end_with_a_local_source(tmp_path):
    docs = support.make_docs(30)
    path = tmp_path / "source.jsonl"
    support.write_source(path, docs)
    root = tmp_path / "run"
    prepare_cmd = _run_cli([
        "prepare", "--source", "local-jsonl", "--path", str(path),
        "--root", str(root), "--docs", "24", "--shard-docs", "8",
        "--val-permille", "200", "--test-permille", "200", "--min-chars", "100",
    ], cwd=tmp_path)
    assert prepare_cmd.returncode == 0, prepare_cmd.stderr
    assert "documents: train=" in prepare_cmd.stdout
    assert "digest:" in prepare_cmd.stdout

    token_cmd = _run_cli(["tokenize", "--root", str(root), "--vocab", "512"],
                         cwd=tmp_path)
    assert token_cmd.returncode == 0, token_cmd.stderr
    assert "total tokens:" in token_cmd.stdout

    verify_cmd = _run_cli(["verify", "--root", str(root), "--deep"], cwd=tmp_path)
    assert verify_cmd.returncode == 0, verify_cmd.stderr
    assert verify_cmd.stdout.startswith("ok: digest=")

    batch_cmd = _run_cli(["batches", "--root", str(root), "--seq-len", "16",
                          "--batch-size", "2", "--batches", "1"], cwd=tmp_path)
    assert batch_cmd.returncode == 0, batch_cmd.stderr
    assert "plan:" in batch_cmd.stdout and "order=" in batch_cmd.stdout

    manifest = ds.read_json(ds.manifest_path(str(root)))
    shard = manifest["documents"]["train"][0]
    shard_path = ds.resolve(ds.corpus_dir(str(root)), shard["path"])
    data = bytearray(open(shard_path, "rb").read())
    data[len(data) // 2] ^= 0x01
    with open(shard_path, "wb") as handle:
        handle.write(bytes(data))
    broken = _run_cli(["verify", "--root", str(root)], cwd=tmp_path)
    assert broken.returncode == 1
    assert "sha256 mismatch" in broken.stdout


def test_cli_maps_source_errors_to_exit_codes(monkeypatch, capsys):
    from data_pipeline import cli

    def boom(*args, **kwargs):
        raise sources.SourceAuthError("denied")

    monkeypatch.setattr(cli.sources, "open_source", boom)
    assert cli.main(["inspect", "--limit", "1"]) == 2
    assert "access denied" in capsys.readouterr().err

    def offline(*args, **kwargs):
        raise sources.SourceNetworkError("offline")

    monkeypatch.setattr(cli.sources, "open_source", offline)
    assert cli.main(["inspect", "--limit", "1"]) == 3
    assert "network failure" in capsys.readouterr().err

    def broken(*args, **kwargs):
        raise sources.SourceError("bad source")

    monkeypatch.setattr(cli.sources, "open_source", broken)
    assert cli.main(["inspect", "--limit", "1"]) == 4
    assert "source error" in capsys.readouterr().err


@pytest.mark.skipif(os.environ.get("VEYLITH_NETWORK_TESTS") != "1",
                    reason="set VEYLITH_NETWORK_TESTS=1 for the network smoke")
def test_fineweb_http_smoke():
    source = sources.FineWebEdu(backend="http")
    documents = list(source.iter_documents(limit_rows=3))
    assert len(documents) == 3
    assert documents[0].revision == source.info()["sha"]
    assert documents[0].doc_key.startswith(
        "HuggingFaceFW/fineweb-edu@" + documents[0].revision)
    assert documents[0].text
