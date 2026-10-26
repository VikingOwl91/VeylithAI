"""CPU-only tests for veylith/evaluate.py.

Run from the repository root:

    python -m pytest veylith/tests -q
"""

import json
import math
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # veylith/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluate as E  # noqa: E402

V1_CHAR = os.path.join(E.V1_DIR, "checkpoints", "ckpt_char.pt")
V1_BPE = os.path.join(E.V1_DIR, "checkpoints", "ckpt.pt")
V2_BPE = os.path.join(E.V2_DIR, "checkpoints", "ckpt.pt")


def test_section_selection_is_deterministic():
    corpus = E.read_corpus()
    first, marks = E.make_sections(corpus)
    second, _ = E.make_sections(corpus)
    assert [item["sha256"] for item in first] == [item["sha256"] for item in second]
    assert [item["start"] for item in first] == sorted(item["start"] for item in first)
    for item in first:
        assert item["start"] >= marks["char_model_start"]
        assert item["start"] >= marks["bpe_model_start_char"]
        assert item["end"] <= len(corpus)
        assert len(item["text"]) == item["chars"]
    for left, right in zip(first, first[1:]):
        assert left["end"] <= right["start"]


def test_metrics_use_evaluated_characters():
    result = E.metrics_from_totals(nats=150.0, tokens=50, chars=100)
    assert result["main_ce"] == pytest.approx(3.0)
    assert result["main_perplexity"] == pytest.approx(math.exp(3.0), rel=1e-6)
    assert result["bits_per_char"] == pytest.approx(150.0 / 100.0 / math.log(2))
    other = E.metrics_from_totals(nats=150.0, tokens=50, chars=200)
    assert other["main_ce"] == pytest.approx(3.0)
    assert other["bits_per_char"] == pytest.approx(150.0 / 200.0 / math.log(2))
    assert other["bits_per_char"] == pytest.approx(result["bits_per_char"] / 2.0)
    with pytest.raises(ValueError):
        E.metrics_from_totals(1.0, 0, 1)


def test_bpe_tokenizer_roundtrips_the_corpus():
    pytest.importorskip("sentencepiece")
    sp = E._bpe_processor()
    ids = torch.load(E.BPE_IDS_PATH, weights_only=True).tolist()
    assert sp.decode(ids) == E.read_corpus()


def test_char_and_bpe_tokenizers_differ():
    pytest.importorskip("sentencepiece")
    corpus = E.read_corpus()
    sections, _ = E.make_sections(corpus, count=1, width=128, gap=64)
    text = sections[0]["text"]
    char_record = E.load_entry({"name": "c", "path": V1_CHAR})
    bpe_record = E.load_entry({"name": "b", "path": V1_BPE})
    assert char_record["status"] == "ok" and bpe_record["status"] == "ok"
    char_ids = char_record["_encode"](text)
    bpe_ids = bpe_record["_encode"](text)
    assert len(char_ids) == len(text)
    assert len(bpe_ids) < len(char_ids)
    assert char_record["_decode"](char_ids) == text
    assert bpe_record["_decode"](bpe_ids) == text


def test_char_checkpoint_evaluation_shape():
    corpus = E.read_corpus()
    sections, _ = E.make_sections(corpus, count=1, width=256, gap=64)
    record = E.load_entry({"name": "v1-char", "path": V1_CHAR})
    result = E.evaluate_model(record, sections)
    assert result["measured"]["tokens"] == 255
    assert result["measured"]["chars"] == 255
    assert result["derived"]["main_ce"] > 0
    assert result["derived"]["bits_per_char"] == pytest.approx(
        result["derived"]["main_ce"] / math.log(2), rel=1e-5)


def test_bpe_checkpoint_evaluation_uses_prefix_char_spans():
    pytest.importorskip("sentencepiece")
    corpus = E.read_corpus()
    sections, _ = E.make_sections(corpus, count=1, width=256, gap=64)
    record = E.load_entry({"name": "v1-bpe", "path": V1_BPE})
    result = E.evaluate_model(record, sections)
    tokens = result["measured"]["tokens"]
    chars = result["measured"]["chars"]
    assert 0 < tokens < chars  # BPE tokens cover more than one character
    assert chars <= sections[0]["chars"]
    assert result["derived"]["bits_per_char"] == pytest.approx(
        result["measured"]["nats"] / chars / math.log(2), rel=1e-4)


def test_evaluation_is_reproducible_for_one_model():
    corpus = E.read_corpus()
    sections, _ = E.make_sections(corpus, count=1, width=256, gap=64)
    record = E.load_entry({"name": "v1-bpe", "path": V1_BPE})
    assert E.evaluate_model(record, sections) == E.evaluate_model(record, sections)


def test_missing_and_incompatible_checkpoints(tmp_path):
    missing = E.load_entry({"name": "ghost", "path": str(tmp_path / "nope.pt")})
    assert missing["status"] == "missing"

    bad_path = tmp_path / "bad.pt"
    torch.save({"model": {}, "config": {"bogus": 1}}, bad_path)
    bad = E.load_entry({"name": "bad", "path": str(bad_path)})
    assert bad["status"] == "incompatible"
    assert "schema" in bad["error"] or "config" in bad["error"]


def test_generation_is_deterministic():
    record = E.load_entry({"name": "v1-char", "path": V1_CHAR})
    prompt_ids = record["_encode"]("ROMEO:")
    first = E.generate_ids(record["_model"], prompt_ids, 8, 0.8, 40, 123, greedy=False)
    second = E.generate_ids(record["_model"], prompt_ids, 8, 0.8, 40, 123, greedy=False)
    assert first == second
    greedy_a = E.generate_ids(record["_model"], prompt_ids, 8, 0.8, 40, None, greedy=True)
    greedy_b = E.generate_ids(record["_model"], prompt_ids, 8, 0.8, 40, None, greedy=True)
    assert greedy_a == greedy_b
    assert all(0 <= token < record["_cfg"].vocab_size for token in first)


def test_prompt_support_is_explicit():
    char_record = E.load_entry({"name": "c", "path": V1_CHAR})
    with pytest.raises(KeyError):
        char_record["_encode"]("€")
    pytest.importorskip("sentencepiece")
    bpe_record = E.load_entry({"name": "b", "path": V1_BPE})
    assert bpe_record["_encode"]("€")


def test_router_statistics_are_consistent():
    corpus = E.read_corpus()
    sections, _ = E.make_sections(corpus, count=1, width=256, gap=64)
    record = E.load_entry({"name": "v2", "path": V2_BPE})
    stats = E.router_stats(record, sections)
    assert stats is not None
    k = record["_cfg"].n_experts_per_tok
    experts = record["_cfg"].n_experts
    for layer, measured in stats["measured"].items():
        derived = stats["derived"][layer]
        assert sum(measured["counts"]) == measured["tokens"] * k
        assert measured["slots"] == measured["tokens"] * k
        assert len(measured["counts"]) == experts
        assert sum(derived["share_pct_of_slots"]) == pytest.approx(100.0, abs=1e-6)
        assert 0.0 <= derived["entropy_norm"] <= 1.0
        assert derived["min_share_pct"] <= derived["max_share_pct"]
        assert derived["mean_topk_weight"] and all(0.0 <= value <= 1.0
                                                   for value in derived["mean_topk_weight"])


def test_repetition_ratio_bounds():
    assert E.repetition_ratio([1, 2, 3], 3) == 0.0
    assert E.repetition_ratio([1, 1, 1, 1, 1, 1], 3) == pytest.approx(0.75)
    assert E.repetition_ratio([1, 2], 3) is None


def test_cli_writes_a_valid_payload(tmp_path):
    out = tmp_path / "results.json"
    cmd = [sys.executable, os.path.join(ROOT, "evaluate.py"),
           "--out", str(out), "--sections", "1", "--section-chars", "256",
           "--gap-chars", "64", "--no-generation", "--no-router"]
    run = subprocess.run(cmd, cwd=os.path.dirname(ROOT), capture_output=True,
                         text=True, timeout=180)
    assert run.returncode == 0, run.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["device"] == "cpu"
    assert set(payload) >= {"checkpoints", "evaluation", "protocol"}
    assert payload["evaluation"], "no checkpoint was evaluated"
    for entry in payload["evaluation"].values():
        assert entry["derived"]["bits_per_char"] > 0
    assert "Traceback" not in run.stderr
