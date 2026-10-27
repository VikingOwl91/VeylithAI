"""CPU smoke tests for the v3.1-A training infrastructure."""

import json
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from architecture.config import CANDIDATES, get_candidate, tiny_config
from architecture.model import GPT
from data_pipeline import multi_source, prepare, tokenize
from training.checkpoint import load_state
from training.config import sha256_file
from training.config import TrainConfig
from training.data import iter_source_batches, load_manifest, source_names
from training.evaluate import evaluate
from training.schedule import lr_at
from training.train import train
import support


@pytest.fixture(autouse=True)
def reset_determinism():
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("v31-corpus"))
    sources_docs = {
        "src-a": (support.make_docs(20), "local/a"),
        "src-b": ([{"id": f"b-{index:03d}", "text": support.make_text(1000 + index),
                    "meta": {"language": "en", "language_score": 0.95, "int_score": 4}}
                   for index in range(20)], "local/b"),
    }
    for name, (docs, dataset_id) in sources_docs.items():
        source = support.FakeSource(docs, name=name, dataset_id=dataset_id)
        prepare.run_prepare(
            source, os.path.join(root, "sources", name), limit_docs=20,
            filter_config=prepare.FilterConfig(min_chars=100), shard_docs=4,
            val_permille=300, test_permille=300, log=lambda *_: None)
    quotas = {"src-a": {"train_tokens_target": 1000000, "est_chars_per_token": 4.0},
              "src-b": {"train_tokens_target": 1000000, "est_chars_per_token": 4.0}}
    multi_source.merge_sources(root, quotas, seed=0, shard_docs=8,
                               val_permille=300, test_permille=300,
                               log=lambda *_: None)
    tokenize.tokenize_corpus(root, vocab_size=512, log=lambda *_: None)
    return root


def make_config(root, out_dir, **overrides) -> TrainConfig:
    model = tiny_config(vocab_size=512, block_size=16, n_layer=2, n_head=2,
                        n_kv_head=1, n_embd=32, n_experts=4,
                        n_experts_per_tok=2, expert_ffn_dim=32,
                        shared_expert=True, shared_ffn_dim=32, first_k_dense=1)
    values = dict(data_root=root, out_dir=str(out_dir), model=model,
                  device="cpu", precision="fp32", batch_size=2, grad_accum=2,
                  seq_len=16, max_steps=4, lr=1e-3, warmup_steps=1,
                  eval_interval=1000, eval_batch_size=2,
                  eval_max_batches=1, train_gap_batches=1, save_every=2,
                  log_every=1, seed=0, deterministic=True)
    values.update(overrides)
    return TrainConfig(**values)


def test_lr_schedule_has_warmup_and_a_floor():
    assert lr_at(0, 1e-3, 4, 10, 0.1) == pytest.approx(0.25e-3)
    assert lr_at(3, 1e-3, 4, 10, 0.1) == pytest.approx(1e-3)
    assert lr_at(10, 1e-3, 4, 10, 0.1) == pytest.approx(0.1e-3, rel=1e-6)
    assert lr_at(0, 1e-3, 0, 10, 0.1) == pytest.approx(1e-3)


def _grads(model):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()]


def test_attention_and_moe_paths_match_in_forward_and_gradients():
    torch.manual_seed(0)
    base = tiny_config(vocab_size=512, block_size=16, n_layer=2, n_head=2,
                       n_kv_head=1, n_embd=32, n_experts=4,
                       n_experts_per_tok=2, expert_ffn_dim=32,
                       shared_expert=True, shared_ffn_dim=32, dropout=0.0,
                       attention_impl="manual", moe_impl="sparse")
    manual = GPT(base)
    sdpa_cfg = tiny_config(**{**base.to_dict(), "attention_impl": "sdpa"})
    sdpa = GPT(sdpa_cfg)
    sdpa.load_state_dict(manual.state_dict())
    dense_cfg = tiny_config(**{**base.to_dict(), "moe_impl": "dense"})
    dense = GPT(dense_cfg)
    dense.load_state_dict(manual.state_dict())

    idx = torch.randint(0, 512, (2, 9))
    targets = torch.randint(0, 512, (2, 9))

    def run(model):
        model.zero_grad(set_to_none=True)
        logits, loss, metrics = model(idx, targets=targets, return_metrics=True)
        loss.backward()
        return logits.detach(), float(loss.detach()), _grads(model), metrics

    logits_m, loss_m, grads_m, metrics_m = run(manual)
    logits_s, loss_s, grads_s, metrics_s = run(sdpa)
    logits_d, loss_d, grads_d, metrics_d = run(dense)

    assert torch.allclose(logits_m, logits_s, atol=1e-5, rtol=1e-5)
    assert torch.allclose(logits_m, logits_d, atol=1e-5, rtol=1e-5)
    assert loss_m == pytest.approx(loss_s, rel=1e-5)
    assert loss_m == pytest.approx(loss_d, rel=1e-5)
    for left, right in zip(grads_m, grads_s):
        assert torch.allclose(left, right, atol=1e-4, rtol=1e-4)
    for left, right in zip(grads_m, grads_d):
        assert torch.allclose(left, right, atol=1e-4, rtol=1e-4)
    assert float(metrics_m["aux_loss"].detach()) == pytest.approx(
        float(metrics_s["aux_loss"].detach()), rel=1e-6)


def test_gradient_accumulation_equivalence():
    torch.manual_seed(1)
    cfg = tiny_config(vocab_size=512, block_size=16, n_layer=2, n_head=2,
                      n_kv_head=1, n_embd=32, n_experts=4,
                      n_experts_per_tok=2, expert_ffn_dim=32,
                      shared_expert=True, shared_ffn_dim=32, dropout=0.0,
                      moe_aux_weight=0.0)
    model = GPT(cfg)
    batches = []
    for _ in range(4):
        x = torch.randint(0, 512, (1, 8))
        y = torch.randint(0, 512, (1, 8))
        valid = torch.ones((1, 8), dtype=torch.bool)
        batches.append((x, y, valid))
    total_valid = sum(int(v.sum()) for _, _, v in batches)

    model.zero_grad(set_to_none=True)
    for x, y, valid in batches:
        logits, _, metrics = model(x, return_metrics=True)
        ce = F.cross_entropy(logits.reshape(-1, 512), y.reshape(-1),
                             reduction="none").reshape(y.shape)
        count = int(valid.sum())
        loss = (ce[valid].sum() + cfg.moe_aux_weight * metrics["aux_loss"] * count) / total_valid
        loss.backward()
    accumulated = _grads(model)

    model.zero_grad(set_to_none=True)
    x = torch.cat([item[0] for item in batches], dim=0)
    y = torch.cat([item[1] for item in batches], dim=0)
    valid = torch.cat([item[2] for item in batches], dim=0)
    logits, _, metrics = model(x, return_metrics=True)
    ce = F.cross_entropy(logits.reshape(-1, 512), y.reshape(-1),
                         reduction="none").reshape(y.shape)
    loss = (ce[valid].sum() + cfg.moe_aux_weight * metrics["aux_loss"] * int(valid.sum())) / total_valid
    loss.backward()
    full = _grads(model)
    # Main CE only (aux weight 0): the aux term is microbatch-local by design.
    for left, right in zip(accumulated, full):
        assert torch.allclose(left, right, atol=5e-4, rtol=1e-3)


def test_tiny_training_decreases_loss(corpus, tmp_path):
    cfg = make_config(corpus, tmp_path, max_steps=10, eval_interval=1000,
                      save_every=1000, lr=3e-3)
    result = train(cfg, log=lambda *_: None)
    assert result["steps"] == 10
    lines = [json.loads(line) for line in open(cfg.curve_jsonl, encoding="utf-8")]
    assert len(lines) == 10
    assert lines[-1]["train_loss"] < lines[0]["train_loss"]
    assert all(math.isfinite(line["train_loss"]) for line in lines)


def test_resume_matches_uninterrupted_run(corpus, tmp_path):
    run_a = tmp_path / "a"
    cfg_a = make_config(corpus, run_a, max_steps=4, save_every=2)
    train(cfg_a, log=lambda *_: None)

    run_b = tmp_path / "b"
    cfg_b = make_config(corpus, run_b, max_steps=4, save_every=2, stop_after=2)
    stopped = train(cfg_b, log=lambda *_: None)
    assert stopped["stopped_early"]
    resumed = make_config(corpus, run_b, max_steps=4, save_every=2,
                          resume=str(run_b / "ckpt_state.pt"),
                          state_path=str(run_b / "ckpt_state_resumed.pt"))
    train(resumed, log=lambda *_: None)

    state_a = torch.load(run_a / "ckpt_state.pt", map_location="cpu", weights_only=False)
    state_b = torch.load(run_b / "ckpt_state_resumed.pt", map_location="cpu", weights_only=False)
    assert state_a["step"] == state_b["step"] == 4
    for key in state_a["model"]:
        assert torch.allclose(state_a["model"][key], state_b["model"][key], atol=1e-6)
    assert state_a["tokens_seen"] == state_b["tokens_seen"]


def test_evaluation_reports_sources_and_router(corpus, tmp_path):
    cfg = make_config(corpus, tmp_path)
    manifest = load_manifest(corpus)
    model = GPT(cfg.model)
    metrics = evaluate(model, cfg, "validation", max_batches=2)
    assert metrics["split"] == "validation"
    assert set(metrics["sources"]) == set(source_names(manifest))
    for entry in metrics["sources"].values():
        assert entry["tokens"] > 0
        assert entry["main_ce"] is not None and math.isfinite(entry["main_ce"])
        assert entry["bits_per_char"] > 0
    assert metrics["overall"]["main_ce"] > 0
    assert metrics["overall"]["macro_main_ce"] > 0
    assert 0 < metrics["router"]["mean_entropy_norm"] <= 1.0
    for layer in metrics["router"]["layers"]:
        assert abs(sum(layer["share_pct"]) - 100.0) < 0.1


def test_artifact_has_no_mtp_and_selection_is_validation(corpus, tmp_path):
    cfg = make_config(corpus, tmp_path, max_steps=2)
    train(cfg, log=lambda *_: None)
    artifact = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    assert artifact["selection"]["split"] == "validation"
    assert artifact["selection"]["metric"] == "macro_main_ce"
    assert not any("mtp" in key for key in artifact)
    assert not any("mtp" in key for key in artifact["model"])
    assert artifact["data_manifest_sha256"] == sha256_file(
        os.path.join(corpus, "corpus", "manifest.json"))


def test_resume_rejects_a_changed_fingerprint(corpus, tmp_path):
    cfg = make_config(corpus, tmp_path, max_steps=2)
    train(cfg, log=lambda *_: None)
    state_path = str(tmp_path / "ckpt_state.pt")
    with pytest.raises(ValueError, match="fingerprint"):
        load_state(state_path, "0" * 64)


def test_cpu_guard_refuses_the_real_0_1b_model(corpus, tmp_path):
    model = get_candidate("A")
    cfg = TrainConfig(data_root=corpus, out_dir=str(tmp_path / "guard"),
                      model=model, batch_size=1, grad_accum=1, seq_len=16,
                      max_steps=1, eval_interval=1, save_every=1)
    with pytest.raises(ValueError, match="refusing"):
        train(cfg, log=lambda *_: None)
