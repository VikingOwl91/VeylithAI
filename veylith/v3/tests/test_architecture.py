"""CPU-only architecture validation for v3.1-A.

No full-size forward or backward. Full-size parameter counts use the meta
device; behavior tests use tiny configurations.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from architecture.accounting import (
    active_parameter_count,
    analytical_counts,
    count_model_parameters,
    memory_estimate,
    training_flops,
    wall_clock_estimate,
)
from architecture.config import CANDIDATES, ModelConfig, get_candidate, tiny_config
from architecture.model import GPT


@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_candidate_counts_are_in_the_target_range(name):
    cfg = get_candidate(name)
    counts = analytical_counts(cfg)
    assert 95_000_000 <= counts["total"] <= 105_000_000
    assert 25_000_000 <= counts["active_sparse"] <= 40_000_000
    category_sum = sum(counts[key] for key in (
        "embedding", "output_head", "attention", "norms", "dense_ffn",
        "router", "routed_experts", "shared_experts"))
    assert category_sum == counts["total"]
    assert counts["active_sparse"] <= counts["total"]
    assert counts["active_dense"] >= counts["active_sparse"]


@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_meta_model_matches_the_analytical_total(name):
    cfg = get_candidate(name)
    with torch.device("meta"):
        model = GPT(cfg)
    assert count_model_parameters(model) == analytical_counts(cfg)["total"]


def test_tied_embeddings_count_once():
    cfg = tiny_config(tie_embeddings=True)
    counts = analytical_counts(cfg)
    assert counts["output_head"] == 0
    with torch.device("meta"):
        model = GPT(cfg)
    assert count_model_parameters(model) == counts["total"]


def test_active_parameter_count_matches_hooks():
    for overrides in (
            dict(n_experts_per_tok=2, shared_expert=True, tie_embeddings=False),
            dict(n_experts_per_tok=1, shared_expert=True, tie_embeddings=False),
            dict(n_experts_per_tok=2, shared_expert=False, tie_embeddings=True)):
        cfg = tiny_config(**overrides)
        model = GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (1, 1))
        measured = active_parameter_count(model, idx)
        expected = analytical_counts(cfg)["active_unique"]
        assert measured == expected, (overrides, measured, expected)


def test_forward_shapes_metrics_and_router():
    cfg = tiny_config()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, metrics, router = model(
        idx, targets=targets, return_metrics=True, return_router=True)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)
    assert set(metrics) == {"main_ce", "aux_loss", "aux_loss_per_layer",
                        "composite_loss"}
    assert "mtp_ce" not in metrics
    expected = metrics["main_ce"] + cfg.moe_aux_weight * metrics["aux_loss"]
    assert torch.allclose(metrics["composite_loss"], expected)
    assert len(router) == cfg.n_layer - cfg.first_k_dense
    for top_idx in router:
        assert top_idx.shape == (2 * 16, cfg.n_experts_per_tok)
        assert int(top_idx.min()) >= 0
        assert int(top_idx.max()) < cfg.n_experts
        for row in top_idx:
            assert len(set(row.tolist())) == cfg.n_experts_per_tok


def test_causal_mask_blocks_future_tokens():
    cfg = tiny_config(dropout=0.0, full_attn_every=1)
    model = GPT(cfg).eval()
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        first = model(idx)[0]
        changed = idx.clone()
        changed[0, -1] = (changed[0, -1] + 1) % cfg.vocab_size
        second = model(changed)[0]
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5)
    assert not torch.allclose(first[:, -1], second[:, -1])


def test_sliding_window_blocks_old_tokens_in_a_single_layer():
    cfg = tiny_config(n_layer=1, first_k_dense=0, full_attn_every=2,
                        sliding_window=4, dropout=0.0)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        first = model(idx)[0]
        changed = idx.clone()
        changed[0, 0] = (changed[0, 0] + 1) % cfg.vocab_size
        second = model(changed)[0]
    assert torch.allclose(first[:, -1], second[:, -1], atol=1e-5)
    assert not torch.allclose(first[:, 3], second[:, 3])


def test_no_mtp_fields_or_loss():
    cfg = tiny_config()
    assert not hasattr(cfg, "n_mtp")
    assert not hasattr(cfg, "mtp_weight")
    model = GPT(cfg)
    assert not any("mtp" in key for key in model.state_dict())
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    _, _, metrics = model(idx, targets=targets, return_metrics=True)
    assert "mtp_ce" not in metrics


def test_tiny_backward_is_finite():
    cfg = tiny_config(dropout=0.0)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss, metrics = model(idx, targets=targets, return_metrics=True)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters()
             if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert model.blocks[-1].mlp.gate.weight.grad is not None
    assert model.blocks[-1].mlp.gate.weight.grad.abs().sum() > 0
    assert metrics["aux_loss"] > 0


def test_config_validation_rejects_invalid_combinations():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=30, n_head=4).validate()
    with pytest.raises(ValueError):
        ModelConfig(n_experts=4, n_experts_per_tok=5).validate()
    with pytest.raises(ValueError):
        ModelConfig(shared_expert=True, shared_ffn_dim=0).validate()
    with pytest.raises(ValueError):
        ModelConfig(moe_impl="magic").validate()
    with pytest.raises(ValueError):
        ModelConfig(n_embd=32, n_head=4, first_k_dense=99).validate()


def test_memory_estimates_are_monotonic_and_fit_16gb():
    cfg = get_candidate("A")
    estimates = [memory_estimate(cfg, batch, seq_len=512,
                                 activation_checkpointing=True)
                 for batch in (1, 2, 4, 8)]
    totals = [estimate["total_gb"] for estimate in estimates]
    assert totals == sorted(totals)
    assert estimates[-1]["fits_16gb"]
    no_ckpt = memory_estimate(cfg, 8, seq_len=512, activation_checkpointing=False)
    assert no_ckpt["activation_gb"] > estimates[-1]["activation_gb"]
    dense = memory_estimate(cfg, 8, seq_len=512, dense_combined=True)
    assert dense["activation_gb"] >= estimates[-1]["activation_gb"]


def test_compute_estimates_scale_and_are_positive():
    cfg = get_candidate("A")
    small = training_flops(cfg, 95_951_233)
    medium = training_flops(cfg, 500_000_000)
    large = training_flops(cfg, 1_000_000_000)
    assert 0 < small["total_flops"] < medium["total_flops"] < large["total_flops"]
    assert small["attention_flops"] > 0
    dense = training_flops(cfg, 95_951_233, dense_combined=True)
    assert dense["total_flops"] > small["total_flops"]
    wall = wall_clock_estimate(cfg, 95_951_233)
    assert wall["hours"] > 0


@pytest.mark.parametrize("moe_impl", ["sparse", "dense"])
def test_moe_bf16_paths_are_finite_on_cpu(moe_impl):
    # Regression for the bf16 index_add dtype bug found by the GPU smoke.
    cfg = tiny_config(dropout=0.0, moe_impl=moe_impl)
    model = GPT(cfg).to(torch.bfloat16)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss, metrics = model(idx, targets=targets, return_metrics=True)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_aux_balanced_baseline_is_one_per_layer_and_sums_over_layers():
    cfg = tiny_config(n_layer=2, first_k_dense=0, dropout=0.0)
    model = GPT(cfg)
    for block in model.blocks:
        torch.nn.init.zeros_(block.mlp.gate.weight)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    _, _, metrics = model(idx, targets=targets, return_metrics=True)
    n_moe = sum(1 for block in model.blocks if block.is_moe)
    assert n_moe == 2
    assert metrics["aux_loss_per_layer"].item() == pytest.approx(1.0, rel=1e-5)
    assert metrics["aux_loss"].item() == pytest.approx(float(n_moe), rel=1e-5)
    assert metrics["composite_loss"].item() == pytest.approx(
        metrics["main_ce"].item() + cfg.moe_aux_weight * metrics["aux_loss"].item(),
        rel=1e-6)
