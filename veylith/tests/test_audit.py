"""CPU-only tests for veylith/audit.py.

Run from the repository root:

    python -m pytest veylith/tests -q
"""

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # veylith/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import audit as A  # noqa: E402
import evaluate as E  # noqa: E402

V1_BPE = os.path.join(E.V1_DIR, "checkpoints", "ckpt.pt")
V2_BPE = os.path.join(E.V2_DIR, "checkpoints", "ckpt.pt")


def test_train_sections_are_deterministic_and_disjoint_from_validation():
    corpus = E.read_corpus()
    train = A.make_train_sections(corpus)
    again = A.make_train_sections(corpus)
    assert [item["sha256"] for item in train] == [item["sha256"] for item in again]
    marks = E.boundaries(corpus)
    val_sections, _ = E.make_sections(corpus)
    for item in train:
        assert item["end"] <= marks["char_model_start"]
        assert item["end"] <= marks["bpe_model_start_char"]
        assert item["start"] + item["chars"] == item["end"]
        for val in val_sections:
            assert item["end"] <= val["start"] or val["end"] <= item["start"]


def test_parameter_accounting_matches_the_model_structure():
    v1 = E.load_entry({"name": "v1", "path": V1_BPE})
    acc1 = A.parameter_accounting(v1)
    assert acc1["total_trainable"] == sum(p.numel() for p in v1["_model"].parameters())
    assert acc1["active_per_token_main"] == acc1["total_trainable"]
    assert acc1["moe_layers"] == 0
    assert acc1["categories"]["routed_experts"] == 0

    v2 = E.load_entry({"name": "v2", "path": V2_BPE})
    acc2 = A.parameter_accounting(v2)
    assert sum(acc2["categories"].values()) == acc2["total_trainable"]
    assert acc2["active_per_token_main"] < acc2["total_trainable"]
    assert acc2["active_per_token_main"] == (
        acc2["always_active"] + acc2["moe_layers"] * acc2["top_k"] * acc2["routed_expert_size"])
    assert acc2["moe_layers"] == 5
    assert acc2["top_k"] == 2
    assert acc2["total_to_active_ratio"] > 2.0
    assert acc2["mtp_extra_parameters"] == 0


def test_attention_layout():
    v2 = E.load_entry({"name": "v2", "path": V2_BPE})
    layout = A.attention_layout(v2["_cfg"], "v2")
    assert layout["full_layers"] == [3]
    assert layout["swa_layers"] == [0, 1, 2, 4, 5]
    assert layout["window"] == 128

    v1 = E.load_entry({"name": "v1", "path": V1_BPE})
    assert A.attention_layout(v1["_cfg"], "v1")["full_layers"] == [0, 1, 2, 3, 4, 5]

    all_full = SimpleNamespace(n_layer=4, full_attn_every=1, sliding_window=128)
    assert A.attention_layout(all_full, "v2")["full_layers"] == [0, 1, 2, 3]


def test_documented_decomposition_is_consistent():
    documented = A.documented_decomposition()
    shares = documented["derived"]["share_pct_of_composite"]
    assert sum(shares.values()) == pytest.approx(100.0, abs=1e-6)
    assert documented["measured"]["main_nats_per_token_implied"] == pytest.approx(3.539, abs=0.001)
    assert documented["measured"]["mtp_ce_implied"] == pytest.approx(4.892, abs=0.01)
    assert documented["derived"]["weighted_terms"]["mtp_weighted"] > 2.0


def test_gradient_components_do_not_change_weights():
    module = E.V2
    cfg = module.Config(n_layer=2, n_head=2, n_kv_head=1, n_embd=32, block_size=16,
                        sliding_window=8, full_attn_every=4, dropout=0.0,
                        n_experts=4, n_experts_per_tok=2, first_k_dense=1,
                        n_mtp=1, vocab_size=64)
    torch.manual_seed(0)
    model = module.GPT(cfg).eval()
    record = {"_model": model, "_cfg": cfg}
    x = torch.randint(0, 64, (1, 12))
    y = torch.randint(0, 64, (1, 12))
    result = A.gradient_components(record, [(x, y)], max_batches=1)
    assert result["status"] == "measured"
    assert result["weights_unchanged"] is True
    batch = result["batches"][0]
    assert batch["measured"]["grad_norm_main"] > 0
    assert batch["measured"]["grad_norm_mtp"] > 0
    assert -1.0 <= batch["derived"]["cosine_main_mtp"] <= 1.0
    assert 0.0 <= batch["derived"]["mtp_share_of_composite_norm"] <= 1.0


def test_training_budget_arithmetic():
    record = {"checkpoint": {"args": {"max_iters": 10, "batch_size": 4}},
              "_cfg": SimpleNamespace(block_size=8)}
    budget = A.training_budget(record, train_tokens=100, selection_step=5)
    assert budget["tokens_per_step"] == 32
    assert budget["token_positions"] == 320
    assert budget["corpus_passes"] == 3.2
    assert budget["selected_corpus_passes"] == 1.6
    assert budget["possible_window_starts"] == 93
    assert budget["windows_drawn"] == 40
    assert budget["mean_draws_per_start"] == pytest.approx(40 / 93, rel=1e-4)
    assert budget["expected_unseen_fraction"] == pytest.approx(math.exp(-40 / 93), rel=1e-3)


def test_evaluate_reuses_the_same_metrics():
    corpus = E.read_corpus()
    sections, _ = E.make_sections(corpus, count=1, width=256, gap=64)
    v2 = E.load_entry({"name": "v2", "path": V2_BPE})
    result = E.evaluate_model(v2, sections)
    assert result["measured"]["tokens"] > 0
    assert result["derived"]["main_ce"] > 0
