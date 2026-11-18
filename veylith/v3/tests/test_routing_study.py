"""Check the CE mask, class alignment and paired resampling on tiny CPU inputs."""

import hashlib

import numpy as np
import pytest
import torch

from analysis.routing_study import evaluate, paired_stats, verified_hash


class Pieces:
    pieces = ["<unk>", "<s>", "</s>", "<pad>", "7", ".", "word"]

    def get_piece_size(self):
        return len(self.pieces)

    def id_to_piece(self, index):
        return self.pieces[index]

    def eos_id(self):
        return 2


def test_mask_and_input_token_classes():
    pieces = Pieces()
    logits = torch.tensor([0., 1., 2., 3., 4., 5., 6.])

    def model(x):
        return logits.expand(*x.shape, -1), None

    batch = {"tokens": np.array([[4, 5, 2, 6], [6, 4, 3, 3]], dtype=np.uint16),
             "lengths": [4, 2]}
    result = evaluate(model, [("test", batch)], pieces, [], [])
    baseline = result["baseline"]
    nll = -logits.log_softmax(0)
    assert baseline["all"]["targets"] == 4
    assert baseline["all"]["ce"] == pytest.approx(nll[[5, 2, 6, 4]].mean().item())
    assert baseline["numeric"]["ce"] == pytest.approx(nll[5].item())
    assert baseline["eos"]["ce"] == pytest.approx(nll[6].item())
    assert baseline["punct_only"]["targets"] == 1


def test_paired_bootstrap_weights_tokens_and_handles_absent_classes():
    counts = np.array([[1, 0], [3, 0], [2, 0], [2, 0]])
    deltas = np.array([[1., 0.], [-3., 0.], [0., 0.], [0., 0.]])
    groups = np.array(["a", "a", "b", "b"])
    stats = paired_stats(deltas, counts, groups, draws=100)
    assert stats[0]["delta_ce"] == -.25
    assert stats[0]["bootstrap_nonempty"] == 100
    assert stats[1] == {"delta_ce": None, "sequence_bootstrap_ci95": None, "bootstrap_nonempty": 0}
    assert stats == paired_stats(deltas, counts, groups, draws=100)


def test_hash_mismatch_prevents_input_loading():
    from pathlib import Path
    path = Path(__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verified_hash(path, digest) == digest
    with pytest.raises(ValueError, match="SHA-256"):
        verified_hash(path, "0" * 64)


def test_paired_recovery_is_zero_when_all_valid_inputs_share_one_group():
    from architecture.config import tiny_config
    from architecture.model import GPT
    model = GPT(tiny_config(moe_impl="dense", vocab_size=7)).eval()
    # The final token is a target only. All three valid inputs are punctuation.
    batch = {"tokens": np.array([[5, 5, 5, 6]], dtype=np.uint16), "lengths": [4]}
    report = evaluate(model, [("test", batch)], Pieces(), [1], [11], compare_sentence_end=True)
    comparison = report["layers"]["1"]["sentence_end_comparison"]
    recovery = comparison["unrestricted_minus_grouped"]["sentence_end"]
    assert report["baseline"]["sentence_end"]["targets"] == 3
    assert recovery["delta_ce"] == 0
    assert recovery["batch_bootstrap_ci95"] == [0., 0.]
    assert comparison["recovered_ce_per_seed"]["11"]["numeric"] is None


def test_sentence_end_recovery_matches_two_full_forward_passes():
    from analysis.routing_control import routing_control
    from architecture.config import tiny_config
    from architecture.model import GPT
    from training.data import to_tensors

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = GPT(tiny_config(moe_impl="dense", vocab_size=7)).eval()
    batch = {"tokens": np.array([[4, 5, 6, 5, 2], [5, 6, 4, 4, 5]], dtype=np.uint16),
             "lengths": [5, 5]}
    x, y, valid = to_tensors(batch, "cpu")
    punctuation = x == 5
    batch_seed = int(np.random.SeedSequence([11, 0]).generate_state(1)[0])
    with torch.inference_mode():
        losses = []
        for group_mask in (None, punctuation):
            with routing_control(model, 1, valid, batch_seed, group_mask):
                logits, _ = model(x)
            losses.append(torch.nn.functional.cross_entropy(
                logits[punctuation], y[punctuation], reduction="none").double())
        expected = (losses[0] - losses[1]).mean().item()
    report = evaluate(model, [("test", batch)], Pieces(), [1], [11], compare_sentence_end=True)
    comparison = report["layers"]["1"]["sentence_end_comparison"]
    assert abs(expected) > 1e-8
    assert comparison["unrestricted_minus_grouped"]["sentence_end"]["delta_ce"] == pytest.approx(expected)
