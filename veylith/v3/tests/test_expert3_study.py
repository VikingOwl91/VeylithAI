import numpy as np
import pytest
import torch

from analysis.expert3_study import evaluate_expert3, replace_expert3
from architecture.config import tiny_config
from architecture.model import GPT
from training.data import to_tensors


def mixture(logits):
    scores, ids = logits.topk(2)
    return torch.zeros_like(logits).scatter(1, ids, scores.softmax(1))


def test_substitution_leaves_second_expert_padding_and_other_positions_intact():
    logits = torch.tensor([[2., 5., 1., 8., 3.], [2., 5., 1., 8., 3.],
                           [9., 0., 5., 8., 7.], [6., 5., 6., 8., 0.]])
    mask = torch.tensor([[True, False], [True, True]])
    changed, eligible, replacements = replace_expert3(logits, mask)
    assert eligible.tolist() == [[True, False], [False, True]]
    assert replacements[0] == 4
    assert torch.equal(logits[~eligible.flatten()], changed[~eligible.flatten()])
    original = mixture(logits)
    expected = original.clone()
    rows = eligible.flatten().nonzero(as_tuple=True)[0]
    expected[rows, replacements] = original[rows, 3]
    expected[rows, 3] = 0
    assert torch.equal(mixture(changed), expected)


def test_zero_eligible_is_identity():
    logits = torch.randn(3, 8, generator=torch.Generator().manual_seed(11))
    changed, eligible, replacements = replace_expert3(logits, torch.zeros((1, 3), dtype=torch.bool))
    assert torch.equal(logits, changed)
    assert not eligible.any() and replacements.numel() == 0


def test_report_scores_the_first_target_after_a_replacement_and_excludes_padding():
    class Pieces:
        def get_piece_size(self):
            return 7

        def id_to_piece(self, idx):
            return ["<unk>", "<s>", "</s>", "<pad>", "word", ".", "!"][idx]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = GPT(tiny_config(moe_impl="dense", n_layer=3, vocab_size=7)).eval()
    batch = {"tokens": np.array([[4, 5, 4, 6, 2], [5, 4, 2, 6, 6]], dtype=np.uint16),
             "lengths": [5, 3]}
    x, y, _ = to_tensors(batch, "cpu")
    gate = model.blocks[1].mlp.gate
    with torch.inference_mode():
        # Constant logits make E3 first, E1 second, and E0 the substitute.
        def fixed_gate(module, args, logits):
            return logits.new_tensor([2., 3., 1., 4.]).expand_as(logits)

        def reference_gate(module, args, logits):
            result = logits.clone()
            result[[1, 3, 4]] = logits.new_tensor([4., 3., 1., 2.])
            return result

        with gate.register_forward_hook(fixed_gate):
            regular = model(x)[0]
            with gate.register_forward_hook(reference_gate):
                alternate = model(x)[0]
            report = evaluate_expert3(model, [("test", batch)], Pieces())

    def nll(logits):
        return -logits.log_softmax(-1).gather(-1, y.unsqueeze(-1)).flatten()

    diff = nll(alternate).double() - nll(regular).double()
    assert report["substitute_counts"] == [3, 0, 0, 0]
    for key, positions in (("replaced", [1, 3, 4]), ("first_replaced_in_sequence", [1, 4]),
                           ("unaffected_prefix", [0]), ("all", [0, 1, 2, 3, 4, 5])):
        item = report["outcomes"][key]
        assert item["targets"] == len(positions)
        assert item["delta_ce"] == pytest.approx(diff[positions].mean().item(), abs=1e-7)
