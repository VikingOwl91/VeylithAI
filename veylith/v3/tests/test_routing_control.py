"""CPU-only controls for the frozen routing ablation."""

import pytest

torch = pytest.importorskip("torch")

from analysis.routing_control import routing_control
from architecture.config import tiny_config
from architecture.model import GPT


def test_only_valid_router_rows_move_together_in_one_layer():
    model = GPT(tiny_config(n_layer=3, moe_impl="dense")).eval()
    target = model.blocks[2].mlp
    other = model.blocks[1].mlp
    x = torch.arange(12 * model.cfg.n_embd).float().sin().reshape(2, 6, -1)
    mask = torch.tensor([[True, False, True, False, True, False],
                         [True, True, True, False, True, False]])
    positions = mask.flatten().nonzero(as_tuple=True)[0]
    shuffled = torch.randperm(len(positions), generator=torch.Generator().manual_seed(12))
    order = torch.arange(12)
    order[positions] = positions[shuffled]
    assert not torch.equal(order, torch.arange(12))

    def assignments(gate):
        values, ids = gate(x.flatten(0, 1)).topk(model.cfg.n_experts_per_tok)
        return ids, values.softmax(dim=-1)

    with torch.inference_mode():
        ids, weights = assignments(target.gate)
        other_before = other(x)[0]
        rng_before = torch.get_rng_state()
        with routing_control(model, 2, mask, seed=12):
            moved_ids, moved_weights = assignments(target.gate)
            assert torch.equal(moved_ids, ids[order])
            assert torch.equal(moved_weights, weights[order])
            assert torch.equal(other_before, other(x)[0])
        assert torch.equal(assignments(target.gate)[1], weights)
        assert torch.equal(torch.get_rng_state(), rng_before)


def test_identity_control_reproduces_logits_and_hook_is_removed_on_error():
    model = GPT(tiny_config(moe_impl="dense")).eval()
    tokens = torch.arange(12).reshape(2, 6)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    with torch.inference_mode():
        logits, _ = model(tokens)
        with routing_control(model, 1, valid, seed=None):
            assert torch.equal(model(tokens)[0], logits)
        with pytest.raises(ValueError, match="input batch"):
            with routing_control(model, 1, valid.T, seed=13):
                model(tokens)
        assert torch.equal(model(tokens)[0], logits)


def test_rejects_training_and_dense_layer():
    model = GPT(tiny_config())
    mask = torch.ones((1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="eval"):
        with routing_control(model, 1, mask, seed=7):
            pass
    with pytest.raises(ValueError, match="MoE block"):
        with routing_control(model.eval(), 0, mask, seed=7):
            pass


@pytest.mark.parametrize("labels", [
    [1, 0, 0, 1, 1, 0, 1, 0],
    [0] * 8,
    [1, 0, 0, 0, 0, 0, 0, 0],
])
def test_grouped_shuffle_preserves_exact_gate_rows_and_padding(labels):
    model = GPT(tiny_config(moe_impl="dense")).eval()
    gate = model.blocks[1].mlp.gate
    x = torch.arange(8 * model.cfg.n_embd).float().sin().reshape(8, -1)
    valid = torch.tensor([[True, False, True, True], [True, True, True, False]])
    group = torch.tensor(labels, dtype=torch.bool).reshape_as(valid)
    with torch.inference_mode():
        baseline = gate(x)
        with routing_control(model, 1, valid, 11):
            unrestricted = gate(x)
        with routing_control(model, 1, valid, 11, group):
            grouped = gate(x)
    # Recover donor identities by matching every logit of each gate row.
    matches = (unrestricted[:, None] == baseline[None, :]).all(-1)
    assert torch.all(matches.sum(1) == 1)
    donors = matches.int().argmax(1)[valid.flatten()]
    for member in (True, False):
        receivers = (valid & (group == member)).flatten()
        expected_donors = donors[group.flatten()[donors] == member]
        assert torch.equal(grouped[receivers], baseline[expected_donors])
    assert torch.equal(grouped[~valid.flatten()], baseline[~valid.flatten()])


@pytest.mark.parametrize("tied", [False, True])
def test_crossed_mixtures_take_ranked_ids_and_weights_from_their_own_donors(tied):
    model = GPT(tiny_config(moe_impl="dense")).eval()
    moe = model.blocks[1].mlp
    with torch.no_grad():
        # Use the first input coordinates as gate logits, with known ties.
        moe.gate.weight.zero_()
        for i in range(moe.cfg.n_experts):
            moe.gate.weight[i, i] = 1
        x = torch.randn(12, model.cfg.n_embd, generator=torch.Generator().manual_seed(42))
        if tied:
            x[::2] = 1
        valid = torch.tensor([[True, True, False, True, True, True]] * 2)
        group = torch.tensor([[True, False, True, False, True, False]] * 2)
        base = moe.gate(x)
        with routing_control(model, 1, valid, 11):
            unrestricted = moe.gate(x)
        with routing_control(model, 1, valid, 11, group):
            grouped = moe.gate(x)
        assert not torch.equal(unrestricted, grouped)
        for component, id_rows, weight_rows in (("ids", grouped, unrestricted),
                                                ("weights", unrestricted, grouped)):
            with routing_control(model, 1, valid, 11, group, preserve=component):
                actual = moe.gate(x)
            k = moe.cfg.n_experts_per_tok
            ids = id_rows.topk(k).indices
            weights = weight_rows.topk(k).values.softmax(-1)
            expected = torch.zeros_like(base).scatter(1, ids, weights)
            values, selected = actual.topk(k)
            mixture = torch.zeros_like(base).scatter(1, selected, values.softmax(-1))
            assert torch.equal(mixture, expected)
            assert torch.equal(actual[~valid.flatten()], base[~valid.flatten()])
