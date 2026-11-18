"""Inference-time intervention on one v3 MoE gate."""

from contextlib import contextmanager

import torch


@contextmanager
def routing_control(model, layer_index, valid_positions, seed):
    """Permute complete router rows across true positions of one CPU batch.

    layer_index is the absolute index in model.blocks. A None seed selects
    the identity permutation. Equal masks and seeds produce equal permutations.
    """
    if model.training:
        raise ValueError("routing control requires model.eval()")
    if not 0 <= layer_index < len(model.blocks) or not model.blocks[layer_index].is_moe:
        raise ValueError("layer_index must identify an MoE block")
    moe = model.blocks[layer_index].mlp
    if moe.gate.weight.device.type != "cpu":
        raise ValueError("routing control requires a CPU gate")
    if valid_positions.device.type != "cpu" or valid_positions.dtype != torch.bool or valid_positions.ndim != 2:
        raise ValueError("valid_positions must be a boolean CPU tensor of shape (batch, sequence)")

    positions = valid_positions.flatten().nonzero(as_tuple=True)[0]
    donors = positions
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        donors = positions[torch.randperm(positions.numel(), generator=generator)]

    def check_shape(module, args):
        if args[0].shape[:2] != valid_positions.shape:
            raise ValueError("input batch and sequence dimensions differ from the routing mask")

    def permute(module, args, logits):
        # top-k IDs and their mixture weights are both functions of this row.
        result = logits.clone()
        result[positions] = logits[donors]
        return result

    with moe.register_forward_pre_hook(check_shape), moe.gate.register_forward_hook(permute):
        yield
