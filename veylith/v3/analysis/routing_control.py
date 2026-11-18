"""Inference-time intervention on one v3 MoE gate."""

from contextlib import contextmanager

import torch


@contextmanager
def routing_control(model, layer_index, valid_positions, seed, group_mask=None, *, preserve="both"):
    """Permute complete router rows across true positions of one CPU batch.

    layer_index is the absolute index in model.blocks. A None seed selects
    the identity permutation. An optional boolean group_mask restricts donors
    to the recipient's group, using the same shuffled donor order. With a
    group mask, preserve="ids" or "weights" restricts only that component.
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
    if group_mask is not None and (group_mask.shape != valid_positions.shape
                                   or group_mask.dtype != torch.bool
                                   or group_mask.device != valid_positions.device):
        raise ValueError("group_mask must match the boolean CPU routing mask")
    if preserve not in ("both", "ids", "weights") or (preserve != "both" and group_mask is None):
        raise ValueError("crossing IDs and weights requires a group mask and a valid preserve mode")
    donors = positions
    grouped = positions
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        donors = positions[torch.randperm(positions.numel(), generator=generator)]
        if group_mask is not None:
            group = group_mask.flatten()
            grouped = torch.empty_like(donors)
            for member in (False, True):
                grouped[group[positions] == member] = donors[group[donors] == member]

    def check_shape(module, args):
        if args[0].shape[:2] != valid_positions.shape:
            raise ValueError("input batch and sequence dimensions differ from the routing mask")

    def permute(module, args, logits):
        result = logits.clone()
        if preserve == "both" or seed is None:
            result[positions] = logits[donors if group_mask is None else grouped]
        else:
            id_donors, weight_donors = (grouped, donors) if preserve == "ids" else (donors, grouped)
            # Reconstruct exactly the top-k mixture used for prediction. Equal
            # top scores give equal weights regardless of their internal order.
            scores, ids = logits.float().topk(moe.cfg.n_experts_per_tok, dim=-1)
            crossed = torch.full_like(logits[positions], -torch.inf)
            crossed.scatter_(1, ids[id_donors], scores[weight_donors].to(logits.dtype))
            result[positions] = crossed
        return result

    with moe.register_forward_pre_hook(check_shape), moe.gate.register_forward_hook(permute):
        yield
