"""Precision context for CPU reference and future GPU bf16 runs."""

from __future__ import annotations

import contextlib

import torch


def autocast_context(device: str, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        if device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        raise ValueError("bf16 is only supported on CUDA in this trainer")
    raise ValueError(f"unknown precision: {precision}")
