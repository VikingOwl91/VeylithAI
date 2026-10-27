"""v3.1-PERF-3D experimental SwiGLU forward fusion.

One Triton kernel computes t = BF16(silu(a)) and y = BF16(t * b) with the
original FP32 intermediate arithmetic and the original BF16 rounding points.
The backward is not fused: it calls the same ATen mul and silu_backward ops as
the baseline. Applied to an experimental model by instance-level patching; the
production model is unchanged.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from architecture.model import SwiGLU

BLOCK = 1024


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, t_ptr, y_ptr, n_elements,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    a_f = a.to(tl.float32)
    denominator = 1.0 + tl.exp(-a_f)
    t_f = tl.div_rn(a_f, denominator)
    t = t_f.to(tl.bfloat16)
    tl.store(t_ptr + offsets, t, mask=mask)
    y = (t.to(tl.float32) * b.to(tl.float32)).to(tl.bfloat16)
    tl.store(y_ptr + offsets, y, mask=mask)


class _no_uninitialized_fill:
    """Skip the NaN prefill that deterministic mode adds to fresh allocations.

    ``torch.use_deterministic_algorithms(True)`` sets
    ``fill_uninitialized_memory``, so every ``empty_like`` costs an extra
    device-side fill of the whole tensor. Both buffers below are written in
    full by the kernel (the grid covers ``numel``), so the fill is
    unobservable and only burns bandwidth. The flag is process-global, hence
    save/restore in ``finally``. Gate 1 and Gate 2 re-verify bitwise identity
    with this path active.
    """

    def __enter__(self):
        self.previous = torch.utils.deterministic.fill_uninitialized_memory
        torch.utils.deterministic.fill_uninitialized_memory = False

    def __exit__(self, *exc):
        torch.utils.deterministic.fill_uninitialized_memory = self.previous
        return False


def fused_silu_mul(a: torch.Tensor, b: torch.Tensor):
    """Return (y, t); t is the BF16 intermediate saved for backward."""
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("fused_silu_mul requires bfloat16 inputs")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("fused_silu_mul requires CUDA/HIP tensors")
    if a.shape != b.shape:
        raise ValueError("fused_silu_mul requires identical shapes")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("fused_silu_mul requires contiguous inputs")
    n_elements = a.numel()
    with _no_uninitialized_fill():
        t = torch.empty_like(a)
        y = torch.empty_like(a)
    if n_elements == 0:
        return y, t
    grid = (triton.cdiv(n_elements, BLOCK),)
    _silu_mul_kernel[grid](a, b, t, y, n_elements, BLOCK=BLOCK)
    # Every element is written: grid * BLOCK >= n_elements and the mask only
    # trims the tail, so no uninitialized byte reaches the caller.
    return y, t


class FusedSiluMul(torch.autograd.Function):
    """Forward fused; backward uses the exact ATen ops of the baseline."""

    @staticmethod
    def forward(ctx, a, b):
        y, t = fused_silu_mul(a, b)
        ctx.save_for_backward(a, t, b)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        a, t, b = ctx.saved_tensors
        grad_t = torch.mul(grad_output, b)
        grad_b = torch.mul(grad_output, t)
        grad_a = torch.ops.aten.silu_backward(grad_t, a)
        return grad_a, grad_b


def swiglu_forward_fused(self, x):
    y = FusedSiluMul.apply(self.w1(x), self.w2(x))
    return self.drop(self.w3(y))


def apply_variant(model):
    patched = {"swiglu": 0}
    for module in model.modules():
        if isinstance(module, SwiGLU):
            module.forward = types.MethodType(swiglu_forward_fused, module)
            patched["swiglu"] += 1
    return patched
