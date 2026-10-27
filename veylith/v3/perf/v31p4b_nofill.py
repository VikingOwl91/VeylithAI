"""v3.1-PERF-4B scoped shim for the deterministic uninitialized-memory prefill.

``torch.use_deterministic_algorithms(True)`` sets
``torch.utils.deterministic.fill_uninitialized_memory``, which turns every
``empty``/``empty_like`` into an allocation plus a NaN fill of the whole tensor.
PERF-4A measured 11,545 such fill kernels and 43.8 ms of device time per update
in this model. The fills write NaN into buffers that the model then overwrites,
so they are pure overhead unless something reads uninitialized memory.

Two access paths:

- ``prefill_disabled()`` scoped context manager, for harnesses that control the
  order of the deterministic-mode call themselves.
- ``DeterminismShim``, which wraps ``torch.use_deterministic_algorithms`` so the
  production loop (``training.train.train`` calls it internally at train.py:142)
  runs with kernel determinism on and the prefill off. Restores both the wrapper
  and the flag on exit.

Neither path touches kernel determinism, RNG, the optimizer, or any computed
value.
"""

from __future__ import annotations

import contextlib

import torch

GIB = 2 ** 30


def deterministic_state() -> dict:
    """Snapshot of the two settings this experiment cares about."""
    return {
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "fill_uninitialized_memory": bool(
            torch.utils.deterministic.fill_uninitialized_memory),
    }


@contextlib.contextmanager
def prefill_disabled():
    """Disable the prefill, restoring the previous value on exit."""
    previous = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        yield previous
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


class DeterminismShim:
    """Keep deterministic algorithms on, clear the prefill right after it is set.

    The training loop enables deterministic mode itself, which re-enables the
    prefill; wrapping that call is the only way to disable just the prefill while
    leaving the production path unmodified.
    """

    def __init__(self) -> None:
        self._original = None
        self._previous_prefill = None
        self.intercepted = 0

    def __enter__(self):
        self._previous_prefill = torch.utils.deterministic.fill_uninitialized_memory
        self._original = torch.use_deterministic_algorithms
        original = self._original
        shim = self

        def wrapped(mode=True, *, warn_only=False):
            original(mode, warn_only=warn_only)
            if mode:
                torch.utils.deterministic.fill_uninitialized_memory = False
                shim.intercepted += 1

        torch.use_deterministic_algorithms = wrapped
        return self

    def __exit__(self, *exc):
        if self._original is not None:
            torch.use_deterministic_algorithms = self._original
        if self._previous_prefill is not None:
            torch.utils.deterministic.fill_uninitialized_memory = self._previous_prefill
        return False


def prime_allocator(pattern: int, gib: float) -> dict:
    """Fill a sized allocation with a byte pattern, free it, keep it in the pool.

    The caching allocator hands those bytes back on the next allocations of
    similar size, so any uninitialized read becomes visible as a dependence on
    the pattern. ``fill_`` on a byte tensor is exact, and ``del`` returns the
    block without zeroing it.
    """
    if not 0 <= int(pattern) <= 255:
        raise ValueError("pattern must be a byte value 0..255")
    if gib <= 0:
        raise ValueError("gib must be positive")
    nbytes = int(gib * GIB)
    buffer = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    buffer.fill_(int(pattern))
    torch.cuda.synchronize()
    del buffer
    return {"pattern": hex(int(pattern)), "bytes": nbytes,
            "gib": nbytes / GIB}
