"""Runtime policy for the deterministic uninitialized-memory prefill.

``torch.use_deterministic_algorithms(True)`` enables
``torch.utils.deterministic.fill_uninitialized_memory``, which makes every fresh
allocation cost an extra NaN fill of the whole tensor. In this model that was
11,545 fill kernels and 45.9 ms of device time per update, 25.7 percent of all
launches (PERF-4A), for buffers the model then overwrites.

The prefill is a debugging aid, not a numerical contract: it turns a read of
unwritten device memory into a loud NaN. PERF-4B and PERF-4C verified that
disabling it leaves every computed value bitwise identical, over 150 resumed
updates with non-zero AdamW moments.

Policy:

    unset / "0" / ""   prefill disabled (default for training runs)
    "1"                prefill enabled (debug)

Any other value raises ``ValueError`` instead of being interpreted. The setting
is a runtime policy, not a model configuration field: it never enters the
configuration fingerprint and does not affect resume compatibility.

Deterministic algorithms and their operator checks stay enabled either way.
"""

from __future__ import annotations

import functools
import os

import torch

PREFILL_ENV = "VEYLITH_DETERMINISTIC_PREFILL"
_ENABLED = frozenset({"1"})
_DISABLED = frozenset({"0", ""})


def parse_prefill_policy(value):
    """Map a raw environment value to a boolean, refusing anything ambiguous."""
    if value is None:
        return False
    if value in _ENABLED:
        return True
    if value in _DISABLED:
        return False
    raise ValueError(
        f"{PREFILL_ENV} must be '1' to enable the deterministic prefill or "
        f"'0' to disable it (unset means disabled); got {value!r}")


def resolve_prefill_policy(environ=None):
    source = os.environ if environ is None else environ
    raw = source.get(PREFILL_ENV)
    policy = {
        "enabled": parse_prefill_policy(raw),
        "env_value": raw,
        "source": "default" if raw in (None, "") else "environment",
    }
    return policy


def current_prefill_state() -> bool:
    return bool(torch.utils.deterministic.fill_uninitialized_memory)


def apply_prefill_policy(policy) -> bool:
    """Set the process-global flag; return the previous value."""
    previous = current_prefill_state()
    torch.utils.deterministic.fill_uninitialized_memory = bool(policy["enabled"])
    return previous


def restore_prefill_state(previous: bool) -> None:
    torch.utils.deterministic.fill_uninitialized_memory = bool(previous)


def restores_prefill_state(func):
    """Restore the process-global flag after the call, including on exception."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        previous = current_prefill_state()
        try:
            return func(*args, **kwargs)
        finally:
            restore_prefill_state(previous)

    return wrapper


def describe_prefill_policy(policy) -> str:
    state = "enabled" if policy["enabled"] else "disabled"
    if policy["source"] == "default":
        origin = f"{PREFILL_ENV} unset"
    else:
        origin = f"{PREFILL_ENV}={policy['env_value']!r}"
    return f"deterministic prefill {state} ({origin})"
