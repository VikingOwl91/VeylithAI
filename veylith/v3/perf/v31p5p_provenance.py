"""PERF-5P provenance instrumentation.

Two layers, both experimental and additive:

  layer 1  the CUDA allocator's own history (``_record_memory_history``), which
           records each allocation with its stacks and lifetime at C++ level and
           does not intercept operators. Snapshots are dumped periodically and
           from the fault handler so a SIGABRT cannot lose everything.
  layer 2  a ``TorchFunctionMode`` restricted to a small allowlist of Python
           level operators, recording the producer of every BF16 tensor of
           exactly 4,194,300 bytes, plus a gradient hook on those tensors so the
           backward tensor that feeds a GEMM is tagged as well.

Joins are keyed on (pointer, byte count, lifetime id). A pointer alone is never
enough: the PERF-5N record shows the fault address falling inside a storage that
an earlier, unrelated allocation had used.

Nothing here changes tensor storage, autograd behaviour or GEMM dispatch: the
mode calls the original function and only inspects the returned tensors.
"""

from __future__ import annotations

import json
import os
import time

import torch

TARGET_BYTES = 4194300
TARGET_DTYPE = torch.bfloat16
MAX_EVENTS = 20000
WATCH = frozenset({"linear", "silu", "mul", "mm", "addmm", "dropout", "_to_copy"})

ALLOW_CPU = os.environ.get("V31P5P_ALLOW_CPU") == "1"   # CPU gate only
_STATE = {"seq": 0, "phase": "forward", "enabled": False, "out_dir": None,
          "truncated": False, "events_path": None, "alloc_history": "off",
          "dumps": [], "write_error": False}


def _next_seq() -> int:
    _STATE["seq"] += 1
    return _STATE["seq"]


def _now() -> float:
    return round(time.time(), 6)


def is_target(t) -> bool:
    try:
        return ((t.is_cuda or ALLOW_CPU) and t.dtype == TARGET_DTYPE
                and t.numel() * t.element_size() == TARGET_BYTES)
    except Exception:  # noqa: BLE001
        return False


def _write(msg: dict) -> None:
    """Append one record and flush, so a SIGABRT keeps it."""
    if _STATE["out_dir"] is None:
        return
    path = os.path.join(_STATE["out_dir"], "prov-events.jsonl")
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(msg, sort_keys=True) + "\n")
            fh.flush()
    except OSError:
        # bookkeeping must never break the run being observed
        _STATE["write_error"] = True


def _emit(msg: dict) -> None:
    if _STATE["seq"] >= MAX_EVENTS:
        if not _STATE["truncated"]:
            _STATE["truncated"] = True
            _write({"kind": "truncated", "limit": MAX_EVENTS, "t": _now()})
        return
    msg.setdefault("seq", _next_seq())
    msg.setdefault("phase", _STATE["phase"])
    msg.setdefault("t", _now())
    _write(msg)


def note_tensor(t, producer: str, origin: str, extra: dict | None = None) -> dict:
    """Record a candidate tensor and hook its gradient if it has one."""
    rec = {"kind": "producer", "producer": producer, "origin": origin,
           "ptr": hex(t.data_ptr()), "nbytes": t.numel() * t.element_size(),
           "shape": list(t.shape), "dtype": str(t.dtype),
           "stride": list(t.stride()), "storage_offset": t.storage_offset(),
           "storage_ptr": hex(t.untyped_storage().data_ptr()),
           "storage_bytes": t.untyped_storage().nbytes(),
           "requires_grad": bool(t.requires_grad)}
    if extra:
        rec.update(extra)
    _emit(rec)
    if t.requires_grad:
        def _grad_hook(g):
            _emit({"kind": "producer", "producer": f"grad_of:{producer}",
                   "origin": "tensor_grad_hook", "ptr": hex(g.data_ptr()),
                   "nbytes": g.numel() * g.element_size(),
                   "shape": list(g.shape), "dtype": str(g.dtype),
                   "stride": list(g.stride()), "storage_offset": g.storage_offset(),
                   "storage_ptr": hex(g.untyped_storage().data_ptr()),
                   "storage_bytes": g.untyped_storage().nbytes(),
                   "requires_grad": False,
                   "grad_of_ptr": hex(t.data_ptr())})
        t.register_hook(_grad_hook)
    return rec


def note_consumer(name: str, t, role: str, out) -> None:
    _emit({"kind": "consumer", "op": name, "role": role,
           "ptr": hex(t.data_ptr()), "nbytes": t.numel() * t.element_size(),
           "out_shape": list(out.shape) if hasattr(out, "shape") else None})


def _iter_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)


class TagMode(torch.overrides.TorchFunctionMode):
    """Record producers and Python-level consumers of candidate tensors."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        if name in WATCH:
            for t in _iter_tensors(out):
                if is_target(t):
                    note_tensor(t, f"{getattr(func, '__module__', '?')}.{name}",
                                "torch_function_mode")
            for t in _iter_tensors(args):
                if is_target(t):
                    note_consumer(name, t, "arg", out)
        return out


class Phase:
    """Forward/backward phase flag driven by module hooks, not by interception."""

    def __init__(self, impl):
        self.impl = impl
        self._orig_forward = impl.GPT.forward
        self._orig_init = impl.GPT.__init__

    def __enter__(self):
        impl = self.impl
        orig_forward, orig_init = self._orig_forward, self._orig_init

        def forward(self, *a, **k):
            _STATE["phase"] = "forward"
            return orig_forward(self, *a, **k)

        def init(self, *a, **k):
            orig_init(self, *a, **k)
            self.register_full_backward_pre_hook(
                lambda *_: _STATE.__setitem__("phase", "backward"))

        impl.GPT.forward = forward
        impl.GPT.__init__ = init
        return self

    def __exit__(self, *exc):
        self.impl.GPT.forward = self._orig_forward
        self.impl.GPT.__init__ = self._orig_init
        return False


def capture_first_loss(impl, store: dict):
    """Capture the first forward loss at full precision, once.

    The reproducer already reads ``loss.item()`` for its display at iteration 0,
    so this adds no new synchronisation or transfer; it only keeps the scalar
    instead of formatting it to four decimals.
    """
    original = impl.GPT.forward

    def forward(self, *a, **k):
        out = original(self, *a, **k)
        if "first_loss" not in store:
            try:
                loss = out[1] if isinstance(out, tuple) else None
                if loss is not None:
                    value = float(loss.detach())
                    store["first_loss"] = value
                    print("FIRSTLOSS " + json.dumps(
                        {"iter0_loss": repr(value),
                         "iter0_loss_4dp": f"{value:.4f}"}), flush=True)
            except Exception as exc:  # noqa: BLE001
                store["first_loss_error"] = f"{type(exc).__name__}: {exc}"
        return out

    impl.GPT.forward = forward
    return original


def enable_alloc_history() -> str:
    """Turn on the allocator's own history; returns its status."""
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="all",
            max_entries=50000, clear_history=False)
        _STATE["alloc_history"] = "recording"
    except Exception as exc:  # noqa: BLE001
        _STATE["alloc_history"] = f"unavailable: {type(exc).__name__}: {exc}"
    _emit({"kind": "alloc_history", "status": _STATE["alloc_history"]})
    return _STATE["alloc_history"]


def dump_snapshot(tag: str) -> str | None:
    """Dump the allocator snapshot; bounded by two rotating files."""
    if _STATE["out_dir"] is None:
        return None
    path = os.path.join(_STATE["out_dir"], f"prov-snap-{tag}.pickle")
    try:
        torch.cuda.memory._dump_snapshot(path)
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        path, status = None, f"unavailable: {type(exc).__name__}: {exc}"
    _STATE["dumps"].append({"tag": tag, "path": path, "status": status, "t": _now()})
    _emit({"kind": "snapshot", "tag": tag, "path": path, "status": status})
    return path


def enable(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _STATE.update(enabled=True, out_dir=out_dir,
                  events_path=os.path.join(out_dir, "prov-events.jsonl"))
    _emit({"kind": "start", "pid": os.getpid(), "torch": torch.__version__})
    enable_alloc_history()


def finish() -> None:
    _emit({"kind": "finish", "dumps": _STATE["dumps"],
           "alloc_history": _STATE["alloc_history"],
           "truncated": _STATE["truncated"], "events": _STATE["seq"]})


# --------------------------------------------------------------- join logic

def resolve(records, ptr: str, nbytes: int, at: float):
    """Pick the allocation whose lifetime covers ``at`` for (ptr, nbytes).

    ``records`` is any iterable of dicts with ``ptr``, ``nbytes``,
    ``alloc_t`` and optional ``free_t``. Returns (match, status) where status is
    ``observed``, ``ambiguous`` or ``unknown``. A pointer alone never decides:
    the byte count must agree and exactly one lifetime must cover the time.
    """
    cands = [r for r in records
             if r.get("ptr") == ptr and int(r.get("nbytes", -1)) == nbytes]
    if not cands:
        return None, "unknown"
    covering = [r for r in cands
                if float(r.get("alloc_t", 0.0)) <= at
                and (r.get("free_t") is None or at <= float(r["free_t"]))]
    if len(covering) == 1:
        return covering[0], "observed"
    if len(covering) > 1:
        return None, "ambiguous"
    return None, "unknown"
