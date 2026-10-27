"""PERF-5E tensor observation around the faulting iteration.

Two additions over the PERF-5D rolling profiler, both metadata-only:

  * ``record_shapes=True`` so each aten event carries input shapes
  * a ``TorchDispatchMode`` that records shape, stride, dtype, storage offset,
    data pointer, storage base and storage size for ``aten::mm``, ``aten::copy_``
    and ``aten::_to_copy``, one JSON line per op, flushed immediately so the
    sequence survives a SIGABRT

Tensors are never retained and lifetimes are never extended: only metadata is
read, and the op is dispatched normally. No synchronization is added, no
serialization env var is used.

Modes
    selftest   CPU-only: one tiny mm plus a dtype copy, proving the recorder
               writes and flushes
    repro      the GPU reproduction with both the rolling profiler and the
               recorder
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.path.join(HERE, "..", "..", "debug")
sys.path.insert(0, DEBUG)


def describe(tensor):
    storage = tensor.untyped_storage()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": int(tensor.storage_offset()),
        "data_ptr": int(tensor.data_ptr()),
        "storage_ptr": int(storage.data_ptr()),
        "storage_nbytes": int(storage.nbytes()),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
    }


class OpRecorder(TorchDispatchMode):
    """Metadata-only record of the ops that touch GEMM outputs."""

    MATCH = ("aten.mm.", "aten.copy_.", "aten._to_copy.", "aten.to.")

    def __init__(self, path: str):
        self.path = path
        self.handle = open(path, "w", encoding="utf-8")
        self.seq = 0

    def _record(self, name, args, kwargs, out):
        row = {"seq": self.seq, "op": name, "utc_ns": time.time_ns()}
        self.seq += 1
        for index, value in enumerate(args[:3]):
            if isinstance(value, torch.Tensor):
                row[f"in{index}"] = describe(value)
        if isinstance(out, torch.Tensor):
            row["out"] = describe(out)
        self.handle.write(json.dumps(row, sort_keys=True) + "\n")
        # flush every record so an abort cannot lose the tail; fsync only
        # occasionally, since process death does not lose flushed data and a
        # per-record fsync would stall the run
        self.handle.flush()
        if self.seq % 200 == 0:
            try:
                os.fsync(self.handle.fileno())
            except OSError:
                pass

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        out = func(*args, **kwargs)
        if name.startswith(self.MATCH):
            self._record(name, args, kwargs, out)
        return out


class RollingExporter:
    def __init__(self, out_dir: str, state: dict):
        self.out_dir = out_dir
        self.state = state
        self.counter = itertools.count(1)
        self.rows = []

    def __call__(self, prof):
        cycle = next(self.counter)
        final = os.path.join(self.out_dir, f"trace-cycle-{cycle:04d}.json")
        temporary = final + ".tmp"
        row = {"cycle": cycle, "covered_iteration": self.state["iter"] - 1,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            prof.export_chrome_trace(temporary)
            os.replace(temporary, final)
            row["path"] = final
            row["bytes"] = os.path.getsize(final)
        except BaseException as exc:  # noqa: BLE001
            row["error"] = repr(exc)
            if os.path.exists(temporary):
                os.replace(temporary, final)
                row["path"] = final
                row["bytes"] = os.path.getsize(final)
        self.rows.append(row)
        try:
            with open(os.path.join(self.out_dir, "cycles.jsonl"), "a",
                      encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            print("CYCLE_META_ERROR " + repr(exc), flush=True)
        print("CYCLE " + json.dumps(row), flush=True)


def selftest(out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    state = {"iter": 0}
    exporter = RollingExporter(out_dir, state)
    ops_path = os.path.join(out_dir, "ops.jsonl")
    prof = profile(activities=[ProfilerActivity.CPU],
                   schedule=schedule(wait=0, warmup=0, active=1, repeat=0),
                   on_trace_ready=exporter, record_shapes=True, with_stack=False)
    with prof, OpRecorder(ops_path):
        for _ in range(2):
            state["iter"] += 1
            a = torch.randn(32, 16, requires_grad=True)
            b = torch.randn(16, 8)
            c = torch.mm(a, b)
            _ = c.to(torch.float64)
            prof.step()
    rows = [json.loads(line) for line in open(ops_path, encoding="utf-8")]
    payload = {"mode": "selftest", "ops": len(rows),
               "op_names": sorted({r["op"] for r in rows}),
               "has_mm": any(r["op"].startswith("aten.mm") for r in rows),
               "has_copy": any(r["op"].startswith(("aten.copy_", "aten._to_copy"))
                               for r in rows),
               "cycles": len(exporter.rows),
               "published": sum(1 for r in exporter.rows if r.get("bytes")),
               "sample": rows[:2]}
    json.dump(payload, open(os.path.join(out_dir, "selftest.json"), "w"),
              indent=2, sort_keys=True)
    print("SELFTEST " + json.dumps({k: payload[k] for k in
                                    ("ops", "has_mm", "has_copy", "cycles", "published")}))
    return 0


def repro(out_dir: str, repro_args) -> int:
    import reproducer
    import sparse_moe_v2_train as sparse_impl

    os.makedirs(out_dir, exist_ok=True)
    state = {"iter": 0}
    exporter = RollingExporter(out_dir, state)
    ops_path = os.path.join(out_dir, "ops.jsonl")
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule=schedule(wait=0, warmup=0, active=1, repeat=0),
                   on_trace_ready=exporter, record_shapes=True, with_stack=False)
    original_get_batch = sparse_impl.get_batch

    def hooked_get_batch(*args, **kwargs):
        state["iter"] += 1
        try:
            prof.step()
        except BaseException:  # noqa: BLE001
            pass
        return original_get_batch(*args, **kwargs)

    sparse_impl.get_batch = hooked_get_batch
    sys.argv = [os.path.join(DEBUG, "reproducer.py")] + list(repro_args)
    payload = {"mode": "repro", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime()),
               "record_shapes": True, "serialization": None}
    exit_code = 0
    started = time.monotonic()
    try:
        with prof, OpRecorder(ops_path):
            reproducer.main()
    except BaseException as exc:  # noqa: BLE001
        payload["exception"] = {"type": type(exc).__name__,
                                "message": str(exc).splitlines()[0][:300]}
        exit_code = 1
    finally:
        sparse_impl.get_batch = original_get_batch
        payload["iterations_started"] = state["iter"]
        payload["cycles"] = exporter.rows
        payload["elapsed_s"] = round(time.monotonic() - started, 3)
        json.dump(payload, open(os.path.join(out_dir, "rolling.json"), "w"),
                  indent=2, sort_keys=True)
    print("ROLLING " + json.dumps({"iterations_started": state["iter"],
                                   "cycles": len(exporter.rows),
                                   "exception": payload.get("exception")}),
          flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selftest", "repro"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repro-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.mode == "selftest":
        return selftest(args.out_dir)
    return repro(args.out_dir, args.repro_args)


if __name__ == "__main__":
    raise SystemExit(main())
