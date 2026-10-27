"""PERF-5D rolling profiler around the historical reproducer.

One profiler cycle per training iteration (``repeat=0`` is the infinite repeat in
this build; ``repeat=-1`` is rejected), exported immediately and atomically
from ``on_trace_ready``, so the trace of the iteration that faults is already on
disk before the process dies.

The step boundary is the reproducer's own ``impl.get_batch`` call: it runs at
the top of every iteration, so ``prof.step()`` there closes the previous
iteration's cycle *before* the fetch that historically raises. No GPU
synchronization is added for logging, no serialization env var is set, and the
reproducer's own arguments, data and model are untouched.

Modes
    selftest   the same profiler lifecycle on a CPU-only workload; proves that
               cycles fire, files are published atomically and metadata lands
    repro      the GPU reproduction
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

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.path.join(HERE, "..", "..", "debug")
sys.path.insert(0, DEBUG)


class RollingExporter:
    """Publish each completed cycle to its own file, never overwriting."""

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
        except BaseException as exc:  # noqa: BLE001 - a latched fault may break export
            row["error"] = repr(exc)
            if os.path.exists(temporary):
                os.replace(temporary, final)
                row["path"] = final
                row["bytes"] = os.path.getsize(final)
        self.rows.append(row)
        self.state["cycles"] = self.rows
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
    """CPU-only: three steps, three cycles, three published files."""
    os.makedirs(out_dir, exist_ok=True)
    state = {"iter": 0, "cycles": []}
    exporter = RollingExporter(out_dir, state)
    steps_target = 3
    prof = profile(activities=[ProfilerActivity.CPU],
                   schedule=schedule(wait=0, warmup=0, active=1, repeat=0),
                   on_trace_ready=exporter, record_shapes=False, with_stack=False)
    with prof:
        for _ in range(steps_target):
            state["iter"] += 1
            a = torch.randn(64, 64, requires_grad=True)
            b = torch.randn(64, 64)
            (a @ b).sum().backward()
            prof.step()
    published = [r for r in exporter.rows if r.get("bytes")]
    payload = {"mode": "selftest", "steps": steps_target,
               "cycles": exporter.rows, "published": len(published),
               "all_nonempty": all(r["bytes"] > 0 for r in published),
               "covered_iterations": [r["covered_iteration"] for r in exporter.rows]}
    json.dump(payload, open(os.path.join(out_dir, "selftest.json"), "w"),
              indent=2, sort_keys=True)
    print("SELFTEST " + json.dumps({k: payload[k] for k in
                                    ("cycles", "published", "all_nonempty")})
          + f" ({len(payload['cycles'])} rows)")
    return 0


def repro(out_dir: str, repro_args) -> int:
    import reproducer
    import sparse_moe_v2_train as sparse_impl

    os.makedirs(out_dir, exist_ok=True)
    state = {"iter": 0, "cycles": [], "step_errors": []}
    exporter = RollingExporter(out_dir, state)
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule=schedule(wait=0, warmup=0, active=1, repeat=0),
                   on_trace_ready=exporter, record_shapes=False, with_stack=False)
    original_get_batch = sparse_impl.get_batch

    def hooked_get_batch(*args, **kwargs):
        state["iter"] += 1
        try:
            prof.step()
        except BaseException as exc:  # noqa: BLE001
            state["step_errors"].append({"iteration": state["iter"],
                                         "error": repr(exc)})
        return original_get_batch(*args, **kwargs)

    sparse_impl.get_batch = hooked_get_batch
    sys.argv = [os.path.join(DEBUG, "reproducer.py")] + list(repro_args)
    payload = {"mode": "repro", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime()),
               "serialize_kernel": os.environ.get("AMD_SERIALIZE_KERNEL"),
               "exception": None}
    exit_code = 0
    started = time.monotonic()
    try:
        with prof:
            reproducer.main()
    except BaseException as exc:  # noqa: BLE001 - the fault is expected
        payload["exception"] = {"type": type(exc).__name__,
                                "message": str(exc).splitlines()[0][:300]}
        exit_code = 1
    finally:
        sparse_impl.get_batch = original_get_batch
        payload["iterations_started"] = state["iter"]
        payload["cycles"] = state["cycles"]
        payload["step_errors"] = state["step_errors"]
        payload["elapsed_s"] = round(time.monotonic() - started, 3)
        try:
            payload["flush_error"] = None
            prof.stop()
        except BaseException as exc:  # noqa: BLE001
            payload["flush_error"] = repr(exc)
        json.dump(payload, open(os.path.join(out_dir, "rolling.json"), "w"),
                  indent=2, sort_keys=True)
    print("ROLLING " + json.dumps({"iterations_started": state["iter"],
                                   "cycles": len(state["cycles"]),
                                   "exception": payload["exception"],
                                   "step_errors": len(state["step_errors"])}),
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
