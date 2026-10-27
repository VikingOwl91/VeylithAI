"""PERF-5C attribution wrapper around the historical reproducer.

Two modes:

  plain     run the reproducer exactly as PERF-5B did (reference)
  profiled  run it under torch.profiler with CUDA activity so every kernel
            dispatch is recorded with its name and correlation id, and export
            the trace from a finally block. Meant to be combined with
            AMD_SERIALIZE_KERNEL=3 so only one dispatch is in flight and the
            last recorded kernel is the faulting one rather than one of several.

The reproducer is imported, not copied: its arguments and behaviour are
unchanged. Nothing here touches the model, the data or the training loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.path.join(HERE, "..", "..", "debug")
sys.path.insert(0, DEBUG)

import reproducer  # noqa: E402

KERNEL_KINDS = ("Cijk_", "silu", "elementwise", "reduce", "index", "scatter",
                "fill", "copy", "rocprim", "cat", "native", "attn")


def summarize_trace(path, tail=12):
    """Last kernel dispatches in launch order, with the op that launched them."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    events = data.get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == "kernel"]
    kernels.sort(key=lambda e: e.get("ts", 0))
    ops = {}
    for event in events:
        if event.get("cat") == "cpu_op":
            ext = (event.get("args") or {}).get("External id")
            if ext is not None:
                ops[ext] = event.get("name")
    counts = {}
    for event in kernels:
        counts[event.get("name", "?")] = counts.get(event.get("name", "?"), 0) + 1
    out = {
        "kernel_events": len(kernels),
        "distinct_kernel_names": len(counts),
        "top_names": sorted(counts.items(), key=lambda kv: -kv[1])[:6],
        "tail": [],
    }
    for event in kernels[-tail:]:
        ext = (event.get("args") or {}).get("External id")
        out["tail"].append({
            "name": event.get("name"),
            "ts": event.get("ts"),
            "dur": event.get("dur"),
            "external_id": ext,
            "launched_by": ops.get(ext),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plain", "profiled"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--repro-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    sys.argv = [os.path.join(DEBUG, "reproducer.py")] + list(args.repro_args)

    payload = {"mode": args.mode, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              time.gmtime()),
               "serialize_kernel": os.environ.get("AMD_SERIALIZE_KERNEL"),
               "hipblaslt_log_mask": os.environ.get("HIPBLASLT_LOG_MASK"),
               "exception": None, "trace": None, "summary": None}
    started = time.monotonic()
    exit_code = 0

    if args.mode == "plain":
        try:
            reproducer.main()
        except BaseException as exc:  # noqa: BLE001 - the fault is expected
            payload["exception"] = {"type": type(exc).__name__,
                                    "message": str(exc).splitlines()[0][:300]}
            exit_code = 1
    else:
        from torch.profiler import ProfilerActivity, profile

        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=False, with_stack=False, profile_memory=False)
        try:
            try:
                with prof:
                    reproducer.main()
            except BaseException as exc:  # noqa: BLE001
                payload["exception"] = {"type": type(exc).__name__,
                                        "message": str(exc).splitlines()[0][:300]}
                exit_code = 1
        finally:
            try:
                prof.stop()
            except BaseException:  # noqa: BLE001 - stop may fail after a device fault
                pass
            try:
                prof.export_chrome_trace(args.trace)
                payload["trace"] = args.trace
            except BaseException as exc:  # noqa: BLE001
                payload["trace_error"] = repr(exc)

    payload["elapsed_s"] = round(time.monotonic() - started, 3)
    if payload.get("trace") and os.path.exists(args.trace):
        try:
            payload["summary"] = summarize_trace(args.trace)
        except BaseException as exc:  # noqa: BLE001
            payload["summary_error"] = repr(exc)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("WRAPPER " + json.dumps({"mode": args.mode, "exception": payload["exception"],
                                  "trace": payload["trace"],
                                  "kernels": (payload["summary"] or {}).get("kernel_events")}),
          flush=True)
    if payload.get("summary"):
        for row in payload["summary"]["tail"][-5:]:
            print("TAIL " + json.dumps(row), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
