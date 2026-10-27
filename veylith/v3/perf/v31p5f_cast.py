"""PERF-5F case B: a standalone BF16-to-FP32 cast on the boundary shape.

Allocates exactly one 682 x 3075 bfloat16 tensor, which PyTorch rounds to a
4 MiB block ending 4 bytes after the tensor, i.e. the tensor end sits 4 bytes
before a 2 MiB-aligned allocation end. Fills it, then casts it to float32 once
and synchronizes.

Geometry is printed before the cast and the cast kernel name is captured with a
one-op profiler, so the record shows whether the isolated consumer actually ran
and what it was. Nothing else runs in the process.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

MIB = 2 << 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    torch.cuda.init()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    tensor = torch.empty(682, 3075, dtype=torch.bfloat16, device="cuda")
    storage = tensor.untyped_storage()
    base = tensor.data_ptr()
    tensor_bytes = tensor.numel() * tensor.element_size()
    storage_bytes = storage.nbytes()
    tensor_end = base + tensor_bytes
    storage_end = storage.data_ptr() + storage_bytes
    next_page = ((tensor_end + MIB - 1) // MIB) * MIB

    prep = {
        "event": "prep", "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "contiguous": bool(tensor.is_contiguous()),
        "data_ptr": hex(base), "storage_ptr": hex(storage.data_ptr()),
        "tensor_bytes": tensor_bytes, "storage_bytes": storage_bytes,
        "tensor_end": hex(tensor_end), "storage_end": hex(storage_end),
        "slack_in_block": storage_end - tensor_end,
        "next_2mib_page_from_tensor_end": hex(next_page),
        "gap_to_next_page": next_page - tensor_end,
        "storage_end_is_2mib_aligned": storage_end % MIB == 0,
        "free_gib": round(free_bytes / 2 ** 30, 3),
        "total_gib": round(total_bytes / 2 ** 30, 3),
        "torch": torch.__version__, "hip": torch.version.hip,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("PREP " + json.dumps(prep, sort_keys=True), flush=True)

    tensor.normal_(0.0, 1.0)  # touch every page of the tensor, including the last

    from torch.profiler import ProfilerActivity, profile

    calls = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for index in range(args.repeats):
            out = tensor.to(torch.float32)
            torch.cuda.synchronize()
            calls.append({"call": index, "out_dtype": str(out.dtype),
                          "checksum": float(out.mean())})
            print("CALL " + json.dumps(calls[-1], sort_keys=True), flush=True)
        kernels = [event.key for event in prof.key_averages()
                   if "bfloat16tofloat32" in event.key or "copy" in event.key]
        cast_events = [{"key": event.key, "count": event.count,
                        "self_cuda_ms": round(event.self_device_time_total, 3)}
                       for event in prof.key_averages()
                       if "bfloat16tofloat32" in event.key]

    payload = dict(prep)
    payload.update({"event": "result", "calls": calls, "cast_kernels": cast_events,
                    "copy_like_keys": kernels[:6],
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)})
    json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=2, sort_keys=True)
    print("RESULT " + json.dumps({"gap_to_next_page": payload["gap_to_next_page"],
                                  "storage_end_aligned": payload["storage_end_is_2mib_aligned"],
                                  "cast_kernels": [k["key"][:60] for k in cast_events]}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
