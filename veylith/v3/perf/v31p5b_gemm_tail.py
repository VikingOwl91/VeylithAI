"""PERF-5B single-GEMM K-tail case (one process, one execution).

Builds the call that the archived 16-expert/top-4 capture showed
(`transA=N transB=T`, Adesc rows=256 cols=4108 ld=256, Bdesc rows=682 cols=4108
ld=682, bf16 in, C 256x682) by using transposed views for both operands, then
places the B operand inside a controlled byte buffer so that either

  tight     the buffer ends 16 bytes after B, at a 2 MiB boundary, or
  generous  B starts at offset 0, leaving >= 64 KiB of slack,

and runs one to three GEMMs. Everything needed for attribution is printed before
the first GEMM and flushed, so a crash still leaves the descriptors, the pointer
arithmetic and the predicted boundary behind.

No model, no training, no checkpoints, no allocator tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

MIB = 1024 * 1024
SEGMENT_ALIGN = 2 * MIB


def allocation_plan(n: int, k: int, itemsize: int, placement: str) -> dict:
    """Choose a 2 MiB-aligned buffer and a 512-byte-aligned offset for B.

    The buffer is a whole number of 2 MiB units, so its end (and therefore the
    end of the mapping, assuming a 2 MiB-aligned base) lands on a 2 MiB
    boundary. With ``tight`` the offset is the largest 512-byte multiple that
    leaves the smallest possible trailing slack, which for these shapes is 16
    bytes, matching the archived capture.
    """
    b_bytes = n * k * itemsize
    segment = ((b_bytes + 16 + SEGMENT_ALIGN - 1) // SEGMENT_ALIGN) * SEGMENT_ALIGN
    if placement == "tight":
        offset = ((segment - b_bytes - 16) // 512) * 512
    elif placement == "generous":
        offset = 0
    else:
        raise ValueError(f"unknown placement: {placement}")
    if offset % 512:
        raise ValueError("B offset is not 512-byte aligned")
    if offset + b_bytes > segment:
        raise ValueError("B does not fit into the buffer")
    return {"b_bytes": b_bytes, "segment_bytes": segment, "b_offset": offset,
            "slack_bytes": segment - offset - b_bytes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=682)
    parser.add_argument("--k", type=int, default=4108)
    parser.add_argument("--placement", choices=("tight", "generous"), default="tight")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--backend", choices=("default", "rocblas"), default="default")
    parser.add_argument("--orientation", choices=("m-first", "boundary-first"),
                        default="m-first",
                        help="m-first puts the 256-row operand first; boundary-first "
                             "puts the controlled 682-row operand first so the kernel "
                             "sees it as its B operand")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    itemsize = 2 if args.dtype == "bf16" else 4
    plan = allocation_plan(args.n, args.k, itemsize, args.placement)

    if args.backend == "rocblas":
        torch.backends.cuda.preferred_blas_library("rocblas")

    torch.cuda.init()
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    # Order matters. Warm the backend first so hipBLASLt's workspace (76 MB in
    # this build) is allocated before the controlled buffer, then the operands,
    # and only then the buffer itself, so it is the last segment in the arena
    # and the bytes after it are most likely unmapped. Nothing may allocate
    # after it: the GEMM writes into the preallocated output.
    warm_a = torch.empty(64, 64, dtype=dtype, device="cuda")
    warm_b = torch.empty(64, 64, dtype=dtype, device="cuda")
    torch.mm(warm_a, warm_b)
    torch.cuda.synchronize()
    del warm_a, warm_b
    alloc_before = torch.cuda.memory_allocated()

    a_base = torch.empty(args.k, args.m, dtype=dtype, device="cuda")
    a_operand = a_base.t()
    a_base.normal_(0.0, 1.0)
    if args.orientation == "m-first":
        c = torch.empty(args.m, args.n, dtype=dtype, device="cuda")
    else:
        c = torch.empty(args.n, args.m, dtype=dtype, device="cuda")

    storage = torch.empty(plan["segment_bytes"], dtype=torch.uint8, device="cuda")
    base = storage.data_ptr()
    end = base + plan["segment_bytes"]

    # B operand: N x K stored N-major (ld = N), as the archived capture shows.
    # A transposed view of a (K, N) row-major block is exactly that layout.
    b_block = storage[plan["b_offset"]:plan["b_offset"] + plan["b_bytes"]]
    b_rows = b_block.view(dtype).view(args.k, args.n)
    b_operand = b_rows.t()
    b_rows.normal_(0.0, 1.0)  # touch every page, including the last

    prep = {
        "event": "prep",
        "dtype": args.dtype, "backend": args.backend,
        "orientation": args.orientation,
        "m": args.m, "n": args.n, "k": args.k,
        "n_mod_64": args.n % 64, "k_mod_64": args.k % 64,
        "predicted_overread_bytes": 2 * (64 - args.n % 64)
        if (args.n % 64 and args.k % 64) else 0,
        "placement": args.placement,
        "b_bytes": plan["b_bytes"], "segment_bytes": plan["segment_bytes"],
        "b_offset": plan["b_offset"], "slack_bytes": plan["slack_bytes"],
        "storage_ptr": base, "storage_end_ptr": end,
        "storage_2mib_aligned": base % SEGMENT_ALIGN == 0,
        "end_2mib_aligned": end % SEGMENT_ALIGN == 0,
        "b_ptr": b_operand.data_ptr(),
        "b_end_exact_ptr": b_operand.data_ptr() + plan["b_bytes"],
        "predicted_fault_address": end,
        "storage_nbytes": storage.untyped_storage().nbytes(),
        "allocated_before_storage": alloc_before,
        "free_gib": round(free_bytes / 2 ** 30, 3),
        "total_gib": round(total_bytes / 2 ** 30, 3),
        "torch": torch.__version__, "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("PREP " + json.dumps(prep, sort_keys=True), flush=True)

    started = time.monotonic()
    calls = []
    for index in range(args.iters):
        if args.orientation == "m-first":
            c = torch.mm(a_operand, b_operand.t(), out=c)
        else:
            c = torch.mm(b_operand, a_operand.t(), out=c)
        torch.cuda.synchronize()
        calls.append({"call": index, "checksum": float(c.float().sum()),
                      "elapsed_s": round(time.monotonic() - started, 4)})
        print("CALL " + json.dumps(calls[-1], sort_keys=True), flush=True)

    result = dict(prep)
    result.update({"event": "result", "calls": calls,
                   "elapsed_s": round(time.monotonic() - started, 4),
                   "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                   "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)})
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("RESULT " + json.dumps({"checksum": calls[-1]["checksum"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
