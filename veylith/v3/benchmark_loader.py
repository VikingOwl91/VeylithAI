#!/usr/bin/env python3
"""CPU-only loader benchmark for a materialized v3 corpus.

Measures plan construction, structural verification, batch throughput and peak
RSS, and checks that a mid-epoch resume reproduces the saved position.

    python3 veylith/v3/benchmark_loader.py --root ~/veylith-v3-data/fineweb-10m-20261026
"""

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import dataset as ds  # noqa: E402
from data_pipeline import loader  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CPU loader benchmark")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train", choices=list(ds.SPLITS))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=200, help="measured batches")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.batches <= 0 or args.warmup < 0:
        raise SystemExit("--batches must be positive and --warmup non-negative")

    start = time.time()
    plan = loader.plan_sequences(args.root, split=args.split,
                                 seq_len=args.seq_len, mode="concat")
    plan_seconds = time.time() - start
    problems = loader.verify_plan(plan)
    if problems:
        print("plan is invalid: " + "; ".join(problems), file=sys.stderr)
        return 1

    total = args.warmup + args.batches
    start = time.time()
    tokens = 0
    measured = 0
    last_digest = None
    for index, batch in enumerate(loader.iter_batches(
            plan, args.root, batch_size=args.batch_size, seed=args.seed,
            max_batches=total)):
        if index < args.warmup:
            continue
        tokens += sum(batch["lengths"])
        measured += 1
        last_digest = batch["order_digest"]
    wall = time.time() - start
    usage = resource.getrusage(resource.RUSAGE_SELF)

    full = list(loader.iter_batches(plan, args.root, batch_size=args.batch_size,
                                    seed=args.seed, max_batches=args.warmup + 5))
    tail = list(loader.iter_batches(plan, args.root, batch_size=args.batch_size,
                                    seed=args.seed, max_batches=5,
                                    start_batch=args.warmup))
    resume_equal = [batch["indices"] for batch in tail] == [
        batch["indices"] for batch in full[args.warmup:]]

    payload = {
        "root": args.root,
        "split": args.split,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "plan_sequences": len(plan["sequences"]),
        "plan_tokens": plan["tokens"],
        "plan_seconds": round(plan_seconds, 3),
        "measured_batches": measured,
        "measured_tokens": tokens,
        "wall_seconds": round(wall, 3),
        "tokens_per_second": round(tokens / wall, 1) if wall else None,
        "batches_per_second": round(measured / wall, 1) if wall else None,
        "peak_rss_mb": round(usage.ru_maxrss / 1024, 1),
        "order_digest": last_digest,
        "resume_equal": resume_equal,
        "verified": not problems,
    }
    if args.out:
        ds.atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if resume_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
