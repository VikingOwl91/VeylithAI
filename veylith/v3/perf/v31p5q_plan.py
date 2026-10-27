"""PERF-5Q runner: instrumented stock replication, up to two runs.

Same provenance implementation as PERF-5P, now with the full-precision first
loss captured at iteration 0. Runs case A, then case B only if A did not fault,
and stops at the first GPUVM fault.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5g_plan as containment  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5q-20261027")
REPRO = os.path.join(HERE, "v31p5p_repro.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-runs", type=int, default=2)
    args = parser.parse_args()
    if args.dry_run:
        print(f"up to {args.max_runs} stock+provenance runs into {SCRATCH}")
        print(f"child entry point={REPRO}")
        print(f"python={containment.PYTHON}")
        print(f"repro args={containment.REPRO_ARGS} timeout={containment.TIMEOUT_S}s")
        print("stop at the first fault")
        return 0
    if not args.execute:
        parser.error("choose --dry-run or --execute")
    os.makedirs(SCRATCH, exist_ok=True)
    containment.SCRATCH = SCRATCH
    containment.PAD = REPRO
    results = []
    for index, case in enumerate(["A", "B"][:args.max_runs]):
        record = containment.run(case, 0)
        results.append(record)
        if record.get("faulted"):
            print(f"STOP: fault in case {case}; second run not performed")
            break
    json.dump({"runs": results}, open(os.path.join(SCRATCH, "matrix.json"), "w"),
              indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
