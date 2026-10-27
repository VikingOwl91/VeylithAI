"""PERF-5K runner: independent replication of the PERF-5J product arm.

Same containment and the same intervention as PERF-5J: `v31p5g_plan.run` with
mode 2, the unchanged harness, the same reproducer arguments, the same 64 KiB
slack and the same 100 iterations. Only the scratch directory differs, so the
two runs can be compared side by side.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5g_plan as containment  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5k-20261027")
MODE = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"one execution: replication of the product arm (mode {MODE})")
        print(f"scratch={SCRATCH}")
        print(f"python={containment.PYTHON}")
        print(f"harness={containment.PAD}")
        print(f"repro args={containment.REPRO_ARGS} timeout={containment.TIMEOUT_S}s")
        return 0
    if not args.execute:
        parser.error("choose --dry-run or --execute")
    os.makedirs(SCRATCH, exist_ok=True)
    containment.SCRATCH = SCRATCH
    record = containment.run("H", MODE)
    json.dump(record, open(os.path.join(SCRATCH, "record.json"), "w"), indent=2,
              sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
