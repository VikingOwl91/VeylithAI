"""PERF-5I runner: the corrected padded arm, exactly once.

Same containment as PERF-5G/5H (`v31p5g_plan.run`): idle gate, no foreign KFD
process, `ulimit -c 0`, 120 s timeout, kernel-log window, health check, no
serialization, no retry. Only the scratch directory differs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5g_plan as containment  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5i-20261027")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"one execution: corrected padded arm into {SCRATCH}")
        print(f"python={containment.PYTHON}")
        print(f"harness={containment.PAD}")
        print(f"repro args={containment.REPRO_ARGS} timeout={containment.TIMEOUT_S}s")
        return 0
    if not args.execute:
        parser.error("choose --dry-run or --execute")
    os.makedirs(SCRATCH, exist_ok=True)
    containment.SCRATCH = SCRATCH
    record = containment.run("H", 1)
    json.dump(record, open(os.path.join(SCRATCH, "record.json"), "w"), indent=2,
              sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
