"""PERF-5J runner: the SwiGLU-product arm, exactly once.

Same containment as PERF-5G through 5I (`v31p5g_plan.run`): idle gate, no
foreign KFD process, `ulimit -c 0`, 120 s timeout, kernel-log window, health
check, no serialization, no retry. Only the scratch directory and the padded
storage differ: mode 2 pads the SwiGLU product instead of the w3 input gradient.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5g_plan as containment  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5j-20261027")
MODE = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"one execution: SwiGLU product padding (mode {MODE}) into {SCRATCH}")
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
