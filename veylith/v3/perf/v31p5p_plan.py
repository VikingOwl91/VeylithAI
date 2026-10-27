"""PERF-5P runner: stock reproducer with provenance recording, once.

Reuses the shared containment (idle gate, no foreign KFD process, `ulimit -c 0`,
120 s timeout, kernel-log window, health check, no serialization, no retry) and
swaps the child entry point for one that enables the provenance recorder around
the unmodified reproducer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5g_plan as containment  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5p-20261027")
REPRO = os.path.join(HERE, "v31p5p_repro.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"one execution: stock reproducer + provenance into {SCRATCH}")
        print(f"child entry point={REPRO}")
        print(f"python={containment.PYTHON}")
        print(f"repro args={containment.REPRO_ARGS} timeout={containment.TIMEOUT_S}s")
        return 0
    if not args.execute:
        parser.error("choose --dry-run or --execute")
    os.makedirs(SCRATCH, exist_ok=True)
    containment.SCRATCH = SCRATCH
    containment.PAD = REPRO          # same CLI, provenance enabled inside
    record = containment.run("H", 0)  # stock: no padding of any kind
    json.dump(record, open(os.path.join(SCRATCH, "record.json"), "w"), indent=2,
              sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
