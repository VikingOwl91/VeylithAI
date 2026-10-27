"""PERF-5N runner: product front-padded by exactly 2 MiB, once.

Mode 4 places each SwiGLU product at offset 2,097,152 inside a storage of
tensor_bytes + 2,097,152. Because 2 MiB is a multiple of both the page size and
the block granularity, a product ending 4 bytes before a 2 MiB line keeps that
distance. The product gradient is not padded and w3 is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import v31p5g_plan as containment  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf5n-20261027")
MODE = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"one execution: 2 MiB front-padded product arm (mode {MODE})")
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
