"""PERF-5P child entry point: stock reproducer with provenance recording.

Accepts the same CLI as the shared harness (``--mode repro --pad 0 --out ...``)
so the containment in ``v31p5g_plan.run`` can launch it unchanged. It refuses
any mode other than stock and records provenance into the case directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.path.join(HERE, "..", "..", "debug")
sys.path.insert(0, DEBUG)
sys.path.insert(0, HERE)

import v31p5p_provenance as provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--out", default="geometry.json")
    ap.add_argument("--repro-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    if args.mode != "repro" or args.pad != 0:
        raise SystemExit("PERF-5P runs the stock reproducer only (pad 0)")

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    provenance.enable(out_dir)
    import reproducer
    import sparse_moe_v2_train as impl

    status = {"pad": 0,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    code = 0

    orig_get_batch = impl.get_batch
    counter = {"n": 0}

    def get_batch_counted(*a, **k):
        counter["n"] += 1
        if counter["n"] % 25 == 0:
            provenance.dump_snapshot("roll")   # bounded: one rotating file
        return orig_get_batch(*a, **k)

    impl.get_batch = get_batch_counted
    first = {}
    restore_forward = provenance.capture_first_loss(impl, first)
    try:
        with provenance.Phase(impl):
            with provenance.TagMode():
                sys.argv = ([os.path.join(DEBUG, "reproducer.py")]
                            + list(args.repro_args))
                try:
                    reproducer.main()
                except BaseException as exc:  # noqa: BLE001
                    status["exception"] = {"type": type(exc).__name__,
                                           "message": str(exc).splitlines()[0][:300]}
                    code = 1
                    provenance.dump_snapshot("fault")
                    print("PROVFAULT " + json.dumps(status), flush=True)
    finally:
        impl.GPT.forward = restore_forward
        impl.get_batch = orig_get_batch
        provenance.dump_snapshot("final")
        provenance.finish()
        status["first_loss"] = first.get("first_loss")
        status["first_loss_error"] = first.get("first_loss_error")
        status["get_batch_calls"] = counter["n"]
        status["events"] = provenance._STATE["seq"]
        status["alloc_history"] = provenance._STATE["alloc_history"]
        status["truncated"] = provenance._STATE["truncated"]
        json.dump(status, open(os.path.join(out_dir, "prov.json"), "w"),
                  indent=2, sort_keys=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
