"""PERF-4D GPU integration gate.

Runs the production entrypoint (``training.train.train``) resumed from the
verified step-250 fixture for two matched optimizer updates, once with the
prefill enabled (debug) and once with the default policy, then dumps everything
needed for a bitwise comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, "veylith/v3")
sys.path.insert(0, os.path.join("veylith/v3", "perf"))

from v31p4c_harness import FIXTURE, FIXTURE_SHA256, SCRATCH, fixture_config
from v31p4b_nofill import deterministic_state
from training.determinism import PREFILL_ENV

STOP = 252


def digest(items):
    h = hashlib.sha256()
    for name, t in items:
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()


def state_digests(path):
    s = torch.load(path, map_location="cpu", weights_only=False)
    opt = s["optimizer"]["state"]
    opt_items = []
    for i in sorted(opt):
        for k in sorted(opt[i]):
            v = opt[i][k]
            if torch.is_tensor(v):
                opt_items.append((f"{i}.{k}", v))
    rng = hashlib.sha256()
    for part in s["rng"]["cuda"]:
        rng.update(part.cpu().numpy().tobytes())
    rng.update(s["rng"]["torch"].numpy().tobytes())
    return {
        "step": int(s["step"]),
        "tokens_seen": int(s["tokens_seen"]),
        "loader": s["loader"],
        "model": digest(sorted(s["model"].items())),
        "optimizer": digest(opt_items),
        "schedule": hashlib.sha256(
            json.dumps(s["schedule"], sort_keys=True, default=str).encode()
        ).hexdigest(),
        "rng": rng.hexdigest(),
        "best_metric": str(s.get("best_metric")),
        "best_step": s.get("best_step"),
        "provenance": s["versions"]["environment"]["deterministic_prefill"],
        "fingerprint": s["fingerprint"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", choices=("on", "off"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from training.train import state_path_for, train

    if args.flag == "on":
        os.environ[PREFILL_ENV] = "1"
    else:
        os.environ.pop(PREFILL_ENV, None)

    run_dir = os.path.join(SCRATCH, "integration", f"arm_{args.flag}")
    os.makedirs(run_dir, exist_ok=True)
    cfg, state = fixture_config(run_dir)
    del state
    cfg.stop_after = STOP
    cfg.max_wall_seconds = 900.0

    before = deterministic_state()
    logs = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = train(cfg, log=lambda *a: logs.append(" ".join(str(x) for x in a)))
    wall = time.monotonic() - started
    after = deterministic_state()

    payload = {
        "flag": args.flag,
        "fixture": {"path": FIXTURE, "sha256": FIXTURE_SHA256},
        "stop_after": STOP,
        "env_value": os.environ.get(PREFILL_ENV),
        "state_before": before,
        "state_after": after,
        "fill_restored": after["fill_uninitialized_memory"]
        == before["fill_uninitialized_memory"],
        "deterministic_after": after["deterministic_algorithms"],
        "result": {k: v for k, v in result.items() if k != "best_metrics"},
        "state_digests": state_digests(state_path_for(cfg, STOP)),
        "curve": [json.loads(line) for line in
                  open(cfg.curve_jsonl, encoding="utf-8")],
        "startup_log": logs,
        "wall_seconds": wall,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("integration arm", args.flag, "step", payload["state_digests"]["step"],
          "provenance", payload["state_digests"]["provenance"],
          "fill_restored", payload["fill_restored"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
