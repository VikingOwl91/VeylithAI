"""v3.1-PERF-1 baseline repetition on the production training loop.

Runs the unmodified veylith training entry point for 60 optimizer updates
(10 warmup + 50 measured). The only configuration changes relative to the
v3.1-L run are run-control fields that do not enter the configuration
fingerprint: stop_after=60 (instead of 500), log_every=1 (per-update timing),
eval_interval=1000 (no mid-run validation), and no artifact write. The
production loop still writes one final 1.57 GB state at stop_step; that write
happens after the step-60 timing record and is reported as checkpoint overhead.
The LR schedule horizon stays at 500.

Writes only inside the experiment scratch directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

SCRATCH_ROOT = os.path.expanduser("~/veylith-perf1-20261027")
DATA = os.path.expanduser("~/veylith-v3-data/multi-100m-20261026")
EXPECTED_FINGERPRINT = "1212c3d134911dee2745d5deb2b91454296eaf86954d21013d4d641c0aefd260"
HISTORICAL = (
    "veylith-v31l-dense-20261027",
    "veylith-v31k-dense-20261027",
    "veylith-v31-d-20261027",
    "veylith-v31-e-20261027",
    "veylith-v31j-reproducer-20261027",
)


def check_scratch(path: str) -> str:
    resolved = os.path.realpath(path)
    root = os.path.realpath(SCRATCH_ROOT)
    if not resolved.startswith(root + os.sep):
        raise SystemExit(f"refusing output outside scratch: {resolved}")
    for name in HISTORICAL:
        if name in resolved:
            raise SystemExit(f"refusing output inside historical run: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--wall-seconds", type=float, default=300.0)
    args = parser.parse_args()

    sys.path.insert(0, "veylith/v3")
    import torch

    from architecture.config import get_candidate
    from training.config import TrainConfig, sha256_file
    from training.train import train

    run_dir = check_scratch(os.path.join(SCRATCH_ROOT, "baseline", f"rep{args.rep}"))
    os.makedirs(run_dir, exist_ok=True)
    model = get_candidate("A")
    model.attention_impl = "sdpa"
    model.moe_impl = "dense"
    cfg = TrainConfig(
        data_root=DATA,
        out_dir=run_dir,
        model=model,
        device="cuda",
        precision="bf16",
        batch_size=8,
        grad_accum=8,
        seq_len=512,
        max_steps=500,
        stop_after=60,
        lr=3e-4,
        min_lr_ratio=0.1,
        warmup_steps=15,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        grad_clip=1.0,
        seed=0,
        eval_interval=1000,
        eval_batch_size=8,
        eval_max_batches=None,
        train_gap_batches=8,
        save_every=1000,
        log_every=1,
        select_metric="macro_main_ce",
        deterministic=True,
        exact_chars=True,
        max_wall_seconds=args.wall_seconds,
        max_vram_gb=12.0,
        checkpoint=check_scratch(os.path.join(run_dir, "ckpt_best.pt")),
        state_path=check_scratch(os.path.join(run_dir, "ckpt_state_step{step}.pt")),
        curve_jsonl=check_scratch(os.path.join(run_dir, "curve.jsonl")),
        save_steps=(),
        save_initial_state=False,
    )
    cfg.validate()
    data_sha = sha256_file(os.path.join(DATA, "corpus", "manifest.json"))
    fingerprint = cfg.fingerprint(data_sha)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"configuration fingerprint mismatch: {fingerprint} != {EXPECTED_FINGERPRINT}")

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    properties = torch.cuda.get_device_properties(0)
    started = time.monotonic()
    cpu_started = time.process_time()
    result = train(cfg, log=print)
    wall = time.monotonic() - started
    cpu_seconds = time.process_time() - cpu_started
    free_after, _ = torch.cuda.mem_get_info()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    payload = {
        "rep": args.rep,
        "run_dir": run_dir,
        "fingerprint": fingerprint,
        "fingerprint_expected": EXPECTED_FINGERPRINT,
        "fingerprint_match": fingerprint == EXPECTED_FINGERPRINT,
        "config": cfg.to_dict(),
        "result": result,
        "wall_seconds": wall,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_fraction_of_wall": cpu_seconds / wall if wall else None,
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "total_bytes": total,
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "gcn_arch_name": getattr(properties, "gcnArchName", None),
            "capability": list(torch.cuda.get_device_capability(0)),
            "multi_processor_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = check_scratch(os.path.join(run_dir, "baseline_result.json"))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("baseline rep", args.rep, "wall", round(wall, 2),
          "cpu", round(cpu_seconds, 2), "peak_res",
          round(peak_reserved / 1e9, 3), "->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
