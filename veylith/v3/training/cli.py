"""CLI for the v3.1-A training loop. CPU is limited to tiny smoke models."""

from __future__ import annotations

import argparse
import os
import sys

from architecture.config import get_candidate
from .config import TrainConfig
from .train import train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="veylith v3.1-A training")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--attention", default="sdpa", choices=["sdpa", "manual"])
    parser.add_argument("--moe", default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-steps", default="")
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--train-gap-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--curve-jsonl", default=None)
    parser.add_argument("--single-pass", action="store_true")
    parser.add_argument("--save-steps", default="")
    parser.add_argument("--save-initial-state", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=5400.0)
    parser.add_argument("--max-vram-gb", type=float, default=None)
    parser.add_argument("--allow-unsafe-sparse", action="store_true")
    parser.add_argument("--exact-chars", dest="exact_chars",
                        action="store_true", default=True)
    parser.add_argument("--no-exact-chars", dest="exact_chars",
                        action="store_false")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model = get_candidate(args.model)
    model.attention_impl = args.attention
    model.moe_impl = args.moe
    save_steps = tuple(int(value) for value in args.save_steps.split(",")
                       if value.strip())
    eval_steps = tuple(int(value) for value in args.eval_steps.split(",")
                       if value.strip())
    state = args.state
    if state is None and args.single_pass:
        state = os.path.join(args.out_dir, "ckpt_state_step{step}.pt")
    cfg = TrainConfig(
        data_root=args.data_root, out_dir=args.out_dir, model=model,
        device=args.device, precision=args.precision,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        seq_len=args.seq_len, max_steps=args.max_steps,
        stop_after=args.stop_after, lr=args.lr, warmup_steps=args.warmup,
        eval_interval=args.eval_interval, eval_steps=eval_steps,
        eval_max_batches=args.eval_max_batches,
        train_gap_batches=args.train_gap_batches,
        save_every=args.save_every, seed=args.seed,
        resume=args.resume, checkpoint=args.checkpoint,
        state_path=state, curve_jsonl=args.curve_jsonl,
        single_pass=args.single_pass, save_steps=save_steps,
        save_initial_state=args.save_initial_state,
        exact_chars=args.exact_chars,
        max_wall_seconds=args.max_wall_seconds,
        max_vram_gb=args.max_vram_gb,
        allow_unsafe_sparse=args.allow_unsafe_sparse)
    try:
        result = train(cfg)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"steps={result['steps']} tokens={result['tokens_seen']} "
          f"best_step={result['best_step']} best={result['best_metric']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
