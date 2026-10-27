"""v3.1-PERF-1 instrumented trace run (one of at most two).

Rebuilds the exact v3.1-L training step (fresh seed-0 init, step-0 RNG state,
same data order, optimizer, schedule and deterministic settings), runs five
warmup updates, then records five updates with CPU+GPU torch.profiler
activities and record_function phase annotations.

CPU only for the parts that are not the GPU training step. Writes only inside
the experiment scratch directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

SCRATCH_ROOT = os.path.expanduser("~/veylith-perf1-20261027")
DATA = os.path.expanduser("~/veylith-v3-data/multi-100m-20261026")
STATE0 = os.path.expanduser("~/veylith-v31l-dense-20261027/ckpt_state_step0.pt")
EXPECTED_FINGERPRINT = "1212c3d134911dee2745d5deb2b91454296eaf86954d21013d4d641c0aefd260"
HISTORICAL = (
    "veylith-v31l-dense-20261027",
    "veylith-v31k-dense-20261027",
    "veylith-v31-d-20261027",
    "veylith-v31-e-20261027",
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
    parser.add_argument("--trace", type=int, required=True, choices=(1, 2))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=5)
    args = parser.parse_args()
    if args.warmup != 5 or args.active != 5:
        raise SystemExit("the profiling budget is fixed at five plus five updates")

    sys.path.insert(0, "veylith/v3")
    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    from architecture.config import get_candidate
    from architecture.model import GPT
    from training.checkpoint import restore_rng
    from training.config import TrainConfig, sha256_file
    from training.data import iter_train_batches, load_manifest
    from training.precision import autocast_context
    from training.schedule import CosineSchedule
    from training.train import _loss_terms, _param_groups, set_seed

    out_dir = check_scratch(os.path.join(SCRATCH_ROOT, "profile", f"trace{args.trace}"))
    os.makedirs(out_dir, exist_ok=True)
    trace_path = check_scratch(os.path.join(out_dir, "trace.json"))
    model_cfg = get_candidate("A")
    model_cfg.attention_impl = "sdpa"
    model_cfg.moe_impl = "dense"
    cfg = TrainConfig(
        data_root=DATA, out_dir=out_dir, model=model_cfg,
        device="cuda", precision="bf16", batch_size=8, grad_accum=8, seq_len=512,
        max_steps=500, stop_after=None, lr=3e-4, min_lr_ratio=0.1,
        warmup_steps=15, weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
        grad_clip=1.0, seed=0, eval_interval=1000, eval_batch_size=8,
        train_gap_batches=8, save_every=1000, log_every=10,
        select_metric="macro_main_ce", deterministic=True,
        exact_chars=True, max_wall_seconds=240.0,
        checkpoint=check_scratch(os.path.join(out_dir, "ckpt_best.pt")),
        state_path=check_scratch(os.path.join(out_dir, "state.pt")),
        curve_jsonl=check_scratch(os.path.join(out_dir, "curve.jsonl")),
        save_steps=(), save_initial_state=False)
    cfg.validate()
    data_sha = sha256_file(os.path.join(DATA, "corpus", "manifest.json"))
    fingerprint = cfg.fingerprint(data_sha)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"configuration fingerprint mismatch: {fingerprint} != {EXPECTED_FINGERPRINT}")
    if cfg.device != "cuda" or model_cfg.moe_impl != "dense":
        raise SystemExit("this harness requires CUDA and dense MoE")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    set_seed(cfg.seed)
    model = GPT(cfg.model).to(cfg.device)
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.weight_decay),
                                  lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                  eps=cfg.eps)
    scheduler = CosineSchedule(cfg.lr, cfg.warmup_steps, cfg.max_steps,
                               cfg.min_lr_ratio)
    state = torch.load(STATE0, map_location="cpu", weights_only=False)
    restore_rng(state["rng"])
    del state
    manifest = load_manifest(DATA)
    aux_weight = cfg.model.moe_aux_weight
    plan, batches_per_epoch, iterator = iter_train_batches(
        DATA, cfg, 0, 0, epochs=1)
    del plan, batches_per_epoch

    def update(annotate: bool) -> dict:
        def maybe(name):
            return record_function(name) if annotate else _null()

        data_started = time.perf_counter()
        with maybe("data_fetch"):
            window = []
            for _ in range(cfg.grad_accum):
                window.append(next(iterator))
        data_seconds = time.perf_counter() - data_started
        total_valid = sum(max(0, int(length) - 1)
                          for batch in window for length in batch["lengths"])
        if total_valid <= 0:
            raise RuntimeError("empty update")
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        cpu_started = time.process_time()

        with maybe("update"):
            optimizer.zero_grad(set_to_none=True)
            window_ce = window_aux = 0.0
            router_counts = None
            with autocast_context(cfg.device, cfg.precision):
                for batch in window:
                    with maybe("forward"):
                        ce_sum, aux, count, router = _loss_terms(
                            model, batch, cfg.device, aux_weight)
                        loss = (ce_sum + aux_weight * aux * count) / total_valid
                        if not math.isfinite(float(loss.detach())):
                            raise RuntimeError("non-finite loss")
                    with maybe("backward"):
                        loss.backward()
                    window_ce += float(ce_sum.detach())
                    window_aux += float(aux.detach()) * count
                    if router_counts is None:
                        router_counts = [torch.zeros(cfg.model.n_experts,
                                                     dtype=torch.long)
                                         for _ in router]
                    for layer, top_idx in enumerate(router):
                        router_counts[layer] += torch.bincount(
                            top_idx.reshape(-1).cpu(),
                            minlength=cfg.model.n_experts)
            with maybe("grad_clip"):
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip))
            if not math.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm")
            with maybe("optimizer"):
                lr = scheduler.advance()
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        cpu_seconds = time.process_time() - cpu_started
        entropies = []
        for counts in router_counts:
            probs = counts.float() / counts.sum()
            entropies.append(float(-(probs * probs.clamp_min(1e-12).log())
                                   .sum() / math.log(cfg.model.n_experts)))
        return {
            "data_seconds": data_seconds,
            "step_seconds": step_seconds,
            "process_cpu_seconds": cpu_seconds,
            "main_ce": window_ce / total_valid,
            "aux_sum": window_aux / total_valid,
            "loss": window_ce / total_valid + aux_weight * window_aux / total_valid,
            "grad_norm": grad_norm,
            "lr": lr,
            "router_mean_entropy_norm": sum(entropies) / len(entropies),
            "router_layer_entropy_norm": entropies,
            "tokens": total_valid,
            "t_start": step_started,
            "t_end": step_started + step_seconds,
            "t_end_data": data_started + data_seconds,
        }

    warmup = []
    for _ in range(args.warmup):
        warmup.append(update(annotate=False))

    exported = {"called": False}
    trace_started = time.monotonic()

    def on_trace(prof):
        prof.export_chrome_trace(trace_path)
        exported["called"] = True

    measured = []
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=(args.trace == 2),
        profile_memory=False,
        with_stack=False,
        schedule=schedule(wait=0, warmup=0, active=args.active, repeat=1),
        on_trace_ready=on_trace,
    )
    with prof:
        for _ in range(args.active):
            measured.append(update(annotate=True))
            prof.step()
    trace_seconds = time.monotonic() - trace_started

    tables = {}
    for label, sort_key in (("cuda", "self_cuda_time_total"),
                            ("cpu", "self_cpu_time_total")):
        try:
            text = prof.key_averages().table(sort_by=sort_key, row_limit=80,
                                             max_name_column_width=120)
        except Exception as exc:  # keep the run usable if the field moved
            text = f"table failed: {exc!r}"
        tables[label] = text
        table_path = check_scratch(os.path.join(out_dir, f"key_averages_{label}.txt"))
        with open(table_path, "w", encoding="utf-8") as handle:
            handle.write(text)

    payload = {
        "trace": args.trace,
        "trace_path": trace_path,
        "trace_exported": exported["called"],
        "profiled_block_seconds": trace_seconds,
        "trace_size_bytes": os.path.getsize(trace_path) if os.path.exists(trace_path) else None,
        "record_shapes": args.trace == 2,
        "fingerprint": fingerprint,
        "fingerprint_match": fingerprint == EXPECTED_FINGERPRINT,
        "config": cfg.to_dict(),
        "warmup": warmup,
        "measured": measured,
        "tables": tables,
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = check_scratch(os.path.join(out_dir, "profile_result.json"))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("trace", args.trace, "exported", exported["called"],
          "measured", len(measured), "seconds", round(trace_seconds, 2),
          "->", out_path)
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
