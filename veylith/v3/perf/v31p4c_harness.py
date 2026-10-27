"""v3.1-PERF-4C harness: resumed-state gate, kernel profile, paired A/B arms.

Every arm resumes from the same immutable step-250 state
(``ckpt_state_step250.pt``, sha256 ab259ebf...) which carries non-zero AdamW
first and second moments, optimizer step counters at 250, scheduler state at
step 250, all four RNG states and the loader position {epoch 0, batch 2000}.

Modes
    gate1      one arm of the two-update bitwise gate on the resumed state
    profile    five profiled updates from the resumed state, prefill on or off
    train      production train() resumed from the fixture, prefill on or off

Reuses the PERF-4B shim (``v31p4b_nofill``) unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "veylith/v3")
sys.path.insert(0, os.path.join("veylith/v3", "perf"))

from v31p4b_nofill import (DeterminismShim, deterministic_state,
                           prefill_disabled, prime_allocator)  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf4c-20261027")
DATA = os.path.expanduser("~/veylith-v3-data/multi-100m-20261026")
FIXTURE = os.path.expanduser("~/veylith-v31l-dense-20261027/ckpt_state_step250.pt")
FIXTURE_SHA256 = "ab259ebf78166d3f5a2b87f4d1f60acb71f19a0ba3cddd257f899a6a0871d33d"
EXPECTED_FINGERPRINT = "1212c3d134911dee2745d5deb2b91454296eaf86954d21013d4d641c0aefd260"
STOP_STEP = 400


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def digest_tensors(items):
    hasher = hashlib.sha256()
    for name, tensor in items:
        data = tensor.detach().cpu().contiguous().float().numpy().tobytes()
        hasher.update(name.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(data)
    return hasher.hexdigest()


def model_digest(model):
    return digest_tensors(sorted(model.state_dict().items()))


def grad_digest(model):
    return digest_tensors([(name, param.grad)
                           for name, param in model.named_parameters()
                           if param.grad is not None])


def optimizer_digest(optimizer):
    state = optimizer.state_dict()["state"]
    items = []
    for index in sorted(state):
        for key in sorted(state[index]):
            value = state[index][key]
            if torch.is_tensor(value):
                items.append((f"{index}.{key}", value))
    return digest_tensors(items)


def scheduler_digest(schedule):
    payload = json.dumps(schedule.state_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def rng_digest():
    hasher = hashlib.sha256()
    hasher.update(repr(random.getstate()).encode())
    for part in np.random.get_state():
        hasher.update(repr(part).encode())
    hasher.update(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():
        hasher.update(state.cpu().numpy().tobytes())
    return hasher.hexdigest()


def fixture_config(out_dir: str, log_every: int = 1):
    """Rebuild the fixture's own config, replacing only run-control fields.

    Rebuilding from the stored config keeps the fingerprint identical; only
    out_dir, resume, stop_after, log_every, wall guard and output paths change.
    """
    from architecture.config import ModelConfig
    from training.config import TrainConfig, sha256_file

    state = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    stored = dict(state["config"])
    model_cfg = ModelConfig(**stored.pop("model"))
    stored["eval_steps"] = tuple(stored.get("eval_steps") or ())
    stored["save_steps"] = tuple(stored.get("save_steps") or ())
    stored.update({
        "out_dir": out_dir,
        "resume": FIXTURE,
        "stop_after": STOP_STEP,
        "log_every": log_every,
        "max_wall_seconds": 1800.0,
        "checkpoint": os.path.join(out_dir, "ckpt_best.pt"),
        "state_path": os.path.join(out_dir, "ckpt_state_step{step}.pt"),
        "curve_jsonl": os.path.join(out_dir, "curve.jsonl"),
    })
    cfg = TrainConfig(model=model_cfg, **stored)
    cfg.validate()
    data_sha = sha256_file(os.path.join(DATA, "corpus", "manifest.json"))
    fingerprint = cfg.fingerprint(data_sha)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise SystemExit(f"fingerprint mismatch: {fingerprint}")
    return cfg, state


def resume_build(cfg, state):
    """Mirror training.train resume: model, AdamW state, scheduler, RNG, loader."""
    from architecture.model import GPT
    from data_pipeline import loader
    from training.checkpoint import move_optimizer_state, restore_rng
    from training.schedule import CosineSchedule
    from training.train import _param_groups, set_seed

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    set_seed(cfg.seed)
    model = GPT(cfg.model).to("cuda")
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg.weight_decay), lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    optimizer.load_state_dict(state["optimizer"])
    move_optimizer_state(optimizer, "cuda")
    schedule = CosineSchedule(cfg.lr, cfg.warmup_steps, cfg.max_steps,
                              cfg.min_lr_ratio)
    schedule.load_state_dict(state["schedule"])
    restore_rng(state["rng"])

    plan = loader.plan_sequences(DATA, split="train", seq_len=cfg.seq_len,
                                 mode="concat")
    batches_per_epoch = math.ceil(len(plan["sequences"]) / cfg.batch_size)
    remaining = (STOP_STEP - int(state["step"])) * cfg.grad_accum
    epochs_needed = math.ceil(remaining / batches_per_epoch) + 1
    iterator = loader.iter_batches(
        plan, DATA, batch_size=cfg.batch_size, epochs=epochs_needed,
        seed=cfg.seed, start_epoch=int(state["loader"]["epoch"]),
        start_batch=int(state["loader"]["batch"]))
    return model, optimizer, schedule, iterator


def run_updates(model, optimizer, schedule, iterator, count, capture=False,
                annotate=False, head_capture=None):
    from torch.profiler import record_function

    from training.precision import autocast_context
    from training.train import _loss_terms

    auxiliary_weight = 0.01
    records = []
    for _ in range(count):
        def maybe_local(name):
            return record_function(name) if annotate else _null()

        data_started = time.perf_counter()
        with maybe_local("data_fetch"):
            window = [next(iterator) for _ in range(8)]
        data_seconds = time.perf_counter() - data_started
        total_valid = sum(max(0, int(length) - 1)
                          for batch in window for length in batch["lengths"])
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        cpu_started = time.process_time()
        with maybe_local("update"):
            optimizer.zero_grad(set_to_none=True)
            ce_terms, aux_terms, loss_terms, router_parts = [], [], [], []
            with autocast_context("cuda", "bf16"):
                for batch in window:
                    with maybe_local("forward"):
                        ce_sum, aux, count_terms, router = _loss_terms(
                            model, batch, "cuda", auxiliary_weight)
                        loss = (ce_sum + auxiliary_weight * aux * count_terms) / total_valid
                    with maybe_local("backward"):
                        loss.backward()
                    ce_terms.append(ce_sum.detach())
                    aux_terms.append(aux.detach() * count_terms)
                    loss_terms.append(loss.detach())
                    for top in router:
                        router_parts.append(top.detach().reshape(-1))
            with maybe_local("checks"):
                ce_total = torch.stack(ce_terms).sum()
                aux_total = torch.stack(aux_terms).sum()
                finite = torch.isfinite(torch.stack(loss_terms)).all()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                router_flat = torch.cat(router_parts)
                scalars = torch.cat([ce_total.reshape(1), aux_total.reshape(1),
                                     grad_norm.detach().reshape(1)]).cpu()
                router_host = router_flat.cpu().numpy().tobytes()
            if not bool(finite.item()):
                raise RuntimeError("non-finite loss")
            ce_sum_total = float(scalars[0])
            aux_sum_total = float(scalars[1])
            grad_norm = float(scalars[2])
            if not math.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm")
            with maybe_local("optimizer"):
                lr = schedule.advance()
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        router_hash = hashlib.sha256()
        router_hash.update(router_host)
        record = {
            "main_ce": ce_sum_total / total_valid,
            "aux_sum": aux_sum_total / total_valid,
            "loss": ce_sum_total / total_valid
            + auxiliary_weight * aux_sum_total / total_valid,
            "grad_norm": grad_norm,
            "lr": lr,
            "ce_targets": total_valid,
            "step_seconds": step_seconds,
            "data_seconds": data_seconds,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "router_digest": router_hash.hexdigest(),
            "scheduler_digest": scheduler_digest(schedule),
            "rng_digest": rng_digest(),
        }
        if capture:
            record["model_digest"] = model_digest(model)
            record["grad_digest"] = grad_digest(model)
            record["optimizer_digest"] = optimizer_digest(optimizer)
        records.append(record)
        if head_capture is not None and "logits_digest" not in head_capture:
            head_capture["logits_digest"] = digest_tensors(
                [("head", head_capture["logits"])])
            del head_capture["logits"]
    return records


def initial_digests(model, optimizer, schedule, state):
    return {
        "step": int(state["step"]),
        "tokens_seen": int(state["tokens_seen"]),
        "loader_epoch": int(state["loader"]["epoch"]),
        "loader_batch": int(state["loader"]["batch"]),
        "model_digest": model_digest(model),
        "optimizer_digest": optimizer_digest(optimizer),
        "scheduler_digest": scheduler_digest(schedule),
        "rng_digest": rng_digest(),
        "adamw_step_counters": sorted({float(t["step"])
                                       for t in optimizer.state.values()
                                       if "step" in t})[:3],
        "adamw_nonzero_exp_avg": sum(
            1 for t in optimizer.state.values()
            if "exp_avg" in t and bool((t["exp_avg"] != 0).any())),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def mode_gate1(args) -> int:
    cfg, state = fixture_config(SCRATCH)
    model, optimizer, schedule, iterator = resume_build(cfg, state)
    before = initial_digests(model, optimizer, schedule, state)
    del state
    priming = None
    if args.arm.startswith("resume_prime_"):
        pattern = 0xAA if args.arm.endswith("aa") else 0x55
        priming = prime_allocator(pattern, 3.0)
    captured = {}

    def capture_head(module, inputs, output):
        # Must return None; a return value replaces the module output.
        captured.setdefault("logits", output.detach())

    handle = model.head.register_forward_hook(capture_head)
    state_before = deterministic_state()
    try:
        if args.arm in ("resume_variant", "resume_prime_aa", "resume_prime_55"):
            with prefill_disabled():
                records = run_updates(model, optimizer, schedule, iterator, 2,
                                      capture=True, head_capture=captured)
                in_effect = deterministic_state()
        else:
            records = run_updates(model, optimizer, schedule, iterator, 2,
                                  capture=True, head_capture=captured)
            in_effect = deterministic_state()
    finally:
        handle.remove()
    payload = {
        "arm": args.arm,
        "prefill": "off" if args.arm != "resume_a" and args.arm != "resume_b" else "on",
        "fingerprint": EXPECTED_FINGERPRINT,
        "fixture": {"path": FIXTURE, "sha256": FIXTURE_SHA256},
        "priming": priming,
        "state_before": state_before,
        "state_in_effect": in_effect,
        "state_after": deterministic_state(),
        "deterministic_enabled_in_effect": bool(in_effect["deterministic_algorithms"]),
        "fill_flag_in_effect": bool(in_effect["fill_uninitialized_memory"]),
        "fill_restored": deterministic_state()["fill_uninitialized_memory"]
        == state_before["fill_uninitialized_memory"]
        and bool(deterministic_state()["deterministic_algorithms"]),
        "initial": before,
        "records": records,
        "logits_digest": captured.get("logits_digest"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - args.started,
    }
    with open(args.out, "w", encoding="utf-8") as handle_out:
        json.dump(payload, handle_out, indent=2, sort_keys=True)
    print("arm", args.arm, "prefill_in_effect", in_effect["fill_uninitialized_memory"],
          "initial_step", before["step"], "adamw_steps", before["adamw_step_counters"][:1],
          "nonzero_moments", before["adamw_nonzero_exp_avg"])
    return 0


def mode_profile(args) -> int:
    from torch.profiler import ProfilerActivity, profile, schedule

    cfg, state = fixture_config(os.path.join(SCRATCH, "profile"))
    model, optimizer, schedule_obj, iterator = resume_build(cfg, state)
    del state
    off = args.flag == "off"
    warm_context = prefill_disabled() if off else _null()
    with warm_context:
        run_updates(model, optimizer, schedule_obj, iterator, 5, annotate=False)
    exported = {"called": False}

    def on_trace(prof):
        prof.export_chrome_trace(args.trace)
        exported["called"] = True

    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   record_shapes=True, profile_memory=False, with_stack=False,
                   schedule=schedule(wait=0, warmup=0, active=5, repeat=1),
                   on_trace_ready=on_trace)
    measured = []
    context = prefill_disabled() if off else _null()
    with prof, context:
        for _ in range(5):
            measured.extend(run_updates(model, optimizer, schedule_obj, iterator,
                                        1, annotate=True))
            prof.step()
        in_effect = deterministic_state()
    payload = {
        "mode": "profile", "flag": args.flag,
        "state_in_effect": in_effect,
        "measured": measured,
        "trace": args.trace, "trace_bytes": os.path.getsize(args.trace)
        if os.path.exists(args.trace) else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - args.started,
    }
    with open(args.out, "w", encoding="utf-8") as handle_out:
        json.dump(payload, handle_out, indent=2, sort_keys=True)
    print("profile", args.flag, "prefill_off_in_effect",
          not in_effect["fill_uninitialized_memory"], "trace", payload["trace_bytes"])
    return 0


def mode_train(args) -> int:
    from training.train import train

    run_dir = os.path.join(SCRATCH, "arms", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    cfg, _state = fixture_config(run_dir)
    del _state
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    before = deterministic_state()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    cpu_started = time.process_time()
    shim = DeterminismShim()
    context = shim if args.flag == "off" else _null()
    with context:
        result = train(cfg, log=print)
        in_effect = deterministic_state()
    wall = time.monotonic() - started
    cpu_seconds = time.process_time() - cpu_started
    finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    records = []
    curve = os.path.join(run_dir, "curve.jsonl")
    if os.path.exists(curve):
        with open(curve, encoding="utf-8") as handle_in:
            for line in handle_in:
                item = json.loads(line)
                if "step_seconds" in item:
                    records.append(item)
    payload = {
        "mode": "train", "tag": args.tag, "flag": args.flag,
        "resume": FIXTURE, "fixture_sha256": FIXTURE_SHA256,
        "stop_after": STOP_STEP,
        "shim_intercepted": shim.intercepted,
        "state_before": before, "state_in_effect": in_effect,
        "state_after": deterministic_state(),
        "result": {k: v for k, v in result.items() if k != "best_metrics"},
        "records": records,
        "wall_seconds": wall, "process_cpu_seconds": cpu_seconds,
        "cpu_fraction_of_wall": cpu_seconds / wall if wall else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "run_dir": run_dir,
        "started_utc": started_utc, "finished_utc": finished_utc,
    }
    with open(args.out, "w", encoding="utf-8") as handle_out:
        json.dump(payload, handle_out, indent=2, sort_keys=True)
    print("train", args.tag, "flag", args.flag, "updates", len(records),
          "wall", round(wall, 1), "reserved", round(payload["peak_reserved_bytes"] / 2**30, 3))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate1", "profile", "train"), required=True)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--flag", choices=("on", "off"), default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.started = time.monotonic()
    if args.mode == "gate1":
        return mode_gate1(args)
    if args.mode == "profile":
        return mode_profile(args)
    return mode_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
