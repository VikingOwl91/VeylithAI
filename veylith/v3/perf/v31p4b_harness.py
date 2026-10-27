"""v3.1-PERF-4B harness: correctness gate, kernel profile, production-path timing.

Modes
    gate1      one arm of the two-update bitwise gate (run one process per arm)
    profile    five profiled updates with the prefill on or off, Chrome trace
    train      the production training loop for 60 updates, prefill on or off

The gate and profile modes use the verified step-0 state so their digests are
comparable with PERF-3B and PERF-3D. The train mode drives
``training.train.train`` with the PERF-1 configuration so its per-update
``step_seconds`` are the same metric as PERF-1's baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, "veylith/v3")
sys.path.insert(0, os.path.join("veylith/v3", "perf"))

from v31p4b_nofill import (DeterminismShim, deterministic_state,
                           prefill_disabled, prime_allocator)  # noqa: E402

SCRATCH = os.path.expanduser("~/veylith-perf4b-20261027")
DATA = os.path.expanduser("~/veylith-v3-data/multi-100m-20261026")
STATE0 = os.path.expanduser("~/veylith-v31l-dense-20261027/ckpt_state_step0.pt")
EXPECTED_FINGERPRINT = "1212c3d134911dee2745d5deb2b91454296eaf86954d21013d4d641c0aefd260"
PRIME_GIB = 3.0


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


def build(purpose: str):
    from architecture.config import get_candidate
    from architecture.model import GPT
    from training.checkpoint import restore_rng
    from training.config import TrainConfig, sha256_file
    from training.data import iter_train_batches
    from training.schedule import CosineSchedule
    from training.train import _param_groups, set_seed

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    model_cfg = get_candidate("A")
    model_cfg.attention_impl = "sdpa"
    model_cfg.moe_impl = "dense"
    cfg = TrainConfig(
        data_root=DATA, out_dir=SCRATCH, model=model_cfg, device="cuda",
        precision="bf16", batch_size=8, grad_accum=8, seq_len=512,
        max_steps=500, lr=3e-4, min_lr_ratio=0.1, warmup_steps=15,
        weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=1.0,
        seed=0, eval_interval=1000, train_gap_batches=8, save_every=1000,
        select_metric="macro_main_ce", deterministic=True, exact_chars=True,
        max_wall_seconds=600.0)
    cfg.validate()
    data_sha = sha256_file(os.path.join(DATA, "corpus", "manifest.json"))
    fingerprint = cfg.fingerprint(data_sha)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise SystemExit(f"fingerprint mismatch: {fingerprint}")

    set_seed(cfg.seed)
    model = GPT(cfg.model).to("cuda")
    state = torch.load(STATE0, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.weight_decay),
                                  lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                  eps=cfg.eps)
    scheduler = CosineSchedule(cfg.lr, cfg.warmup_steps, cfg.max_steps,
                               cfg.min_lr_ratio)
    restore_rng(state["rng"])
    del state
    _plan, _bpe, iterator = iter_train_batches(DATA, cfg, 0, 0, epochs=1)
    return model, optimizer, scheduler, iterator, fingerprint, cfg


def run_updates(model, optimizer, scheduler, iterator, count, capture=False,
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
                lr = scheduler.advance()
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


def mode_gate1(args) -> int:
    model, optimizer, scheduler, iterator, fingerprint, _cfg = build("gate1")
    state_before = deterministic_state()
    priming = None
    if args.arm.startswith("variant_prime_"):
        pattern = 0xAA if args.arm.endswith("aa") else 0x55
        priming = prime_allocator(pattern, PRIME_GIB)
    captured = {}

    def capture_head(module, inputs, output):
        # Must return None: a forward hook that returns a value replaces the
        # module output, and returning the detached copy here severs the graph.
        captured.setdefault("logits", output.detach())

    handle = model.head.register_forward_hook(capture_head)
    try:
        if args.arm.startswith("variant"):
            with prefill_disabled():
                records = run_updates(model, optimizer, scheduler, iterator, 2,
                                      capture=True, head_capture=captured)
                state_in_effect = deterministic_state()
        else:
            records = run_updates(model, optimizer, scheduler, iterator, 2,
                                  capture=True, head_capture=captured)
            state_in_effect = deterministic_state()
    finally:
        handle.remove()
    state_after = deterministic_state()
    payload = {
        "arm": args.arm,
        "prefill": "off" if args.arm.startswith("variant") else "on",
        "fingerprint": fingerprint,
        "priming": priming,
        "state_before": state_before,
        "state_in_effect": state_in_effect,
        "state_after": state_after,
        "flag_restored": state_after == state_before,
        "records": records,
        "logits_digest": captured.get("logits_digest"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - args.started,
    }
    with open(args.out, "w", encoding="utf-8") as handle_out:
        json.dump(payload, handle_out, indent=2, sort_keys=True)
    print("arm", args.arm, "flag_restored", payload["flag_restored"],
          "state_in_effect", state_in_effect)
    if state_in_effect["fill_uninitialized_memory"] != (args.arm.startswith("baseline")):
        raise SystemExit("prefill state does not match the arm")
    return 0


def mode_profile(args) -> int:
    from torch.profiler import ProfilerActivity, profile, schedule

    model, optimizer, scheduler, iterator, fingerprint, _cfg = build("profile")
    off = args.flag == "off"
    run_updates(model, optimizer, scheduler, iterator, 5, annotate=False)
    state_before = deterministic_state()
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
            measured.extend(run_updates(model, optimizer, scheduler, iterator, 1,
                                        annotate=True))
            prof.step()
        state_in_effect = deterministic_state()
    payload = {
        "mode": "profile", "flag": args.flag, "prefill_off": off,
        "fingerprint": fingerprint,
        "state_before": state_before, "state_in_effect": state_in_effect,
        "state_after": deterministic_state(),
        "measured": measured,
        "trace": args.trace, "trace_exported": exported["called"],
        "trace_bytes": os.path.getsize(args.trace) if os.path.exists(args.trace) else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - args.started,
    }
    with open(args.out, "w", encoding="utf-8") as handle_out:
        json.dump(payload, handle_out, indent=2, sort_keys=True)
    print("profile", args.flag, "fills_in_effect_off", not state_in_effect["fill_uninitialized_memory"],
          "trace", payload["trace_bytes"])
    return 0


def mode_train(args) -> int:
    """Production training path, PERF-1 configuration, one arm."""
    from architecture.config import get_candidate
    from training.config import TrainConfig, sha256_file
    from training.train import train

    run_dir = os.path.join(SCRATCH, "arms", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    model_cfg = get_candidate("A")
    model_cfg.attention_impl = "sdpa"
    model_cfg.moe_impl = "dense"
    cfg = TrainConfig(
        data_root=DATA,
        out_dir=run_dir,
        model=model_cfg,
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
        max_wall_seconds=600.0,
        max_vram_gb=12.0,
        checkpoint=os.path.join(run_dir, "ckpt_best.pt"),
        state_path=os.path.join(run_dir, "ckpt_state_step{step}.pt"),
        curve_jsonl=os.path.join(run_dir, "curve.jsonl"),
        save_steps=(),
        save_initial_state=False,
    )
    cfg.validate()
    data_sha = sha256_file(os.path.join(DATA, "corpus", "manifest.json"))
    fingerprint = cfg.fingerprint(data_sha)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise SystemExit(f"configuration fingerprint mismatch: {fingerprint}")

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    state_before = deterministic_state()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    cpu_started = time.process_time()
    shim = DeterminismShim()
    context = shim if args.flag == "off" else _null()
    with context:
        result = train(cfg, log=print)
        state_in_effect = deterministic_state()
    wall = time.monotonic() - started
    cpu_seconds = time.process_time() - cpu_started
    finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    curve = os.path.join(run_dir, "curve.jsonl")
    records = []
    if os.path.exists(curve):
        with open(curve, encoding="utf-8") as handle_in:
            for line in handle_in:
                item = json.loads(line)
                if "step_seconds" in item:
                    records.append(item)
    payload = {
        "mode": "train", "tag": args.tag, "flag": args.flag,
        "prefill_off": args.flag == "off",
        "fingerprint": fingerprint,
        "shim_intercepted": shim.intercepted,
        "state_before": state_before,
        "state_in_effect": state_in_effect,
        "state_after": deterministic_state(),
        "flag_restored": deterministic_state() == state_before,
        "result": {k: v for k, v in result.items() if k != "best_metrics"},
        "records": records,
        "wall_seconds": wall,
        "process_cpu_seconds": cpu_seconds,
        "cpu_fraction_of_wall": cpu_seconds / wall if wall else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": time.monotonic() - started,
    }
    with open(args.out, "w", encoding="utf-8") as handle_out:
        json.dump(payload, handle_out, indent=2, sort_keys=True)
    print("train", args.tag, "flag", args.flag, "updates", len(records),
          "wall", round(wall, 2), "restored", payload["flag_restored"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate1", "profile", "train"),
                        required=True)
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
