"""CPU reference training loop for v3.1-A (NTP + router aux, no MTP)."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from architecture.accounting import analytical_counts
from architecture.model import GPT
from data_pipeline import loader
from .checkpoint import (load_state, move_optimizer_state, restore_rng,
                         save_artifact, save_state)
from .config import TrainConfig, sha256_file
from .data import load_manifest, tokenizer_info
from .determinism import (apply_prefill_policy, describe_prefill_policy,
                          resolve_prefill_policy, restores_prefill_state)
from .evaluate import evaluate
from .precision import autocast_context
from .schedule import CosineSchedule

CPU_LARGE_GUARD = 5_000_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _record_checkpoint(cfg: TrainConfig, path: str, step: int,
                       tokens_seen: int, log) -> dict:
    digest = sha256_file(path)
    size = os.path.getsize(path)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    record = {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "path": os.path.abspath(path),
        "bytes": int(size),
        "sha256": digest,
        "free_bytes": int(free),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _append_jsonl(os.path.join(cfg.out_dir, "checkpoints.jsonl"), record)
    log(f"  state -> {path} (step {step}, {size/1e9:.2f} GB, "
        f"sha {digest[:16]}, free {free/1e9:.1f} GB)")
    return record


def _param_groups(model, weight_decay: float):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def _loss_terms(model, batch, device, aux_weight: float):
    tokens = torch.from_numpy(np.asarray(batch["tokens"], dtype=np.int64)).to(device)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    lengths = torch.tensor(batch["lengths"], device=device)
    positions = torch.arange(x.size(1), device=device)[None, :]
    valid = positions < (lengths[:, None] - 1)
    logits, _, metrics, router = model(x, return_metrics=True,
                                       return_router=True)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                         reduction="none").reshape(y.shape)
    ce_sum = ce[valid].sum()
    count = int(valid.sum())
    aux = metrics["aux_loss"]
    return ce_sum, aux, count, router


def _router_summary(counts, n_experts: int):
    layers = []
    entropies = []
    for index, layer_counts in enumerate(counts):
        total = int(layer_counts.sum())
        if total == 0:
            continue
        probs = layer_counts.float() / total
        entropy = float(-(probs * probs.clamp(min=1e-12).log()).sum())
        norm = entropy / math.log(n_experts)
        entropies.append(norm)
        layers.append({"layer": index, "entropy_norm": norm,
                       "share_pct": [round(float(value) * 100, 2) for value in probs]})
    return {"layers": layers,
            "mean_entropy_norm": sum(entropies) / len(entropies) if entropies else None}


def state_path_for(cfg: TrainConfig, step: int) -> str:
    if "{step" in cfg.state_path:
        return cfg.state_path.format(step=step)
    return cfg.state_path


def _select_score(metrics: dict, cfg: TrainConfig) -> float:
    if cfg.select_metric == "macro_main_ce":
        value = metrics["overall"]["macro_main_ce"]
    elif cfg.select_metric == "token_main_ce":
        value = metrics["overall"]["main_ce"]
    else:
        value = metrics["overall"]["main_ce"]
    if value is None or not math.isfinite(value):
        raise ValueError("validation metric is not finite")
    return float(value)


@restores_prefill_state
def train(cfg: TrainConfig, log=print) -> dict:
    cfg.validate()
    manifest = load_manifest(cfg.data_root)
    tokenization = manifest.get("tokenization")
    if not tokenization:
        raise ValueError("corpus is not tokenized")
    total_params = analytical_counts(cfg.model)["total"]
    if cfg.device == "cpu" and total_params > CPU_LARGE_GUARD and not cfg.allow_large_cpu:
        raise ValueError(
            f"refusing to train a {total_params/1e6:.2f}M-parameter model on CPU; "
            f"this task is limited to tiny CPU smoke tests")
    if tokenization["vocab_size"] != cfg.model.vocab_size:
        raise ValueError(
            f"model vocab {cfg.model.vocab_size} != data vocab "
            f"{tokenization['vocab_size']}")

    set_seed(cfg.seed)
    if cfg.deterministic:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
    # Runtime policy, not a config field: the prefill never enters the
    # fingerprint, so resume compatibility is unaffected.
    prefill_policy = resolve_prefill_policy()
    apply_prefill_policy(prefill_policy)
    data_sha = sha256_file(os.path.join(cfg.data_root, "corpus", "manifest.json"))
    dataset_info = {
        "manifest_sha256": data_sha,
        "manifest_digest": manifest.get("digest"),
        "tokenizer_sha256": tokenization["tokenizer"]["sha256"],
        "train_tokens": tokenization["splits"]["train"]["tokens"],
        "vocab_size": tokenization["vocab_size"],
    }
    fingerprint = cfg.fingerprint(data_sha)

    model = GPT(cfg.model).to(cfg.device)
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg.weight_decay), lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    schedule = CosineSchedule(cfg.lr, cfg.warmup_steps, cfg.max_steps,
                              cfg.min_lr_ratio)
    state = None
    start_step = 0
    tokens_seen = 0
    best_metric = float("inf")
    best_step = None
    best_tokens_seen = 0
    best_state = None
    best_metrics = None
    loader_epoch, loader_batch = 0, 0
    if cfg.resume:
        state = load_state(cfg.resume, fingerprint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        move_optimizer_state(optimizer, cfg.device)
        schedule.load_state_dict(state["schedule"])
        start_step = int(state["step"])
        tokens_seen = int(state["tokens_seen"])
        best_metric = float(state["best_metric"])
        best_step = state.get("best_step")
        best_tokens_seen = int(state.get("best_tokens_seen", 0))
        best_state = state.get("best_state")
        best_metrics = state.get("best_metrics")
        loader_epoch = int(state["loader"]["epoch"])
        loader_batch = int(state["loader"]["batch"])
        restore_rng(state["rng"])
        log(f"resumed {cfg.resume} at step {start_step}, tokens {tokens_seen:,}")

    stop_step = cfg.max_steps if cfg.stop_after is None else min(cfg.stop_after, cfg.max_steps)
    if start_step >= stop_step:
        return {"steps": start_step, "tokens_seen": tokens_seen,
                "best_metric": best_metric, "best_step": best_step,
                "best_tokens_seen": best_tokens_seen,
                "stopped_early": stop_step < cfg.max_steps,
                "reason": "already_at_stop_step",
                "checkpoint": cfg.checkpoint,
                "state_path": state_path_for(cfg, start_step),
                "curve_jsonl": cfg.curve_jsonl}

    plan = loader.plan_sequences(cfg.data_root, split="train",
                                 seq_len=cfg.seq_len, mode="concat")
    if plan["tokens"] <= 0:
        raise ValueError("empty training plan")
    batches_per_epoch = math.ceil(len(plan["sequences"]) / cfg.batch_size)
    if cfg.single_pass:
        expected_tokens = int(tokenization["splits"]["train"]["tokens"])
        problems = loader.verify_plan(plan, expected_tokens=expected_tokens)
        if problems:
            raise ValueError("single-pass plan invalid: " + "; ".join(problems[:3]))
        expected_updates = math.ceil(batches_per_epoch / cfg.grad_accum)
        if cfg.max_steps != expected_updates:
            raise ValueError(
                f"single_pass max_steps {cfg.max_steps} does not match the "
                f"deterministic plan ({expected_updates} updates)")
        if loader_epoch != 0:
            raise ValueError("single-pass resume must be inside the first epoch")
    remaining_batches = (stop_step - start_step) * cfg.grad_accum
    epochs_needed = (max(0, 1 - loader_epoch) if cfg.single_pass
                     else math.ceil(remaining_batches / batches_per_epoch) + 1)
    iterator = loader.iter_batches(
        plan, cfg.data_root, batch_size=cfg.batch_size, epochs=epochs_needed,
        seed=cfg.seed, start_epoch=loader_epoch, start_batch=loader_batch)

    log(f"model {cfg.model.name}: {total_params/1e6:.2f}M total; "
        f"device {cfg.device}; seq {cfg.seq_len}; microbatch {cfg.batch_size} x "
        f"accum {cfg.grad_accum}; fingerprint {fingerprint[:16]}")
    log(describe_prefill_policy(prefill_policy))
    step = start_step
    tokens_seen = int(tokens_seen)
    loader_epoch_out, loader_batch_out = loader_epoch, loader_batch
    t_start = time.monotonic()

    def remaining_seconds():
        if cfg.max_wall_seconds is None:
            return None
        return cfg.max_wall_seconds - (time.monotonic() - t_start)

    def persist(path, step_value, tokens_value, epoch_value, batch_value):
        save_state(
            path, model=model, optimizer=optimizer, schedule=schedule,
            step=step_value, tokens_seen=tokens_value,
            loader_epoch=epoch_value, loader_batch=batch_value,
            best_metric=best_metric, best_step=best_step,
            best_tokens_seen=best_tokens_seen,
            best_state=best_state, best_metrics=best_metrics,
            config=cfg, data_manifest_sha256=data_sha,
            dataset=dataset_info, fingerprint=fingerprint)
        return _record_checkpoint(cfg, path, step_value, tokens_value, log)

    if step == 0 and cfg.eval_interval > 0:
        val = evaluate(model, cfg, "validation", max_batches=cfg.eval_max_batches,
                       exact_chars=cfg.exact_chars)
        gap = evaluate(model, cfg, "train", max_batches=cfg.train_gap_batches,
                       exact_chars=cfg.exact_chars)
        score = _select_score(val, cfg)
        best_metric = score
        best_step = 0
        best_tokens_seen = 0
        best_metrics = val
        best_state = {key: value.detach().cpu().clone()
                      for key, value in model.state_dict().items()}
        initial_record = {
            "step": 0, "tokens_seen": 0, "val": val,
            "train_gap": {
                "train_main_ce": gap["overall"]["main_ce"],
                "gap_nats": (val["overall"]["main_ce"] - gap["overall"]["main_ce"])
                if (val["overall"]["main_ce"] is not None
                    and gap["overall"]["main_ce"] is not None) else None,
            },
            "selection": {"metric": cfg.select_metric, "value": score,
                          "improved": True, "best_value": score},
        }
        _append_jsonl(cfg.curve_jsonl, initial_record)
        log(f"initial validation: macro {val['overall']['macro_main_ce']:.4f} "
            f"token {val['overall']['main_ce']:.4f}")
        if cfg.save_initial_state:
            initial_path = state_path_for(cfg, 0)
            persist(initial_path, 0, 0, 0, 0)
    max_tokens = (int(tokenization["splits"]["train"]["tokens"])
                  if cfg.single_pass else None)
    collapse_streak = 0
    while step < stop_step:
        if cfg.max_wall_seconds is not None and time.monotonic() - t_start > cfg.max_wall_seconds:
            partial_path = state_path_for(cfg, step)
            persist(partial_path, step, tokens_seen,
                    loader_epoch_out, loader_batch_out)
            _append_jsonl(cfg.curve_jsonl, {
                "event": "stopped", "step": step,
                "tokens_seen": tokens_seen, "reason": "wall_clock",
                "elapsed_s": round(time.monotonic() - t_start, 3)})
            log(f"wall-clock limit reached at step {step}; partial state written")
            return {"steps": step, "tokens_seen": tokens_seen,
                    "best_metric": best_metric, "best_step": best_step,
                    "best_tokens_seen": best_tokens_seen,
                    "stopped_early": True, "reason": "wall_clock",
                    "checkpoint": None, "state_path": partial_path,
                    "curve_jsonl": cfg.curve_jsonl}
        window = []
        for _ in range(cfg.grad_accum):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            break
        total_valid = sum(max(0, int(length) - 1) for batch in window
                          for length in batch["lengths"])
        if total_valid <= 0:
            continue
        if max_tokens is not None and tokens_seen + total_valid > max_tokens:
            raise RuntimeError(
                f"token budget exhausted: {tokens_seen} + {total_valid} > "
                f"{max_tokens}")
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        window_ce = window_aux = 0.0
        router_counts = None
        with autocast_context(cfg.device, cfg.precision):
            for batch in window:
                ce_sum, aux, count, router = _loss_terms(
                    model, batch, cfg.device, cfg.model.moe_aux_weight)
                loss = (ce_sum + cfg.model.moe_aux_weight * aux * count) / total_valid
                if not math.isfinite(float(loss.detach())):
                    raise RuntimeError(f"non-finite loss at update {step + 1}")
                loss.backward()
                window_ce += float(ce_sum.detach())
                window_aux += float(aux.detach()) * count
                if router_counts is None:
                    router_counts = [torch.zeros(cfg.model.n_experts, dtype=torch.long)
                                     for _ in router]
                for layer, top_idx in enumerate(router):
                    router_counts[layer] += torch.bincount(
                        top_idx.reshape(-1).cpu(), minlength=cfg.model.n_experts)
        if cfg.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip))
        else:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf")))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at update {step + 1}")
        lr = schedule.advance()
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        tokens_seen += total_valid
        if max_tokens is not None and tokens_seen > max_tokens:
            raise RuntimeError(
                f"token budget exceeded at update {step + 1}: {tokens_seen} > "
                f"{max_tokens}")
        step += 1
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_start
        if step_seconds > 120:
            raise RuntimeError(f"optimizer update {step} exceeded 120 s: {step_seconds:.1f}s")
        last = window[-1]
        loader_epoch_out = int(last["epoch"])
        loader_batch_out = int(last["batch"]) + 1
        if loader_batch_out >= batches_per_epoch:
            loader_epoch_out += 1
            loader_batch_out = 0
        train_main = window_ce / total_valid
        train_aux = window_aux / total_valid
        n_moe = sum(1 for block in model.blocks if block.is_moe)
        train_loss = train_main + cfg.model.moe_aux_weight * train_aux
        if not math.isfinite(train_loss):
            raise RuntimeError(f"non-finite training loss at update {step}")
        router_summary = (_router_summary(router_counts, cfg.model.n_experts)
                          if router_counts is not None else None)
        mean_entropy = (router_summary["mean_entropy_norm"]
                        if router_summary is not None else None)
        if mean_entropy is not None and mean_entropy < 0.5:
            collapse_streak += 1
        else:
            collapse_streak = 0
        if collapse_streak >= 10:
            raise RuntimeError(
                f"persistent router collapse: entropy {mean_entropy:.3f} "
                f"for {collapse_streak} updates")
        record = {
            "step": step,
            "lr": lr,
            "tokens_seen": tokens_seen,
            "train_main_ce": train_main,
            "train_aux": train_aux,
            "train_aux_per_layer": train_aux / max(1, n_moe),
            "train_aux_weighted": cfg.model.moe_aux_weight * train_aux,
            "train_loss": train_loss,
            "grad_norm": grad_norm,
            "step_seconds": round(step_seconds, 4),
            "tokens_per_s": round(total_valid / max(step_seconds, 1e-9), 1),
            "router": router_summary,
            "router_collapse_streak": collapse_streak,
            "elapsed_s": round(time.monotonic() - t_start, 3),
        }
        if step == start_step + 1 and start_step > 0:
            record["resume_trace"] = {
                "loader_epoch": int(window[0]["epoch"]),
                "loader_batch": int(window[0]["batch"]),
                "indices": [int(value) for value in window[0]["indices"]],
                "order_digest": window[0]["order_digest"],
                "tokens_seen_before": tokens_seen - total_valid,
            }
        if cfg.device == "cuda":
            record["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
            record["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
        eval_due = ((step in cfg.eval_steps) if cfg.eval_steps
                    else (step % cfg.eval_interval == 0))
        if eval_due or step == stop_step:
            left = remaining_seconds()
            if left is not None and left < 120:
                raise RuntimeError(
                    f"insufficient time budget before validation at update "
                    f"{step}: {left:.1f}s remaining")
            val = evaluate(model, cfg, "validation",
                           max_batches=cfg.eval_max_batches,
                           exact_chars=cfg.exact_chars)
            train_gap = evaluate(model, cfg, "train",
                                 max_batches=cfg.train_gap_batches,
                                 exact_chars=cfg.exact_chars)
            val_ce = val["overall"]["main_ce"]
            train_ce = train_gap["overall"]["main_ce"]
            record["val"] = val
            record["train_gap"] = {
                "train_main_ce": train_ce,
                "gap_nats": (val_ce - train_ce) if (val_ce is not None and train_ce is not None) else None,
            }
            score = _select_score(val, cfg)
            improved = score < best_metric
            record["selection"] = {"metric": cfg.select_metric, "value": score,
                                   "improved": improved, "best_value": best_metric}
            if improved:
                best_metric = score
                best_step = step
                best_tokens_seen = tokens_seen
                best_metrics = val
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                record["selection"]["best_value"] = best_metric
            log(f"step {step}: train {train_loss:.4f} | val macro "
                f"{val['overall']['macro_main_ce']:.4f} token "
                f"{val['overall']['main_ce']:.4f} | best {best_metric:.4f} @ {best_step}")
        if cfg.device == "cuda":
            record["vram_peak_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 3)
            record["vram_peak_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 1e9, 3)
        if (step <= 10 or step % cfg.log_every == 0 or "val" in record
                or "resume_trace" in record):
            _append_jsonl(cfg.curve_jsonl, record)
        due = (step in cfg.save_steps) if cfg.save_steps else (
            step % cfg.save_every == 0)
        if due or step == stop_step:
            left = remaining_seconds()
            if left is not None and left < 60:
                raise RuntimeError(
                    f"insufficient time budget before checkpoint at update "
                    f"{step}: {left:.1f}s remaining")
            state_path = state_path_for(cfg, step)
            persist(state_path, step, tokens_seen,
                    loader_epoch_out, loader_batch_out)
        if cfg.device == "cuda" and cfg.max_vram_gb is not None:
            peak_vram = torch.cuda.max_memory_reserved() / 1e9
            if peak_vram > cfg.max_vram_gb:
                raise RuntimeError(
                    f"peak VRAM {peak_vram:.3f} GiB exceeds the "
                    f"{cfg.max_vram_gb:.3f} GiB target at update {step}")

    stopped_early = step < cfg.max_steps
    if stopped_early:
        _append_jsonl(cfg.curve_jsonl, {
            "event": "stopped", "step": step, "tokens_seen": tokens_seen,
            "reason": "stop_after", "elapsed_s": round(time.monotonic() - t_start, 3)})
        return {"steps": step, "tokens_seen": tokens_seen,
                "best_metric": best_metric, "best_step": best_step,
                "best_tokens_seen": best_tokens_seen,
                "stopped_early": True, "reason": "stop_after",
                "checkpoint": None,
                "state_path": state_path_for(cfg, step),
                "curve_jsonl": cfg.curve_jsonl}

    if best_state is not None:
        model.load_state_dict(best_state)
    save_artifact(
        cfg.checkpoint, model=model, config=cfg, data_manifest_sha256=data_sha,
        dataset=dataset_info, tokenizer=tokenizer_info(manifest),
        selection={"metric": cfg.select_metric, "value": best_metric,
                   "step": best_step, "split": "validation"},
        metrics=best_metrics, step=best_step or step,
        tokens_seen=best_tokens_seen,
        fingerprint=fingerprint)
    artifact_sha = sha256_file(cfg.checkpoint)
    total_wall = time.monotonic() - t_start
    _append_jsonl(cfg.curve_jsonl, {
        "event": "run_end", "steps": step, "tokens_seen": tokens_seen,
        "best_step": best_step, "best_tokens_seen": best_tokens_seen,
        "best_metric": best_metric, "artifact": cfg.checkpoint,
        "artifact_sha256": artifact_sha,
        "artifact_bytes": os.path.getsize(cfg.checkpoint),
        "total_wall_seconds": round(total_wall, 3),
        "end_to_end_tokens_per_s": round(tokens_seen / max(total_wall, 1e-9), 1),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log(f"best checkpoint -> {cfg.checkpoint} (step {best_step}, "
        f"{cfg.select_metric} {best_metric:.4f}, sha {artifact_sha[:16]})")
    return {"steps": step, "tokens_seen": tokens_seen,
            "best_metric": best_metric, "best_step": best_step,
            "best_tokens_seen": best_tokens_seen,
            "best_metrics": best_metrics, "stopped_early": False,
            "checkpoint": cfg.checkpoint,
            "state_path": state_path_for(cfg, step),
            "artifact_sha256": artifact_sha,
            "total_wall_seconds": total_wall,
            "curve_jsonl": cfg.curve_jsonl}
