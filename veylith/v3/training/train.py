"""CPU reference training loop for v3.1-A (NTP + router aux, no MTP)."""

from __future__ import annotations

import json
import math
import os
import random
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
    logits, _, metrics = model(x, return_metrics=True)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                         reduction="none").reshape(y.shape)
    ce_sum = ce[valid].sum()
    count = int(valid.sum())
    aux = metrics["aux_loss"]
    return ce_sum, aux, count


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
    data_sha = sha256_file(os.path.join(cfg.data_root, "corpus", "manifest.json"))
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
                "stopped_early": stop_step < cfg.max_steps,
                "checkpoint": cfg.checkpoint, "state_path": cfg.state_path,
                "curve_jsonl": cfg.curve_jsonl}

    plan = loader.plan_sequences(cfg.data_root, split="train",
                                 seq_len=cfg.seq_len, mode="concat")
    if plan["tokens"] <= 0:
        raise ValueError("empty training plan")
    batches_per_epoch = math.ceil(len(plan["sequences"]) / cfg.batch_size)
    remaining_batches = (stop_step - start_step) * cfg.grad_accum
    epochs_needed = math.ceil(remaining_batches / batches_per_epoch) + 1
    iterator = loader.iter_batches(
        plan, cfg.data_root, batch_size=cfg.batch_size, epochs=epochs_needed,
        seed=cfg.seed, start_epoch=loader_epoch, start_batch=loader_batch)

    log(f"model {cfg.model.name}: {total_params/1e6:.2f}M total; "
        f"device {cfg.device}; seq {cfg.seq_len}; microbatch {cfg.batch_size} x "
        f"accum {cfg.grad_accum}; fingerprint {fingerprint[:16]}")
    step = start_step
    tokens_seen = int(tokens_seen)
    loader_epoch_out, loader_batch_out = loader_epoch, loader_batch
    t_start = time.time()
    while step < stop_step:
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
        optimizer.zero_grad(set_to_none=True)
        window_ce = window_aux = 0.0
        with autocast_context(cfg.device, cfg.precision):
            for batch in window:
                ce_sum, aux, count = _loss_terms(
                    model, batch, cfg.device, cfg.model.moe_aux_weight)
                loss = (ce_sum + cfg.model.moe_aux_weight * aux * count) / total_valid
                loss.backward()
                window_ce += float(ce_sum.detach())
                window_aux += float(aux.detach()) * count
        if cfg.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip))
        else:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf")))
        lr = schedule.advance()
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        tokens_seen += total_valid
        step += 1
        last = window[-1]
        loader_epoch_out = int(last["epoch"])
        loader_batch_out = int(last["batch"]) + 1
        if loader_batch_out >= batches_per_epoch:
            loader_epoch_out += 1
            loader_batch_out = 0
        train_main = window_ce / total_valid
        train_aux = window_aux / total_valid
        train_loss = train_main + cfg.model.moe_aux_weight * train_aux
        record = {
            "step": step,
            "lr": lr,
            "tokens_seen": tokens_seen,
            "train_main_ce": train_main,
            "train_aux": train_aux,
            "train_loss": train_loss,
            "grad_norm": grad_norm,
            "elapsed_s": round(time.time() - t_start, 3),
        }
        if step % cfg.eval_interval == 0 or step == stop_step:
            val = evaluate(model, cfg, "validation",
                           max_batches=cfg.eval_max_batches)
            train_gap = evaluate(model, cfg, "train",
                                 max_batches=cfg.train_gap_batches)
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
                best_metrics = val
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                record["selection"]["best_value"] = best_metric
            log(f"step {step}: train {train_loss:.4f} | val macro "
                f"{val['overall']['macro_main_ce']:.4f} token "
                f"{val['overall']['main_ce']:.4f} | best {best_metric:.4f} @ {best_step}")
        if step % cfg.log_every == 0 or "val" in record:
            _append_jsonl(cfg.curve_jsonl, record)
        if step % cfg.save_every == 0 or step == stop_step:
            save_state(
                cfg.state_path, model=model, optimizer=optimizer, schedule=schedule,
                step=step, tokens_seen=tokens_seen,
                loader_epoch=loader_epoch_out, loader_batch=loader_batch_out,
                best_metric=best_metric, best_step=best_step,
                best_state=best_state, best_metrics=best_metrics,
                config=cfg, data_manifest_sha256=data_sha,
                fingerprint=fingerprint)
            log(f"  state -> {cfg.state_path} (step {step})")

    stopped_early = step < cfg.max_steps
    if stopped_early:
        return {"steps": step, "tokens_seen": tokens_seen,
                "best_metric": best_metric, "best_step": best_step,
                "stopped_early": True, "checkpoint": None,
                "state_path": cfg.state_path, "curve_jsonl": cfg.curve_jsonl}

    if best_state is not None:
        model.load_state_dict(best_state)
    save_artifact(
        cfg.checkpoint, model=model, config=cfg, data_manifest_sha256=data_sha,
        tokenizer=tokenizer_info(manifest),
        selection={"metric": cfg.select_metric, "value": best_metric,
                   "step": best_step, "split": "validation"},
        metrics=best_metrics, step=best_step or step, tokens_seen=tokens_seen,
        fingerprint=fingerprint)
    log(f"best checkpoint -> {cfg.checkpoint} (step {best_step}, "
        f"{cfg.select_metric} {best_metric:.4f})")
    return {"steps": step, "tokens_seen": tokens_seen,
            "best_metric": best_metric, "best_step": best_step,
            "best_metrics": best_metrics, "stopped_early": False,
            "checkpoint": cfg.checkpoint, "state_path": cfg.state_path,
            "curve_jsonl": cfg.curve_jsonl}
