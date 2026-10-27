"""Read-only, CPU-only analysis of the v3.1-L run.

Three parts, each writing one compact JSON file:

    telemetry   learning dynamics, router telemetry, performance, coverage
    checkpoints hashes, metadata, parameter movement, router/expert weights
    routing     bounded validation inference: init vs step 500, per source

No GPU use, no optimizer update, no write to the experiment directory.

Usage:
    python v31m_analysis.py telemetry   --out telemetry.json
    python v31m_analysis.py checkpoints --out checkpoints.json
    python v31m_analysis.py routing     --out routing.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

RUN = os.path.expanduser("~/veylith-v31l-dense-20261027")
DATA = os.path.expanduser("~/veylith-v3-data/multi-100m-20261026")
FINGERPRINT = "1212c3d134911dee2745d5deb2b91454296eaf86954d21013d4d641c0aefd260"
STATE_STEPS = (0, 10, 25, 50, 100, 250, 500)
CE_CUMULATIVE = {0: 0, 100: 3_270_400, 250: 8_176_000, 500: 16_352_000}
SOURCES = ["cosmopedia-v2", "fineweb-edu", "wikipedia"]


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path, block=8 << 20):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _percentiles(values, points=(10, 50, 90)):
    ordered = sorted(values)
    out = {}
    for point in points:
        index = (len(ordered) - 1) * point / 100.0
        low = int(math.floor(index))
        high = int(math.ceil(index))
        if low == high:
            out[f"p{point}"] = ordered[low]
        else:
            weight = index - low
            out[f"p{point}"] = ordered[low] * (1 - weight) + ordered[high] * weight
    return out


def _share_imbalance(shares):
    values = np.asarray(shares, dtype=float)
    mean = values.mean()
    return {
        "cv": float(values.std(ddof=0) / mean) if mean else None,
        "max_over_min": float(values.max() / values.min()) if values.min() else None,
        "min_share": float(values.min()),
        "max_share": float(values.max()),
    }


def part_telemetry():
    sys.path.insert(0, "veylith/v3")
    from training.schedule import lr_at
    from data_pipeline import loader

    curve = [json.loads(line) for line in open(os.path.join(RUN, "curve.jsonl"),
                                               encoding="utf-8")]
    train = [row for row in curve if "train_loss" in row]
    vals = [row for row in curve if "val" in row]
    end = [row for row in curve if row.get("event") == "run_end"][0]
    info = _read_json(os.path.join(RUN, "run_info.json"))
    manifest = _read_json(os.path.join(DATA, "corpus", "manifest.json"))
    tokenization = manifest["tokenization"]

    evidence = {}
    for name, docs_path, run_path in (
            ("curve", "veylith/v3/docs/dense-run-v31l-2026-10-27-curve.jsonl",
             os.path.join(RUN, "curve.jsonl")),
            ("checkpoints", "veylith/v3/docs/dense-run-v31l-2026-10-27-checkpoints.jsonl",
             os.path.join(RUN, "checkpoints.jsonl")),
            ("log", "veylith/v3/docs/dense-run-v31l-2026-10-27.log",
             os.path.join(RUN, "run.log"))):
        docs_sha, docs_size = _sha256_file(docs_path)
        run_sha, run_size = _sha256_file(run_path)
        evidence[name] = {"docs_sha256": docs_sha, "run_sha256": run_sha,
                          "match": docs_sha == run_sha, "bytes": docs_size,
                          "size_match": docs_size == run_size}

    validation = []
    for row in vals:
        overall = row["val"]["overall"]
        router = row["val"]["router"]
        validation.append({
            "step": row["step"],
            "macro_ce": overall["macro_main_ce"],
            "token_ce": overall["main_ce"],
            "bits_per_char": overall["bits_per_char"],
            "aux_sum": overall["aux_loss"],
            "tokens": overall["tokens"],
            "chars": overall["chars"],
            "mean_entropy": router["mean_entropy_norm"],
            "layers": [{"layer": entry["layer"],
                        "entropy_norm": entry["entropy_norm"],
                        "share_pct": entry["share_pct"],
                        **_share_imbalance(entry["share_pct"])}
                       for entry in router["layers"]],
            "sources": {name: {"main_ce": entry["main_ce"],
                               "bits_per_char": entry["bits_per_char"],
                               "tokens": entry["tokens"]}
                        for name, entry in row["val"]["sources"].items()},
        })

    intervals = []
    for previous, current in zip(validation, validation[1:]):
        updates = current["step"] - previous["step"]
        targets = CE_CUMULATIVE[current["step"]] - CE_CUMULATIVE[previous["step"]]
        delta_macro = previous["macro_ce"] - current["macro_ce"]
        delta_token = previous["token_ce"] - current["token_ce"]
        intervals.append({
            "from": previous["step"], "to": current["step"],
            "updates": updates, "ce_targets": targets,
            "delta_macro_ce": delta_macro,
            "delta_token_ce": delta_token,
            "delta_bits_per_char": previous["bits_per_char"] - current["bits_per_char"],
            "relative_macro_improvement": delta_macro / previous["macro_ce"],
            "macro_per_100_updates": delta_macro / updates * 100,
            "macro_per_million_targets": delta_macro / (targets / 1e6),
            "token_per_100_updates": delta_token / updates * 100,
            "token_per_million_targets": delta_token / (targets / 1e6),
            "source_deltas": {name: previous["sources"][name]["main_ce"]
                              - current["sources"][name]["main_ce"]
                              for name in current["sources"]},
        })
    first, last = validation[0], validation[-1]
    totals = {
        "macro_ce": first["macro_ce"] - last["macro_ce"],
        "token_ce": first["token_ce"] - last["token_ce"],
        "bits_per_char": first["bits_per_char"] - last["bits_per_char"],
        "relative_macro_improvement": (first["macro_ce"] - last["macro_ce"])
        / first["macro_ce"],
        "relative_token_improvement": (first["token_ce"] - last["token_ce"])
        / first["token_ce"],
        "macro_per_100_updates": (first["macro_ce"] - last["macro_ce"]) / 500 * 100,
        "macro_per_million_targets": (first["macro_ce"] - last["macro_ce"])
        / (CE_CUMULATIVE[500] / 1e6),
    }

    train_points = [{"step": row["step"], "main_ce": row["train_main_ce"],
                     "aux": row["train_aux"], "loss": row["train_loss"],
                     "grad_norm": row["grad_norm"], "lr": row["lr"],
                     "entropy": row["router"]["mean_entropy_norm"],
                     "entropy_layers": [entry["entropy_norm"]
                                        for entry in row["router"]["layers"]],
                     "share_cv_mean": statistics.mean(
                         _share_imbalance(entry["share_pct"])["cv"]
                         for entry in row["router"]["layers"])}
                    for row in train]
    grad = [row["grad_norm"] for row in train]
    lr_ok = all(abs(lr_at(row["step"] - 1, 3e-4, 15, 500, 0.1) - row["lr"]) <= 1e-15
                for row in train)
    aux_per_layer = [row["train_aux"] / 7 for row in train]
    train_val_gap = []
    for step in (100, 250, 500):
        t = [row for row in train_points if row["step"] == step]
        v = [row for row in validation if row["step"] == step]
        if t and v:
            train_val_gap.append({"step": step, "train_main_ce": t[0]["main_ce"],
                                  "val_token_ce": v[0]["token_ce"],
                                  "val_minus_train": v[0]["token_ce"] - t[0]["main_ce"]})

    step_seconds = [row["step_seconds"] for row in train]
    throughput = [row["tokens_per_s"] for row in train]
    checkpoints = [json.loads(line) for line in
                   open(os.path.join(RUN, "checkpoints.jsonl"), encoding="utf-8")]

    def clock(stamp):
        parsed = time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
        return parsed.tm_hour * 3600 + parsed.tm_min * 60 + parsed.tm_sec

    start_clock = clock("2026-10-27T08:42:52Z")
    milestones = []
    for step in (100, 250, 500):
        row = [r for r in train if r["step"] == step][0]
        stamp = [c for c in checkpoints if c["step"] == step][0]["time_utc"]
        milestones.append({"step": step, "elapsed_s": row["elapsed_s"],
                           "checkpoint_written_at_s": clock(stamp) - start_clock,
                           "validation_plus_write_s": clock(stamp) - start_clock
                           - row["elapsed_s"]})
    steady = statistics.median(step_seconds[1:])
    performance = {
        "logged_updates": len(train),
        "unlogged_updates": 500 - len(train),
        "step_seconds": {"min": min(step_seconds), "max": max(step_seconds),
                         "mean": statistics.mean(step_seconds),
                         "median": statistics.median(step_seconds),
                         **_percentiles(step_seconds)},
        "steady_state_median_s": steady,
        "first_update_seconds": step_seconds[0],
        "first_update_overhead_s": step_seconds[0] - steady,
        "throughput": {"mean": statistics.mean(throughput),
                       "median": statistics.median(throughput),
                       **_percentiles(throughput)},
        "end_to_end_throughput": end["end_to_end_tokens_per_s"],
        "end_to_end_over_median_training": end["end_to_end_tokens_per_s"]
        / statistics.median(throughput),
        "internal_wall_s": end["total_wall_seconds"],
        "elapsed_at_last_update_s": train[-1]["elapsed_s"],
        "milestones": milestones,
        "estimated_training_s": steady * 500,
        "estimated_validation_and_checkpoint_s": end["total_wall_seconds"]
        - steady * 500,
        "estimated_validation_and_checkpoint_fraction": (
            end["total_wall_seconds"] - steady * 500)
        / end["total_wall_seconds"],
        "memory": {
            "peak_allocated_gb": max(row["vram_peak_allocated_gb"] for row in train),
            "peak_reserved_gb": max(row["vram_peak_reserved_gb"] for row in train),
            "steady_allocated_gb": train[-1]["vram_allocated_gb"],
            "steady_reserved_gb": train[-1]["vram_reserved_gb"],
            "reserved_gap_gb": train[-1]["vram_reserved_gb"]
            - train[-1]["vram_allocated_gb"],
            "allocated_peak_step": max(
                train, key=lambda row: row["vram_peak_allocated_gb"])["step"],
            "reserved_stabilized_step": min(
                row["step"] for row in train
                if row["vram_reserved_gb"] == max(r["vram_reserved_gb"] for r in train)),
        },
    }

    plan = loader.plan_sequences(DATA, split="train", seq_len=512, mode="concat")
    count = len(plan["sequences"])
    sampler = loader.EpochSampler(count, 0)
    order = sampler.order(0)
    consumed = 4000 * 8
    consumed_indices = [int(value) for value in order[:consumed]]
    coverage = {
        "packing_mode": plan["mode"],
        "plan_sequences": count,
        "plan_tokens": plan["tokens"],
        "seq_len": plan["seq_len"],
        "batches_per_epoch_microbatch8": math.ceil(count / 8),
        "consumed_microbatches": 4000,
        "consumed_sequences": consumed,
        "fraction_of_epoch": consumed / count,
        "positions": 500 * 64 * 512,
        "ce_targets": 16_352_000,
        "fraction_of_corpus_positions": (500 * 64 * 512) / plan["tokens"],
        "fraction_of_corpus_ce": 16_352_000 / plan["tokens"],
        "unique_sequences_consumed": len(set(consumed_indices)),
        "repeated_sequences": consumed - len(set(consumed_indices)),
        "full_corpus_pass": False,
        "loader_cursor_epoch": 0,
        "loader_cursor_batch": 4000,
        "epoch0_order_sha256": sampler.digest(0),
        "last_sequence_length": plan["sequences"][-1].length,
        "all_but_last_are_512": all(s.length == 512
                                    for s in plan["sequences"][:-1]),
        "plan_problems": loader.verify_plan(
            plan, expected_tokens=tokenization["splits"]["train"]["tokens"]),
    }

    return {
        "run": "v3.1-l-dense-20261027",
        "evidence": evidence,
        "learning_dynamics": {
            "validation": validation,
            "intervals": intervals,
            "totals": totals,
            "train_ce": train_points,
            "grad_norm": {"min": min(grad), "max": max(grad),
                          "median": statistics.median(grad),
                          "updates_above_clip": [point["step"]
                                                 for point in train_points
                                                 if point["grad_norm"] > 1.0],
                          **_percentiles(grad)},
            "lr_matches_schedule": lr_ok,
            "lr_peak": 3e-4,
            "lr_final": train_points[-1]["lr"],
            "aux_per_layer_mean": statistics.mean(aux_per_layer),
            "aux_per_layer_final": aux_per_layer[-1],
            "aux_per_layer_minimum": 1.0,
            "aux_final_over_minimum": aux_per_layer[-1] / 1.0,
            "train_val_gap": train_val_gap,
            "dropout_during_training": 0.1,
            "config_snapshot": info["config"],
        },
        "performance": performance,
        "coverage": coverage,
    }


def part_checkpoints():
    import torch

    sys.path.insert(0, "veylith/v3")
    from architecture.config import ModelConfig  # noqa: F401  (shape check)

    recorded = {row["step"]: row for row in
                (json.loads(line) for line in
                 open(os.path.join(RUN, "checkpoints.jsonl"), encoding="utf-8"))}
    artifact_path = os.path.join(RUN, "ckpt_best.pt")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)

    groups = {
        "embedding": lambda key: key == "tok_emb.weight",
        "head": lambda key: key == "head.weight",
        "attention": lambda key: ".attn." in key,
        "dense_mlp": lambda key: key.startswith("blocks.0.mlp."),
        "router": lambda key: ".mlp.gate.weight" in key,
        "experts": lambda key: ".mlp.experts." in key,
        "norm": lambda key: key.startswith("ln_f") or ".ln" in key,
    }

    def group_of(key):
        for name, test in groups.items():
            if test(key):
                return name
        raise KeyError(key)

    def stats(tensors):
        total = sum(t.numel() for t in tensors)
        norm = math.sqrt(sum(float(t.double().pow(2).sum()) for t in tensors))
        max_abs = max(float(t.abs().max()) for t in tensors)
        finite = all(bool(torch.isfinite(t).all()) for t in tensors)
        return {"params": int(total), "l2": norm, "max_abs": max_abs,
                "finite": finite}

    def expert_stack(model_state):
        width = 512 * 960 * 3
        stack = torch.empty(7, 8, width, dtype=torch.float32)
        for layer in range(7):
            block = layer + 1
            for expert in range(8):
                parts = [model_state[
                    f"blocks.{block}.mlp.experts.{expert}.{name}.weight"]
                    for name in ("w1", "w2", "w3")]
                stack[layer, expert] = torch.cat([part.reshape(-1) for part in parts])
        return stack

    def cosine_rows(matrix):
        unit = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return unit @ unit.T

    states = {}
    previous_model = None
    expert_snapshots = {}
    router_snapshots = {}
    best_hashes = {}
    for step in STATE_STEPS:
        path = os.path.join(RUN, f"ckpt_state_step{step}.pt")
        sha, size = _sha256_file(path)
        record = recorded[step]
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_state = state["model"]
        param_stats = {}
        for name in groups:
            selected = [t for key, t in model_state.items() if group_of(key) == name]
            param_stats[name] = stats(selected)
        deltas = None
        if previous_model is not None:
            assert set(previous_model) == set(model_state), step
            deltas = {}
            for name in groups:
                keys = [key for key, t in model_state.items() if group_of(key) == name]
                diff = math.sqrt(sum(
                    float((model_state[key].double()
                           - previous_model[key].double()).pow(2).sum())
                    for key in keys))
                base = param_stats[name]["l2"]
                deltas[name] = {"l2_delta": diff,
                                "relative": diff / base if base else None}
            deltas["all"] = {"l2_delta": math.sqrt(sum(
                float((model_state[key].double()
                       - previous_model[key].double()).pow(2).sum())
                for key in model_state))}
        del previous_model
        previous_model = {key: value.clone() for key, value in model_state.items()}

        router_snapshots[step] = {
            f"layer{index + 1}": model_state[
                f"blocks.{index + 1}.mlp.gate.weight"].float().clone()
            for index in range(7)}
        expert_snapshots[step] = expert_stack(model_state)

        best_hasher = hashlib.sha256()
        for key in sorted(state["best_state"]):
            best_hasher.update(key.encode())
            best_hasher.update(state["best_state"][key].float().numpy().tobytes())
        best_content = best_hasher.hexdigest()
        best_hashes[step] = best_content
        best_matches_model = state["best_step"] == step and all(
            torch.equal(state["best_state"][key], model_state[key])
            for key in model_state)
        optimizer_steps = sorted({int(value["step"]) for value in
                                  state["optimizer"]["state"].values()
                                  if "step" in value})
        states[step] = {
            "path": path,
            "bytes": size,
            "sha256": sha,
            "recorded_sha256_match": sha == record["sha256"],
            "recorded_bytes_match": size == record["bytes"],
            "kind": state.get("kind"),
            "format": state.get("format"),
            "fingerprint_match": state.get("fingerprint") == FINGERPRINT,
            "step": state["step"],
            "tokens_seen": state["tokens_seen"],
            "loader": state["loader"],
            "best_step": state["best_step"],
            "best_metric": state["best_metric"],
            "best_tokens_seen": state["best_tokens_seen"],
            "best_state_matches_model": best_matches_model,
            "best_state_content_sha256": best_content,
            "schedule": state["schedule"],
            "optimizer_steps": optimizer_steps,
            "optimizer_entries": len(state["optimizer"]["state"]),
            "rng_keys": sorted(state["rng"]),
            "cuda_rng_tensors": len(state["rng"].get("cuda", [])),
            "dataset": state["dataset"],
            "environment": state["versions"]["environment"],
            "model_params": param_stats,
            "param_groups_total": sum(entry["params"]
                                      for entry in param_stats.values()),
            "tied_embeddings": torch.equal(model_state["tok_emb.weight"],
                                           model_state["head.weight"]),
            "nonfinite_params": not all(entry["finite"]
                                        for entry in param_stats.values()),
            "weight_deltas_from_previous": deltas,
        }
        del state, model_state

    router = {}
    for step in (0, 250, 500):
        per_layer = {}
        for name, matrix in router_snapshots[step].items():
            gram = cosine_rows(matrix)
            off = gram - torch.eye(8)
            per_layer[name] = {
                "row_norm_mean": float(matrix.norm(dim=1).mean()),
                "row_norm_min": float(matrix.norm(dim=1).min()),
                "row_norm_max": float(matrix.norm(dim=1).max()),
                "row_cosine_mean_offdiag": float(off.sum() / (8 * 7)),
                "row_cosine_max_offdiag": float(off.max()),
            }
        router[step] = per_layer
    router_change = {}
    for name in router_snapshots[0]:
        before = router_snapshots[0][name]
        after = router_snapshots[500][name]
        router_change[name] = {
            "relative_l2_change": float((after - before).norm() / before.norm()),
            "row_cosine_over_time_mean": float(
                (before / before.norm(dim=1, keepdim=True))
                .mul(after / after.norm(dim=1, keepdim=True))
                .sum(dim=1).mean()),
        }

    experts = {}
    for step in (0, 250, 500):
        stack = expert_snapshots[step]
        norms = stack.norm(dim=2)
        per_layer = []
        for layer in range(7):
            gram = cosine_rows(stack[layer])
            off = gram - torch.eye(8)
            per_layer.append({
                "layer": layer + 1,
                "expert_norm_mean": float(norms[layer].mean()),
                "expert_norm_min": float(norms[layer].min()),
                "expert_norm_max": float(norms[layer].max()),
                "cross_expert_cosine_mean": float(off.sum() / (8 * 7)),
                "cross_expert_cosine_max": float(off.max()),
            })
        experts[step] = per_layer
    displacement = []
    for layer in range(7):
        delta = expert_snapshots[500][layer] - expert_snapshots[0][layer]
        gram = cosine_rows(delta)
        off = gram - torch.eye(8)
        displacement.append({
            "layer": layer + 1,
            "displacement_norm_mean": float(delta.norm(dim=1).mean()),
            "displacement_norm_min": float(delta.norm(dim=1).min()),
            "displacement_norm_max": float(delta.norm(dim=1).max()),
            "cross_expert_displacement_cosine_mean": float(off.sum() / (8 * 7)),
            "cross_expert_displacement_cosine_max": float(off.max()),
        })

    artifact_sha, artifact_bytes = _sha256_file(artifact_path)
    artifact_matches = len(artifact["model"]) == len(previous_model) and all(
        torch.equal(artifact["model"][key], previous_model[key])
        for key in artifact["model"])
    return {
        "states": states,
        "artifact": {
            "path": artifact_path,
            "sha256": artifact_sha,
            "bytes": artifact_bytes,
            "kind": artifact["kind"],
            "format": artifact["format"],
            "step": artifact["step"],
            "tokens_seen": artifact["tokens_seen"],
            "fingerprint_match": artifact["fingerprint"] == FINGERPRINT,
            "model_matches_step500_state": artifact_matches,
            "selection": artifact["selection"],
            "dataset": artifact["dataset"],
            "environment": artifact["versions"]["environment"],
            "has_mtp_keys": any("mtp" in key for key in artifact),
        },
        "best_state_hashes": best_hashes,
        "router": {"layers": router, "change": router_change},
        "experts": {"weights": experts, "displacement": displacement},
    }


class _Probe:
    """Capture router logits and expert-output similarity, one layer at a time."""

    def __init__(self, model):
        self.logits = {}
        self.expert_out = {}
        self.expert_cos = {}
        self.expert_residual = {}
        self.handles = []
        moe_index = 0
        for index, block in enumerate(model.blocks):
            if not block.is_moe:
                continue
            self.handles.append(block.mlp.gate.register_forward_hook(
                self._gate_hook(moe_index)))
            for expert_index, expert in enumerate(block.mlp.experts):
                self.handles.append(expert.register_forward_hook(
                    self._expert_hook(moe_index, expert_index)))
            self.handles.append(block.mlp.register_forward_hook(
                self._moe_hook(moe_index)))
            moe_index += 1

    def _gate_hook(self, layer):
        def hook(module, inputs, output):
            self.logits[layer] = output.detach().float()
        return hook

    def _expert_hook(self, layer, expert_index):
        def hook(module, inputs, output):
            self.expert_out.setdefault(layer, {})[expert_index] = (
                output.detach().float())
        return hook

    def _moe_hook(self, layer):
        def hook(module, inputs, output):
            outputs = self.expert_out.pop(layer, {})
            if len(outputs) != 8:
                return
            stacked = torch.stack([outputs[index] for index in range(8)])
            positions = stacked.shape[1]
            unit = stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            gram = torch.einsum("ecd,fcd->ef", unit, unit)
            off = gram - torch.eye(8)
            self.expert_cos[layer] = float(off.sum() / (8 * 7) / positions)
            mean_out = stacked.mean(dim=0)
            residual = (stacked - mean_out).norm(dim=-1)
            scale = mean_out.norm(dim=-1).clamp_min(1e-9)
            self.expert_residual[layer] = float((residual / scale).mean())
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def part_routing():
    import torch
    sys.path.insert(0, "veylith/v3")
    from architecture.config import ModelConfig
    from architecture.model import GPT
    from training.data import iter_source_batches, source_tensors

    started = time.monotonic()
    deadline_s = 540
    batch_size = 8
    batches_per_source = 5

    artifact = torch.load(os.path.join(RUN, "ckpt_best.pt"), map_location="cpu",
                          weights_only=False)
    state0 = torch.load(os.path.join(RUN, "ckpt_state_step0.pt"),
                        map_location="cpu", weights_only=False)
    models = {}
    for label, config, weights in (
            ("init", state0["model_config"], state0["model"]),
            ("step500", artifact["model_config"], artifact["model"])):
        model = GPT(ModelConfig(**config))
        model.load_state_dict(weights)
        model.eval()
        models[label] = model
    del artifact, state0

    result = {}
    truncated = False
    for label, model in models.items():
        result[label] = {}
        probe = _Probe(model)
        try:
            for source in SOURCES:
                seconds_start = time.monotonic()
                counts_top1 = np.zeros((7, 8))
                counts_top2 = np.zeros((7, 8))
                halves = {0: np.zeros((7, 8)), 1: np.zeros((7, 8))}
                position_halves = {0: np.zeros((7, 8)), 1: np.zeros((7, 8))}
                entropy_sum = np.zeros(7)
                maxprob_sum = np.zeros(7)
                margin_sum = np.zeros(7)
                expert_cos_sum = np.zeros(7)
                expert_residual_sum = np.zeros(7)
                expert_cos_batches = 0
                tokens = 0
                sequences = 0
                batches = 0
                with torch.no_grad():
                    for batch in iter_source_batches(DATA, "validation", source, 512,
                                                     batch_size,
                                                     max_batches=batches_per_source):
                        if time.monotonic() - started > deadline_s:
                            truncated = True
                            break
                        x, _y, valid, _chars = source_tensors(batch, "cpu")
                        model(x, return_metrics=True, return_router=True)
                        for layer, logits in probe.logits.items():
                            probs = torch.softmax(logits, dim=-1)
                            entropy_sum[layer] += float(
                                -(probs * probs.clamp_min(1e-12).log())
                                .sum(dim=-1).mean() / math.log(8))
                            maxprob_sum[layer] += float(
                                probs.max(dim=-1).values.mean())
                            top_values, top_indices = torch.topk(probs, 2, dim=-1)
                            margin_sum[layer] += float(
                                (top_values[:, 0] - top_values[:, 1]).mean())
                            for rank in range(2):
                                counts_top2[layer] += np.bincount(
                                    top_indices[:, rank].numpy(), minlength=8)
                            counts_top1[layer] += np.bincount(
                                top_indices[:, 0].numpy(), minlength=8)
                            half = 0 if batches < batches_per_source / 2 else 1
                            halves[half][layer] += np.bincount(
                                top_indices[:, 0].numpy(), minlength=8)
                            rows = torch.arange(top_indices.shape[0])
                            positions = (rows % 511) < 256
                            position_halves[0][layer] += np.bincount(
                                top_indices[positions, 0].numpy(), minlength=8)
                            position_halves[1][layer] += np.bincount(
                                top_indices[~positions, 0].numpy(), minlength=8)
                        if probe.expert_cos:
                            for layer, value in probe.expert_cos.items():
                                expert_cos_sum[layer] += value
                            for layer, value in probe.expert_residual.items():
                                expert_residual_sum[layer] += value
                            expert_cos_batches += 1
                        tokens += int(valid.sum())
                        sequences += x.shape[0]
                        batches += 1
                entry = {"sequences": sequences, "batches": batches,
                         "tokens": tokens,
                         "seconds": time.monotonic() - seconds_start,
                         "layers": {}}
                for layer in range(7):
                    top1 = counts_top1[layer] / max(1.0, counts_top1[layer].sum())
                    top2 = counts_top2[layer] / max(1.0, counts_top2[layer].sum())
                    within = None
                    if halves[0][layer].sum() and halves[1][layer].sum():
                        within = _js_divergence(halves[0][layer], halves[1][layer])
                    entry["layers"][str(layer + 1)] = {
                        "top1_share": [float(value) for value in top1],
                        "top2_share": [float(value) for value in top2],
                        "softmax_entropy_norm": float(entropy_sum[layer]
                                                      / max(1, batches)),
                        "softmax_max_prob_mean": float(maxprob_sum[layer]
                                                       / max(1, batches)),
                        "softmax_top2_margin_mean": float(margin_sum[layer]
                                                          / max(1, batches)),
                        "expert_output_cosine_mean": float(
                            expert_cos_sum[layer]
                            / max(1, expert_cos_batches)),
                        "expert_residual_ratio_mean": float(
                            expert_residual_sum[layer]
                            / max(1, expert_cos_batches)),
                        "within_source_half_js_bits": within,
                        "position_half_js_bits": _js_divergence(
                            position_halves[0][layer],
                            position_halves[1][layer]) if (
                                position_halves[0][layer].sum()
                                and position_halves[1][layer].sum()) else None,
                    }
                result[label][source] = entry
                if truncated:
                    break
        finally:
            probe.close()
        if truncated:
            break

    divergence = {}
    for label in result:
        layers = {}
        for layer in range(7):
            key = str(layer + 1)
            pair = {}
            for first, second in ((SOURCES[0], SOURCES[1]),
                                  (SOURCES[0], SOURCES[2]),
                                  (SOURCES[1], SOURCES[2])):
                pair[f"{first} vs {second}"] = _js_divergence(
                    result[label][first]["layers"][key]["top1_share"],
                    result[label][second]["layers"][key]["top1_share"])
            within = [result[label][source]["layers"][key][
                "within_source_half_js_bits"] for source in SOURCES
                if source in result[label]
                and result[label][source]["layers"][key][
                    "within_source_half_js_bits"] is not None]
            layers[key] = {"between_source_js_bits": pair,
                           "within_source_half_js_bits_mean": (
                               statistics.mean(within) if within else None),
                           "position_half_js_bits_mean": statistics.mean(
                               result[label][source]["layers"][key][
                                   "position_half_js_bits"] for source in SOURCES
                               if source in result[label]
                               and result[label][source]["layers"][key][
                                   "position_half_js_bits"] is not None)}
        divergence[label] = layers

    return {
        "protocol": {
            "split": "validation only",
            "sequences_per_source": batches_per_source * batch_size,
            "total_sequences_per_model": batches_per_source * batch_size
            * len(SOURCES),
            "seq_len": 512,
            "models": ["init (step 0 state)", "step500 (best artifact)"],
            "deadline_seconds": deadline_s,
            "elapsed_seconds": time.monotonic() - started,
        },
        "sources": result,
        "divergence": divergence,
        "truncated": truncated,
    }


def part_merge(paths):
    telemetry = _read_json(paths["telemetry"])
    checkpoints = _read_json(paths["checkpoints"])
    routing = _read_json(paths["routing"])

    perf = telemetry["performance"]
    coverage = telemetry["coverage"]
    learning = telemetry["learning_dynamics"]

    parameters = checkpoints["states"]["500"]["param_groups_total"]
    positions = coverage["positions"]
    estimated_training_s = perf["estimated_training_s"]
    flops = 6.0 * parameters * positions
    js_ceiling = _js_divergence(np.ones(8) / 8, np.eye(8)[0])

    source_js = {}
    for label in routing["divergence"]:
        layers = routing["divergence"][label]
        values = [statistics.mean(layers[str(layer)]["between_source_js_bits"].values())
                  for layer in range(1, 8)]
        source_js[label] = {
            "mean_bits": statistics.mean(values),
            "min_bits": min(values),
            "max_bits": max(values),
            "fraction_of_ceiling": statistics.mean(values) / js_ceiling,
        }

    expert_output = {}
    for label in routing["sources"]:
        per_layer = {}
        for layer in range(1, 8):
            values = [routing["sources"][label][source]["layers"][str(layer)]
                      for source in routing["sources"][label]]
            per_layer[str(layer)] = {
                "expert_output_cosine_mean": statistics.mean(
                    entry["expert_output_cosine_mean"] for entry in values),
                "expert_residual_ratio_mean": statistics.mean(
                    entry["expert_residual_ratio_mean"] for entry in values),
                "softmax_entropy_norm_mean": statistics.mean(
                    entry["softmax_entropy_norm"] for entry in values),
                "softmax_max_prob_mean": statistics.mean(
                    entry["softmax_max_prob_mean"] for entry in values),
            }
        expert_output[label] = per_layer

    return {
        "meta": {
            "analysis": "v3.1-M post-training analysis",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "CPU-only, read-only",
            "run": telemetry["run"],
            "reference_doc": "veylith/v3/docs/V3.1-L-DENSE-RUN.md",
            "command": {
                "telemetry": "python veylith/v3/analysis/v31m_analysis.py telemetry --out telemetry.json",
                "checkpoints": "python veylith/v3/analysis/v31m_analysis.py checkpoints --out checkpoints.json",
                "routing": "python veylith/v3/analysis/v31m_analysis.py routing --out routing.json",
            },
        },
        "evidence": telemetry["evidence"],
        "learning": {
            "validation": learning["validation"],
            "intervals": learning["intervals"],
            "totals": learning["totals"],
            "train_ce": learning["train_ce"],
            "grad_norm": learning["grad_norm"],
            "aux": {
                "per_layer_mean": learning["aux_per_layer_mean"],
                "per_layer_final": learning["aux_per_layer_final"],
                "per_layer_minimum": learning["aux_per_layer_minimum"],
                "final_over_minimum": learning["aux_final_over_minimum"],
            },
            "lr": {"matches_schedule": learning["lr_matches_schedule"],
                   "peak": learning["lr_peak"],
                   "final": learning["lr_final"]},
            "train_val_gap": learning["train_val_gap"],
            "dropout_during_training": learning["dropout_during_training"],
        },
        "router": {
            "gate_weights": checkpoints["router"],
        },
        "specialization": {
            "expert_weights": checkpoints["experts"],
            "softmax_and_outputs": expert_output,
            "source_divergence": {
                "ceiling_bits": js_ceiling,
                "per_model": source_js,
                "by_layer": routing["divergence"],
            },
            "protocol": routing["protocol"],
            "truncated": routing["truncated"],
            "verdict": (
                "Balanced utilization and confident per-token routing are measured. "
                "Distinct routing by data source is weak: between-source JS is 0.9 to 2.3 "
                "percent of the divergence ceiling. Learned expert specialization is not "
                "supported: expert outputs became more similar and the expert residual "
                "ratio fell, while weight-space cross-expert cosine stayed near zero."),
        },
        "checkpoints": {
            "states": checkpoints["states"],
            "artifact": checkpoints["artifact"],
            "best_state_hashes": checkpoints["best_state_hashes"],
        },
        "performance": {
            **perf,
            "estimate": {
                "assumption": (
                    "6ND with N = all 98,340,864 parameters, because dense MoE "
                    "activates every expert for every token; attention score and "
                    "value matmuls ignored (about 1.5 percent of per-token FLOPs). "
                    "CE targets, not padded positions, are the token count."),
                "flops": flops,
                "tflops_per_s_over_estimated_training_s": flops
                / estimated_training_s / 1e12,
            },
        },
        "coverage": coverage,
        "limitations": [
            "One seed, one configuration, 500 optimizer updates, 16.35 million CE targets.",
            "17.08 percent of one epoch; no full corpus pass and no repetition.",
            "Training CE is a single 64-sequence microbatch mean with dropout 0.1 active.",
            "Validation CE uses the full split in eval mode, so train and validation "
            "levels are not directly comparable and cannot be used to test overfitting.",
            "The v3.1-L timer lacked explicit GPU synchronization; timings are within "
            "about 2 percent by wall-clock arithmetic.",
            "Router telemetry entropy is computed over top-2 assignment counts, not over "
            "the softmax distribution; the two differ and are reported separately here.",
            "Specialization inference used 120 validation sequences per model at seq_len "
            "512, on three source-pure validation sets only; the test split was not read.",
            "Source-level routing divergence is a coarse probe; token- or feature-level "
            "routing structure was not identified.",
            "No GPU numerical resume parity was tested, and none is claimed.",
            "Theoretical compute is an estimate under documented assumptions, not a "
            "hardware counter measurement.",
        ],
        "recommendations": [
            {
                "rank": 1,
                "option": "B. Investigate expert specialization with bounded probes",
                "expected_information_gain": (
                    "Decides whether the MoE layout adds anything over a dense MLP at this "
                    "scale, and whether the aux weight or routing objective needs revision."),
                "main_risks": (
                    "Cheap probes may again be inconclusive; a change of routing objective "
                    "would alter frozen Architecture A and needs separate authorization."),
                "gpu_work": "None for analysis; one short bounded run if a new objective is approved.",
                "architecture_unchanged": True,
            },
            {
                "rank": 2,
                "option": "A. Continue Dense-Combined training",
                "expected_information_gain": (
                    "Measures the still-decreasing validation curve and tests stability and "
                    "resume parity beyond 500 updates."),
                "main_risks": (
                    "Linear GPU time and disk growth (1.57 GB per state); no new capability "
                    "at this scale; improvement per update already fell by about 10x."),
                "gpu_work": "One bounded run, new budget and checkpoint policy required.",
                "architecture_unchanged": True,
            },
            {
                "rank": 3,
                "option": "C. Revisit Sparse-MoE after a stable PyTorch/ROCm release",
                "expected_information_gain": (
                    "Would cut dense-MoE expert FLOPs by E/k = 4x and remove the workaround "
                    "cost."),
                "main_risks": (
                    "The fault reproduced on torch 2.14.1+rocm7.2; no stable 2.15/2.16 exists. "
                    "Any upgrade changes the validated environment and needs the full ladder."),
                "gpu_work": "Environment validation plus a sparse smoke and short run later.",
                "architecture_unchanged": True,
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["telemetry", "checkpoints", "routing",
                                         "merge"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--telemetry")
    parser.add_argument("--checkpoints")
    parser.add_argument("--routing")
    args = parser.parse_args()
    if args.part == "telemetry":
        payload = part_telemetry()
    elif args.part == "checkpoints":
        payload = part_checkpoints()
    elif args.part == "routing":
        payload = part_routing()
    else:
        payload = part_merge({"telemetry": args.telemetry,
                              "checkpoints": args.checkpoints,
                              "routing": args.routing})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("wrote", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
