"""Atomic checkpoints, resume state and RNG handling for v3.1-A."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import site
import sys

import numpy as np
import torch

from .determinism import PREFILL_ENV

STATE_FORMAT = 1
ARTIFACT_FORMAT = 1


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def environment_provenance() -> dict:
    path = os.environ.get("VEYLITH_ENV_FINGERPRINT")
    return {
        "sys_prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "user_site": bool(site.ENABLE_USER_SITE),
        "hip": getattr(torch.version, "hip", None),
        "fingerprint_path": path,
        "fingerprint_sha256": _file_sha256(path),
        "deterministic_prefill": {
            "effective": bool(
                torch.utils.deterministic.fill_uninitialized_memory),
            "env": os.environ.get(PREFILL_ENV),
        },
    }


def versions() -> dict:
    import sentencepiece
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sentencepiece": getattr(sentencepiece, "__version__", "unknown"),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "environment": environment_provenance(),
    }


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    for key in ("python", "numpy", "torch"):
        if key not in state:
            raise ValueError(f"RNG state missing {key}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, path: str) -> str:
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".tmp"
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    try:
        directory = os.open(os.path.dirname(target), os.O_RDONLY)
        os.fsync(directory)
        os.close(directory)
    except OSError:
        pass
    return target


def save_state(path: str, *, model, optimizer, schedule, step: int,
               tokens_seen: int, loader_epoch: int, loader_batch: int,
               best_metric: float, best_step: int | None, best_tokens_seen: int,
               best_state, best_metrics, config, data_manifest_sha256: str,
               dataset: dict, fingerprint: str) -> str:
    payload = {
        "format": STATE_FORMAT,
        "kind": "train_state",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "loader": {"epoch": int(loader_epoch), "batch": int(loader_batch)},
        "best_metric": float(best_metric),
        "best_step": best_step,
        "best_tokens_seen": int(best_tokens_seen),
        "best_state": best_state,
        "best_metrics": best_metrics,
        "config": config.to_dict(),
        "model_config": config.model.to_dict(),
        "data_manifest_sha256": data_manifest_sha256,
        "dataset": dataset,
        "fingerprint": fingerprint,
        "rng": capture_rng(),
        "versions": versions(),
    }
    return atomic_save(payload, path)


def load_state(path: str, expected_fingerprint: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("kind") != "train_state" or state.get("format") != STATE_FORMAT:
        raise ValueError(
            f"{path} is not a v3.1 training state "
            f"(format={state.get('format')!r}, kind={state.get('kind')!r})")
    if state.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "resume fingerprint mismatch: the model, data manifest, optimizer, "
            "precision or seed changed since this state was written")
    for key in ("model", "optimizer", "schedule", "step", "tokens_seen",
                "loader", "rng", "data_manifest_sha256"):
        if key not in state:
            raise ValueError(f"resume state missing {key}")
    return state


def move_optimizer_state(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_artifact(path: str, *, model, config, data_manifest_sha256: str,
                  dataset: dict, tokenizer: dict, selection: dict,
                  metrics: dict, step: int, tokens_seen: int,
                  fingerprint: str) -> str:
    payload = {
        "format": ARTIFACT_FORMAT,
        "kind": "model",
        "model": model.state_dict(),
        "model_config": config.model.to_dict(),
        "train_config": config.to_dict(),
        "tokenizer": tokenizer,
        "data_manifest_sha256": data_manifest_sha256,
        "dataset": dataset,
        "fingerprint": fingerprint,
        "selection": selection,
        "metrics": metrics,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "versions": versions(),
    }
    return atomic_save(payload, path)
