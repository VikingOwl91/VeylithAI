"""Training configuration and resume fingerprint for v3.1-A."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

from architecture.config import ModelConfig


def canonical_json(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    model: ModelConfig
    device: str = "cpu"
    precision: str = "fp32"       # fp32 | bf16
    batch_size: int = 8           # microbatch
    grad_accum: int = 8           # effective batch = batch_size * grad_accum
    seq_len: int = 512
    max_steps: int = 1000
    stop_after: int | None = None
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    seed: int = 0
    eval_interval: int = 50
    eval_batch_size: int = 8
    eval_max_batches: int | None = None
    train_gap_batches: int = 8
    save_every: int = 50
    log_every: int = 10
    select_metric: str = "macro_main_ce"
    deterministic: bool = True
    checkpoint: str | None = None
    state_path: str | None = None
    curve_jsonl: str | None = None
    resume: str | None = None
    allow_large_cpu: bool = False
    single_pass: bool = False
    save_steps: tuple = ()
    save_initial_state: bool = False
    exact_chars: bool = True
    max_wall_seconds: float | None = 5400.0
    eval_steps: tuple = ()
    allow_unsafe_sparse: bool = False
    max_vram_gb: float | None = None

    def __post_init__(self):
        if self.checkpoint is None:
            self.checkpoint = os.path.join(self.out_dir, "ckpt_best.pt")
        if self.state_path is None:
            self.state_path = os.path.join(self.out_dir, "ckpt_state.pt")
        if self.curve_jsonl is None:
            self.curve_jsonl = os.path.join(self.out_dir, "curve.jsonl")

    def validate(self) -> None:
        problems = []
        if self.device not in ("cpu", "cuda"):
            problems.append("device must be cpu or cuda")
        if self.precision not in ("fp32", "bf16"):
            problems.append("precision must be fp32 or bf16")
        if self.precision == "bf16" and self.device == "cpu":
            problems.append("bf16 on CPU is not part of this CPU reference path")
        if self.batch_size <= 0 or self.grad_accum <= 0:
            problems.append("batch_size and grad_accum must be positive")
        if not 2 <= self.seq_len <= self.model.block_size:
            problems.append("seq_len must be in [2, model.block_size]")
        if self.max_steps <= 0:
            problems.append("max_steps must be positive")
        if self.stop_after is not None and not 0 < self.stop_after <= self.max_steps:
            problems.append("stop_after must be in (0, max_steps]")
        if self.lr <= 0 or not 0 <= self.min_lr_ratio <= 1:
            problems.append("invalid learning rate configuration")
        if self.warmup_steps < 0:
            problems.append("warmup_steps must be non-negative")
        if self.eval_interval <= 0 or self.save_every <= 0:
            problems.append("eval_interval and save_every must be positive")
        if self.grad_clip < 0:
            problems.append("grad_clip must be non-negative")
        if self.select_metric not in ("macro_main_ce", "token_main_ce",
                                      "main_ce"):
            problems.append("select_metric must be macro_main_ce or token_main_ce")
        if not self.data_root or not os.path.exists(
                os.path.join(self.data_root, "corpus", "manifest.json")):
            problems.append(f"no corpus manifest under {self.data_root}")
        if self.max_wall_seconds is not None and self.max_wall_seconds <= 0:
            problems.append("max_wall_seconds must be positive or None")
        if self.max_vram_gb is not None and self.max_vram_gb <= 0:
            problems.append("max_vram_gb must be positive or None")
        if self.eval_steps:
            steps = [int(step) for step in self.eval_steps]
            if steps != sorted(set(steps)):
                problems.append("eval_steps must be sorted and unique")
            if any(step <= 0 or step > self.max_steps for step in steps):
                problems.append("eval_steps must be in (0, max_steps]")
        if (self.device == "cuda" and self.precision == "bf16"
                and self.model.moe_impl == "sparse"
                and not self.allow_unsafe_sparse):
            problems.append(
                "refusing bf16 sparse-MoE training on the known-affected "
                "PyTorch 2.14.1/ROCm stack; use --moe dense, or for a "
                "separately authorized research run set "
                "--allow-unsafe-sparse (see "
                "veylith/v3/docs/V3.1-J-SPARSE-REPRODUCER.md)")
        if self.save_steps:
            steps = [int(step) for step in self.save_steps]
            if steps != sorted(set(steps)):
                problems.append("save_steps must be sorted and unique")
            if any(step <= 0 or step > self.max_steps for step in steps):
                problems.append("save_steps must be in (0, max_steps]")
            if "{step" not in (self.state_path or ""):
                problems.append(
                    "save_steps requires a step-suffixed state_path "
                    "(for example ckpt_state_step{step}.pt)")
            if self.save_initial_state and "{step" not in (self.state_path or ""):
                problems.append("save_initial_state requires a step-suffixed state_path")
        if self.single_pass and "{step" not in (self.state_path or ""):
            problems.append(
                "single_pass requires a step-suffixed state_path "
                "(for example ckpt_state_step{step}.pt)")
        if problems:
            raise ValueError("; ".join(problems))

    def fingerprint(self, data_manifest_sha256: str) -> str:
        payload = {
            "model": self.model.to_dict(),
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "seq_len": self.seq_len,
            "lr": self.lr,
            "min_lr_ratio": self.min_lr_ratio,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "select_metric": self.select_metric,
            "data_manifest_sha256": data_manifest_sha256,
        }
        return sha256_bytes(canonical_json(payload))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        return payload
