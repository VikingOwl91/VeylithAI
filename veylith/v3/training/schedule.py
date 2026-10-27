"""Step-based learning-rate schedule: linear warmup, cosine to min ratio."""

from __future__ import annotations

import math
from dataclasses import dataclass


def lr_at(step: int, base_lr: float, warmup_steps: int, max_steps: int,
          min_lr_ratio: float = 0.1) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    span = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


@dataclass
class CosineSchedule:
    base_lr: float
    warmup_steps: int
    max_steps: int
    min_lr_ratio: float = 0.1
    step: int = 0

    def value(self) -> float:
        return lr_at(self.step, self.base_lr, self.warmup_steps,
                     self.max_steps, self.min_lr_ratio)

    def advance(self) -> float:
        value = self.value()
        self.step += 1
        return value

    def state_dict(self) -> dict:
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps, "min_lr_ratio": self.min_lr_ratio,
                "step": self.step}

    def load_state_dict(self, state: dict) -> None:
        for key in ("base_lr", "warmup_steps", "max_steps", "min_lr_ratio",
                    "step"):
            if key not in state:
                raise ValueError(f"schedule state missing {key}")
            setattr(self, key, state[key])
