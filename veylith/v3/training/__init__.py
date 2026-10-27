"""v3.1-A training infrastructure (NTP + router aux, no MTP)."""

from .checkpoint import load_state, save_artifact, save_state
from .config import TrainConfig
from .evaluate import evaluate
from .train import train

__all__ = ["TrainConfig", "train", "evaluate", "save_state", "load_state",
           "save_artifact"]
