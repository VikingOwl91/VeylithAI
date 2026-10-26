"""v3.1-A architecture reference: configs, CPU model, accounting."""

from .accounting import (analytical_counts, active_parameter_count,
                         count_model_parameters, memory_estimate,
                         training_flops)
from .config import CANDIDATES, ModelConfig, get_candidate, tiny_config
from .model import GPT

__all__ = ["CANDIDATES", "ModelConfig", "GPT", "get_candidate", "tiny_config",
           "analytical_counts", "active_parameter_count", "count_model_parameters",
           "memory_estimate", "training_flops"]
