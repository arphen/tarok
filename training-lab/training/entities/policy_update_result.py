"""Entity: the output produced by one PPO policy update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyUpdateResult:
    """Carries the new model weights and training metrics from a PPO update."""

    new_weights: Any  # dict[str, Tensor] — opaque model state dict
    metrics: dict  # {"total_loss", "policy_loss", "value_loss", "entropy", …}
    ppo_time: float
