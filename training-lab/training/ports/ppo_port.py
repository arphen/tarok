"""Port: PPO training engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from training.entities.training_config import TrainingConfig


class PPOPort(ABC):
    @abstractmethod
    def setup(self, weights: dict, config: TrainingConfig, device: str) -> None:
        """Initialize internal trainer with model weights."""

    @abstractmethod
    def set_lr(self, lr: float) -> None:
        """Update the optimizer learning rate for the next update."""

    @abstractmethod
    def set_imitation_coef(self, coef: float) -> None:
        """Update the imitation/oracle-distillation coefficient for next update."""

    @abstractmethod
    def update(self, raw_experiences: dict[str, Any], nn_seats: list[int]) -> tuple[dict[str, float], dict]:
        """Run PPO update on learner (nn) seats only. Return (metrics_dict, new_weights)."""
