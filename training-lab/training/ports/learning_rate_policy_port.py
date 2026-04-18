"""Port: learning-rate policy for training iterations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from training.entities.training_config import TrainingConfig


class LearningRatePolicyPort(ABC):
    @abstractmethod
    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float) -> float:
        """Return learning rate for a 1-based iteration and current learner Elo."""
