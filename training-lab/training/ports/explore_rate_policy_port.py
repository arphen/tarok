"""Port: self-play explore-rate (epsilon) policy for training iterations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from training.entities.training_config import TrainingConfig


class ExploreRatePolicyPort(ABC):
    @abstractmethod
    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        """Return self-play explore rate for a 1-based iteration and current learner Elo."""
