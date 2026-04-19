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
    def set_entropy_coef(self, coef: float) -> None:
        """Update the entropy coefficient for the next PPO update."""

    @abstractmethod
    def set_behavioral_clone_coef(self, coef: float) -> None:
        """Update the behavioral-cloning coefficient for the next PPO update."""

    @abstractmethod
    def update(self, raw_experiences: dict[str, Any], nn_seats: list[int]) -> tuple[dict[str, float], dict]:
        """Run PPO update on learner (nn) seats only. Return (metrics_dict, new_weights)."""

    @abstractmethod
    def load_human_data(self, data_dir: str) -> dict[str, Any] | None:
        """Load human replay data from directory. Returns None if no data found."""

    @abstractmethod
    def load_expert_data(self, teacher: str, num_games: int) -> dict[str, Any] | None:
        """Load expert replay data for behavioral cloning. Returns None if unavailable."""

    @abstractmethod
    def merge_experiences(self, primary: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        """Merge two experience dicts (e.g. self-play + human replay) into one."""
