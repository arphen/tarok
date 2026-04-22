"""Port: iteration runner — abstracts in-process vs spawn execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig


class IterationRunnerPort(ABC):
    @abstractmethod
    def setup(self, weights: dict, config: TrainingConfig, device: str) -> None:
        """Initialize the runner (e.g. set up PPO, load model). Called once before the loop."""

    @abstractmethod
    def run_iteration(
        self,
        i: int,
        config: TrainingConfig,
        identity: ModelIdentity,
        ts_path: str,
        save_dir: Path,
        *,
        prev_placement: float,
        iter_lr: float | None,
        iter_imitation_coef: float | None,
        iter_behavioral_clone_coef: float | None,
        iter_entropy_coef: float | None,
        iter_explore_rate: float | None = None,
        seats_override: str | None,
        run_benchmark: bool,
    ) -> IterationResult:
        """Run a single training iteration and return the result."""

    def teardown(self) -> None:
        """Optional cleanup after the training loop finishes."""
