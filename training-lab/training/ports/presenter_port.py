"""Port: presenter (output) interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.entities.training_run import TrainingRun


class PresenterPort(ABC):
    @abstractmethod
    def on_model_loaded(self, identity: ModelIdentity, save_dir: str) -> None: ...

    @abstractmethod
    def on_device_selected(self, device: str) -> None: ...

    @abstractmethod
    def on_training_plan(self, config: TrainingConfig) -> None: ...

    @abstractmethod
    def on_initial_benchmark(self, placement: float, n_games: int, seats: str, elapsed: float) -> None: ...

    @abstractmethod
    def on_training_loop_start(self, config: TrainingConfig) -> None: ...

    @abstractmethod
    def on_iteration_start(self, iteration: int, total: int, elapsed: float) -> None: ...

    @abstractmethod
    def on_selfplay_start(self, config: TrainingConfig) -> None: ...

    @abstractmethod
    def on_selfplay_done(self, n_experiences: int, elapsed: float) -> None: ...

    @abstractmethod
    def on_ppo_start(self, config: TrainingConfig, iter_lr: float | None = None) -> None: ...

    @abstractmethod
    def on_ppo_done(self, metrics: dict[str, float], elapsed: float) -> None: ...

    @abstractmethod
    def on_benchmark_start(self, config: TrainingConfig) -> None: ...

    @abstractmethod
    def on_benchmark_done(self, placement: float, elapsed: float) -> None: ...

    @abstractmethod
    def on_iteration_done(self, prev_placement: float, curr_placement: float, elapsed: float) -> None: ...

    @abstractmethod
    def on_training_complete(self, run: TrainingRun) -> None: ...
