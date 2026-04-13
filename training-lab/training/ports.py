"""Ports — abstract interfaces that use cases depend on.

Adapters implement these. Use cases never import concrete infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from training.entities import IterationResult, ModelIdentity, TrainingConfig, TrainingRun


# ── Self-play engine ────────────────────────────────────────────────

class SelfPlayPort(ABC):
    @abstractmethod
    def run(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        explore_rate: float,
        concurrency: int,
    ) -> dict[str, Any]:
        """Run self-play games, return raw experience dict."""


# ── Benchmark engine ────────────────────────────────────────────────

class BenchmarkPort(ABC):
    @abstractmethod
    def measure_placement(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        concurrency: int,
        session_size: int,
    ) -> float:
        """Play greedy games and return average placement (1.0–4.0)."""


# ── PPO training engine ────────────────────────────────────────────

class PPOPort(ABC):
    @abstractmethod
    def setup(self, weights: dict, config: TrainingConfig, device: str) -> None:
        """Initialize internal trainer with model weights."""

    @abstractmethod
    def set_lr(self, lr: float) -> None:
        """Update the optimizer learning rate for the next update."""

    @abstractmethod
    def update(self, raw_experiences: dict[str, Any]) -> tuple[dict[str, float], dict]:
        """Run PPO update, return (metrics_dict, new_weights)."""


# ── Model persistence ──────────────────────────────────────────────

class ModelPort(ABC):
    @abstractmethod
    def load_weights(self, checkpoint_path: str) -> tuple[dict, int, bool]:
        """Load checkpoint → (state_dict, hidden_size, oracle_critic)."""

    @abstractmethod
    def create_new(self, hidden_size: int, oracle: bool) -> dict:
        """Create fresh random weights → state_dict."""

    @abstractmethod
    def export_for_inference(self, weights: dict, hidden_size: int, oracle: bool, path: str) -> None:
        """Export model to format the self-play engine can load."""

    @abstractmethod
    def save_checkpoint(
        self, weights: dict, hidden_size: int, oracle: bool,
        iteration: int, loss: float, placement: float, path: str,
    ) -> None:
        """Save training checkpoint."""

    @abstractmethod
    def copy_best(self, src: str, dst: str) -> None:
        """Copy the best checkpoint."""


# ── Config loading ──────────────────────────────────────────────────

class ConfigPort(ABC):
    @abstractmethod
    def load(self, path: str) -> dict[str, Any]:
        """Load config from file → raw dict."""


# ── Presenter (output) ─────────────────────────────────────────────

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
