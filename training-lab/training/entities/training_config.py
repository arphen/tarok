"""TrainingConfig dataclass and LR schedule helper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.entities.league import LeagueConfig


@dataclass(frozen=True)
class TrainingConfig:
    seats: str = "nn,bot_v5,bot_v5,bot_v5"
    bench_seats: str | None = None
    iterations: int = 10
    games: int = 10_000
    bench_games: int = 3_000
    benchmark_checkpoints: tuple[int, ...] = (0, 4, 7)
    best_model_metric: str = "loss"
    ppo_epochs: int = 6
    batch_size: int = 8192
    lr: float = 3e-4
    lr_schedule: str = "constant"
    lr_min: float | None = None
    explore_rate: float = 0.10
    device: str = "auto"
    save_dir: str = "checkpoints/training_run"
    concurrency: int = 128
    imitation_coef: float = 0.3
    imitation_schedule: str = "constant"  # constant | linear | cosine
    imitation_coef_min: float = 0.0
    # PPO hyperparams (all have sensible defaults)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    memory_telemetry: bool = True
    memory_telemetry_every: int = 1
    model_arch: str = "v4"
    human_data_dir: str | None = None
    league: LeagueConfig | None = None

    def should_benchmark_initial(self) -> bool:
        return 0 in set(self.benchmark_checkpoints)

    def should_benchmark_iteration(self, iteration: int) -> bool:
        return iteration in set(self.benchmark_checkpoints)

    @property
    def nn_seat_indices(self) -> list[int]:
        """Indices of seats occupied by the NN player."""
        return [i for i, s in enumerate(self.seats.split(",")) if s.strip() == "nn"]

    @property
    def bot_seat_indices(self) -> list[int]:
        """Indices of seats occupied by any bot (for imitation learning)."""
        return [i for i, s in enumerate(self.seats.split(",")) if s.strip() != "nn"]

    @property
    def effective_bench_seats(self) -> str:
        return self.bench_seats if self.bench_seats is not None else self.seats

    @property
    def effective_lr_min(self) -> float:
        return self.lr_min if self.lr_min is not None else self.lr / 10

    def scheduled_imitation_coef(self, iteration_1based: int) -> float:
        """Imitation/oracle-distillation coef for a 1-based training iteration."""
        return scheduled_coef(
            iteration=max(0, iteration_1based - 1),
            total_iterations=self.iterations,
            coef_max=self.imitation_coef,
            coef_min=self.imitation_coef_min,
            schedule=self.imitation_schedule,
        )


def scheduled_lr(
    iteration: int, total_iterations: int,
    lr_max: float, lr_min: float, schedule: str,
) -> float:
    """Compute learning rate for a given iteration."""
    if schedule == "constant" or total_iterations <= 1:
        return lr_max
    frac = iteration / (total_iterations - 1)  # 0.0 → 1.0
    if schedule == "linear":
        return lr_max + (lr_min - lr_max) * frac
    if schedule == "cosine":
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * frac))
    return lr_max


def scheduled_coef(
    iteration: int,
    total_iterations: int,
    coef_max: float,
    coef_min: float,
    schedule: str,
) -> float:
    """Generic schedule helper for coefficients such as imitation/distillation."""
    if schedule == "constant" or total_iterations <= 1:
        return coef_max
    frac = iteration / (total_iterations - 1)  # 0.0 -> 1.0
    if schedule == "linear":
        return coef_max + (coef_min - coef_max) * frac
    if schedule == "cosine":
        return coef_min + 0.5 * (coef_max - coef_min) * (1 + math.cos(math.pi * frac))
    return coef_max
