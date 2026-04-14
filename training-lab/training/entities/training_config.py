"""TrainingConfig dataclass and LR schedule helper."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    seats: str = "nn,bot_v5,bot_v5,bot_v5"
    bench_seats: str | None = None
    iterations: int = 10
    games: int = 10_000
    bench_games: int = 10_000
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
    model_arch: str = "v2"

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
