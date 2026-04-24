"""TrainingConfig dataclass and LR schedule helper."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.entities.league import LeagueConfig

from training.entities.duplicate_config import DuplicateConfig

# Seat labels that represent learner agents (NN-backed, generate PPO experiences).
LEARNER_SEAT_LABELS: frozenset[str] = frozenset({"nn", "centaur"})


@dataclass(frozen=True)
class TrainingConfig:
    profile_name: str = "custom"
    seats: str = "nn,bot_v5,bot_v5,bot_v5"
    bench_seats: str | None = None
    iterations: int = 10
    games: int = 10_000
    outplace_session_size: int = 50
    bench_games: int = 3_000
    benchmark_checkpoints: tuple[int, ...] = (0, 4, 7)
    best_model_metric: str = "loss"
    ppo_epochs: int = 6
    batch_size: int = 8192
    lr: float = 3e-4
    lr_schedule: str = "constant"
    lr_min: float | None = None
    explore_rate: float = 0.10
    explore_rate_min: float | None = None
    explore_rate_schedule: str = "constant"  # constant | linear | cosine | elo
    device: str = "auto"
    save_dir: str = "data/checkpoints/training_run"
    concurrency: int = 128
    lapajne_mc_worlds: int | None = None
    lapajne_mc_sims: int | None = None
    centaur_handoff_trick: int | None = None
    centaur_pimc_worlds: int | None = None
    centaur_endgame_solver: str | None = None
    centaur_alpha_mu_depth: int | None = None
    centaur_deterministic_seed: int | None = None
    imitation_coef: float = 0.3
    imitation_schedule: str = "constant"  # constant | linear | cosine | gaussian_elo
    imitation_coef_min: float = 0.0
    imitation_center_elo: float = 1500.0   # gaussian_elo: bell curve centre
    imitation_width_elo: float = 250.0     # gaussian_elo: bell curve σ
    # Behavioral cloning (action-level) controls.
    behavioral_clone_coef: float = 0.0
    behavioral_clone_schedule: str = "constant"  # constant | linear | cosine | exponential | geometric
    behavioral_clone_coef_min: float = 0.0
    behavioral_clone_teacher: str | None = None
    behavioral_clone_games_per_iteration: int = 0
    # PPO hyperparams (all have sensible defaults)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    policy_coef: float = 1.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    bid_entropy_coef: float | None = None
    entropy_schedule: str = "constant"  # constant | linear | cosine | exponential | geometric
    entropy_coef_min: float = 0.005
    iteration_runner_mode: str = "in-process"
    iteration_runner_restart_every: int = 10
    model_arch: str = "v4"
    # When creating a fresh model, include the privileged oracle critic head used for
    # imitation / distillation (matches oracle features from PrivilegedState).
    # Ignored when loading from a checkpoint — identity reflects what the checkpoint carries.
    oracle_critic: bool = True
    human_data_dir: str | None = None
    league: LeagueConfig | None = None
    duplicate: DuplicateConfig = field(default_factory=DuplicateConfig)

    def should_benchmark_initial(self) -> bool:
        return 0 in set(self.benchmark_checkpoints)

    def should_benchmark_iteration(self, iteration: int) -> bool:
        return iteration in set(self.benchmark_checkpoints)

    @property
    def nn_seat_indices(self) -> list[int]:
        """Indices of seats occupied by the NN player (nn or centaur)."""
        return [i for i, s in enumerate(self.seats.split(",")) if s.strip() in LEARNER_SEAT_LABELS]

    @property
    def bot_seat_indices(self) -> list[int]:
        """Indices of seats occupied by any bot (for imitation learning)."""
        return [i for i, s in enumerate(self.seats.split(",")) if s.strip() not in LEARNER_SEAT_LABELS]

    @property
    def effective_bench_seats(self) -> str:
        return self.bench_seats if self.bench_seats is not None else self.seats

    @property
    def effective_lr_min(self) -> float:
        return self.lr_min if self.lr_min is not None else self.lr / 10

    @property
    def effective_explore_rate_min(self) -> float:
        """Minimum epsilon for scheduled explore-rate decay.

        Defaults to ``explore_rate / 10`` when not explicitly configured,
        mirroring the behaviour of :attr:`effective_lr_min`.
        """
        if self.explore_rate_min is not None:
            return self.explore_rate_min
        return self.explore_rate / 10


def scheduled_lr(
    iteration: int, total_iterations: int,
    lr_max: float, lr_min: float, schedule: str,
) -> float:
    """Compute learning rate for a given iteration."""
    # "elo" means the elo-based decay policy handles the schedule; use constant base.
    if schedule in ("constant", "elo") or total_iterations <= 1:
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
    if schedule in ("exponential", "geometric"):
        # Multiplicative decay: coef_max * (coef_min / coef_max) ** frac
        # Clamps to coef_min when coef_max is zero or ratio is degenerate.
        if coef_max <= 0 or coef_min <= 0:
            return coef_max
        return coef_max * (coef_min / coef_max) ** frac
    return coef_max
