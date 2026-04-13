"""Entities — pure data, no dependencies on frameworks or I/O."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Slovenian names ─────────────────────────────────────────────────

_SL_FIRST = [
    "Ana", "Maja", "Eva", "Nina", "Sara", "Lara", "Anja", "Ema",
    "Katja", "Tina", "Živa", "Pia", "Lina", "Zala", "Neža",
    "Teja", "Rok", "Urša", "Janja", "Alja", "Špela", "Manca",
    "Petra", "Metka", "Monika", "Irena", "Andreja", "Brigita",
    "Vera", "Marta", "Klara", "Nataša", "Polona", "Mateja",
    "Lea", "Nika", "Hana", "Julija", "Lucija", "Tamara",
]
_SL_LAST = [
    "Novak", "Horvat", "Kovačič", "Krajnc", "Zupančič",
    "Potočnik", "Kos", "Golob", "Vidmar", "Kolar",
    "Mlakar", "Bizjak", "Žagar", "Turk", "Hribar",
    "Kavčič", "Hočevar", "Rupnik", "Debeljak", "Černe",
    "Gregorčič", "Vesel", "Kern", "Starič", "Oblak",
    "Pečnik", "Gorenc", "Šuštar", "Bogataj", "Kranjc",
]


def random_slovenian_name() -> str:
    return f"{random.choice(_SL_FIRST)}_{random.choice(_SL_LAST)}"


def name_from_checkpoint(path: str) -> str | None:
    p = Path(path)
    parent = p.parent.name
    if parent and parent not in ("checkpoints", "training_run", "pinned", "hall_of_fame", "."):
        return parent
    m = re.match(r"hof_([A-Z]\w+_[A-Z][a-z]+)", p.stem)
    return m.group(1) if m else None


# ── Config ──────────────────────────────────────────────────────────

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


# ── Model identity ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelIdentity:
    name: str
    hidden_size: int
    oracle_critic: bool
    is_new: bool


# ── Iteration result ───────────────────────────────────────────────

@dataclass(frozen=True)
class IterationResult:
    iteration: int
    placement: float
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    n_experiences: int
    selfplay_time: float
    ppo_time: float
    bench_time: float

    @property
    def total_time(self) -> float:
        return self.selfplay_time + self.ppo_time + self.bench_time


# ── Training run (mutable aggregate root) ──────────────────────────

@dataclass
class TrainingRun:
    config: TrainingConfig
    identity: ModelIdentity
    initial_placement: float = 0.0
    results: list[IterationResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def placements(self) -> list[float]:
        return [self.initial_placement] + [r.placement for r in self.results]

    @property
    def best_placement(self) -> float:
        return min(self.placements)

    @property
    def best_iteration(self) -> int:
        p = self.placements
        return p.index(min(p))

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time

    @property
    def improved(self) -> bool:
        return self.best_iteration > 0
