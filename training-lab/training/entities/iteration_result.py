"""IterationResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass


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
