"""IterationResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    seat_config_used: str = "nn,nn,nn,nn"
    mean_scores: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # Per-opponent-seat game outcomes: {seat_idx: (learner_wins, opp_wins, draws)}
    seat_outcomes: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    learner_elo: float = 0.0

    @property
    def total_time(self) -> float:
        return self.selfplay_time + self.ppo_time + self.bench_time
