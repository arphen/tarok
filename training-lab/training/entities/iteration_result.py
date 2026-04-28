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
    # Per-opponent-seat comparison outcomes: {seat_idx: (learner_outplaces, opponent_outplaces, draws)}
    seat_outcomes: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # Duplicate-mode only: pairwise outplace outcomes bucketed by opponent
    # league token (not seat index). Consumed by UpdateLeagueElo to update
    # Elo for the specific opponents the learner met this iteration, which
    # rotate across seats pod-by-pod and therefore don't fit the seat-
    # indexed ``seat_outcomes`` contract. Empty in regular self-play mode.
    opponent_outcomes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    # Duplicate-mode only: mean (learner_score − shadow_score) / 100 across
    # matched active/shadow games this iteration. ``None`` when not in
    # duplicate mode; surfaced by the presenter as an "am I improving?"
    # diagnostic.
    mean_duplicate_advantage: float | None = None
    duplicate_advantage_std: float | None = None
    n_duplicate_games: int = 0
    learner_contract_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)
    learner_elo: float = 0.0

    @property
    def total_time(self) -> float:
        return self.selfplay_time + self.ppo_time + self.bench_time
