"""Entity: the raw and derived output from one self-play collection phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.entities.duplicate_iteration_stats import DuplicateIterationStats


@dataclass
class ExperienceBundle:
    """Holds the self-play payload for one iteration.

    ``raw`` is the opaque tensor dict returned by the Rust engine.  It is
    intentionally mutable so the orchestrator can release it (``del bundle.raw``)
    once the PPO update has consumed it, freeing device memory before benchmark
    and checkpoint I/O.
    """

    raw: Any  # dict[str, Tensor] — opaque Rust self-play output
    nn_seats: list[int]
    seat_labels: list[str]
    effective_seats: str
    n_total: int
    n_learner: int
    mean_scores: tuple  # (p0, p1, p2, p3) floats
    seat_outcomes: dict  # seat_idx -> (learner_outplaces, opponent_outplaces, draws)
    sp_time: float
    learner_contract_stats: dict[str, dict[str, float | int]] | None = None
    # Populated only in duplicate-RL mode (see CollectDuplicateExperiences).
    # ``None`` for regular self-play iterations.
    duplicate_stats: DuplicateIterationStats | None = None
