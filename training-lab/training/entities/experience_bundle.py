"""Entity: the raw and derived output from one self-play collection phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    seat_outcomes: dict  # seat_idx -> (wins, draws, losses)
    sp_time: float
