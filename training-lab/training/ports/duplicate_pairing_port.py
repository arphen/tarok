"""Port: duplicate pairing — how pods are built from the league pool.

The pairing policy is a separable research axis. Swapping rotation sizes,
opponent-sampling strategies, or per-pod deck reuse should not require
touching PPO batch preparation, reward computation, or the Rust engine.

Adapters live under ``training.adapters.duplicate``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.entities.duplicate_pod import DuplicatePod
    from training.entities.league import LeagueConfig, LeaguePool


class DuplicatePairingPort(ABC):
    """Builds the schedule of duplicate pods for one training iteration."""

    @abstractmethod
    def build_pods(
        self,
        pool: "LeaguePool | LeagueConfig | None",
        learner_seat_token: str,
        shadow_seat_token: str,
        n_pods: int,
        rng_seed: int,
    ) -> list["DuplicatePod"]:
        """Return ``n_pods`` pods for the requested iteration.

        ``pool`` is the live league pool (preferred) or static league config
        used to sample opponent identities (one triple per pod). When ``pool``
        is ``None`` or has no usable opponents, adapters may fall back to a
        fixed heuristic bot roster.

        ``learner_seat_token`` is the seat token that identifies the learner
        (typically ``"nn"``) — it must appear exactly once in every
        ``active_seatings`` entry. ``shadow_seat_token`` is the seat token
        used for the frozen snapshot — it replaces the learner token
        position-for-position in ``shadow_seatings``.

        ``rng_seed`` makes pod schedules reproducible across runs.
        """
