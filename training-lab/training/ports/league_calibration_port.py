"""Port: league Elo calibration strategy.

Encapsulates two calibration operations so they can be swapped between the
historical mixed-seat self-play approach and a duplicate-tournament
(paired-deal) approach:

* ``calibrate_initial`` — bootstrap all opponent Elos (and learner Elo) at
  the very beginning of a league run.
* ``calibrate_candidate`` — compute the implied Elo of a candidate NN
  checkpoint (the live model) against the existing pool, used by
  ``MaintainLeaguePool`` when deciding whether to admit a new snapshot.

All concrete adapters must accept bot-token opponents (``bot_v5``, etc.);
duplicate-tournament adapters that cannot natively pair against such
opponents should fall back to self-play for those entries so behavior
remains correct for the whole pool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from training.entities.league import LeaguePool


class LeagueCalibrationPort(ABC):
    @abstractmethod
    def calibrate_initial(
        self,
        *,
        pool: "LeaguePool",
        model_path: str,
        concurrency: int,
        session_size: int,
        n_games_per_pair: int,
        anchor_name: str | None,
        anchor_elo: float,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        on_mixed_result: Callable[[int, int, str, tuple[str, str, str], tuple[float, float, float, float]], None] | None = None,
        variant: int = 0,
    ) -> bool:
        """Bootstrap Elo ratings for every pool entry + the learner.

        Returns ``True`` if calibration ran (pool was mutated), ``False`` if
        the adapter chose to skip (e.g. zero games requested).
        """

    @abstractmethod
    def calibrate_candidate(
        self,
        *,
        pool: "LeaguePool",
        model_path: str,
        concurrency: int,
        session_size: int,
        n_games_per_opponent: int,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        variant: int = 0,
    ) -> float | None:
        """Return the implied Elo of ``model_path`` vs the current pool.

        Returns ``None`` when no usable opponent games were collected.
        """
