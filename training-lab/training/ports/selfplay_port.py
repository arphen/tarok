"""Port: self-play engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SelfPlayPort(ABC):
    @abstractmethod
    def run(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        explore_rate: float,
        concurrency: int,
        include_replay_data: bool = False,
        include_oracle_states: bool = False,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
    ) -> dict[str, Any]:
        """Run self-play games, return raw experience dict."""

    @abstractmethod
    def compute_run_stats(
        self,
        raw: dict[str, Any],
        seat_labels: list[str],
        session_size: int = 50,
    ) -> tuple[int, tuple[float, float, float, float], dict[int, tuple[int, int, int]]]:
        """Compute ``(n_learner, mean_scores, seat_outcomes)`` from one run.

        ``seat_outcomes`` is keyed by seat index (1..3) and stores
        ``(learner_outplaces, opponent_outplaces, draws)`` where outplacing is
        decided from cumulative session totals.  When more than one seat has
        label ``"nn"`` (``min_nn_per_game > 1``), outcomes are accumulated
        across *all* nn seats vs each opponent seat.
        """
