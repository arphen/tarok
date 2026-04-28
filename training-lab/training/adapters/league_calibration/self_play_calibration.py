"""Adapter: historical self-play calibration (mixed-seat greedy games).

Delegates to the existing ``CalibrateInitialLeagueElo`` use case for initial
calibration, and re-implements the candidate-snapshot calibration logic that
previously lived privately inside ``MaintainLeaguePool`` so both calibration
operations go through a single port.
"""

from __future__ import annotations

from itertools import cycle
from typing import Callable

from training.entities.league import LeaguePool
from training.ports.league_calibration_port import LeagueCalibrationPort
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.calibrate_initial_league_elo import CalibrateInitialLeagueElo
from training.use_cases.league_calibration_utils import _ELO_PER_PLACEMENT, _avg_placements


class SelfPlayLeagueCalibrationAdapter(LeagueCalibrationPort):
    """Default adapter: preserves pre-port behavior."""

    def __init__(self, selfplay: SelfPlayPort) -> None:
        self._selfplay = selfplay

    def calibrate_initial(
        self,
        *,
        pool: LeaguePool,
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
        return CalibrateInitialLeagueElo().execute(
            pool=pool,
            selfplay=self._selfplay,
            model_path=model_path,
            n_games_per_pair=n_games_per_pair,
            concurrency=concurrency,
            session_size=session_size,
            anchor_name=anchor_name,
            anchor_elo=anchor_elo,
            lapajne_mc_worlds=lapajne_mc_worlds,
            lapajne_mc_sims=lapajne_mc_sims,
            on_mixed_result=on_mixed_result,
            variant=variant,
        )

    def calibrate_candidate(
        self,
        *,
        pool: LeaguePool,
        model_path: str,
        concurrency: int,
        session_size: int,
        n_games_per_opponent: int,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        variant: int = 0,
    ) -> float | None:
        opponents = list(pool.entries)
        if not opponents:
            return None

        n_games = max(1, n_games_per_opponent)
        n_seats = 3 if int(variant) == 1 else 4
        n_filler = n_seats - 2  # learner + target are seats 0 and 1
        implied_elos: list[float] = []

        for target_idx, target in enumerate(opponents):
            filler_indices = [i for i in range(len(opponents)) if i != target_idx]
            if not filler_indices:
                filler_indices = [target_idx]
            cyc = cycle(filler_indices)
            filler_tokens = [opponents[next(cyc)].opponent.seat_token() for _ in range(n_filler)]

            seat_config = ",".join(
                [model_path, target.opponent.seat_token()] + filler_tokens
            )

            raw = self._selfplay.run(
                model_path=model_path,
                n_games=n_games,
                seat_config=seat_config,
                explore_rate=0.0,
                concurrency=concurrency,
                include_replay_data=False,
                include_oracle_states=False,
                lapajne_mc_worlds=lapajne_mc_worlds,
                lapajne_mc_sims=lapajne_mc_sims,
                variant=variant,
            )
            scores = raw.get("scores")
            places = _avg_placements(scores, session_size=session_size)
            if places is None:
                continue

            learner_place = places[0]
            target_place = places[1]
            implied = target.elo + _ELO_PER_PLACEMENT * (target_place - learner_place)
            implied_elos.append(implied)

        if not implied_elos:
            return None

        implied_elos.sort()
        mid = len(implied_elos) // 2
        if len(implied_elos) % 2 == 0:
            return (implied_elos[mid - 1] + implied_elos[mid]) / 2.0
        return implied_elos[mid]
