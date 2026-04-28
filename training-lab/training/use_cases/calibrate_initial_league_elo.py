"""Use case: bootstrap opponent Elo from greedy bot-vs-bot calibration games."""

from __future__ import annotations

from itertools import cycle
from typing import Callable

from training.entities.league import LeagueOpponent, LeaguePool, LeaguePoolEntry
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.league_calibration_utils import (
    _ELO_PER_PLACEMENT,
    _avg_placements,
    _placement_from_scores,
)


def _pick_n_opponents(tokens: list[str], n: int) -> tuple[str, ...]:
    if n <= 0:
        return ()
    if not tokens:
        return tuple(["nn"] * n)
    cyc = cycle(tokens)
    return tuple(next(cyc) for _ in range(n))


def _pick_three_opponents(tokens: list[str]) -> tuple[str, str, str]:
    picked = _pick_n_opponents(tokens, 3)
    return picked[0], picked[1], picked[2]


class CalibrateInitialLeagueElo:
    """Calibrate baseline opponent Elo before learner-vs-league begins."""

    def execute(
        self,
        pool: LeaguePool,
        selfplay: SelfPlayPort,
        model_path: str,
        n_games_per_pair: int,
        concurrency: int,
        session_size: int,
        anchor_name: str | None = None,
        anchor_elo: float = 1500.0,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        on_pair_start: Callable[[int, int, str, str], None] | None = None,
        on_pair_done: Callable[[int, int, str, str, int, int, int, float], None] | None = None,
        on_mixed_result: Callable[[int, int, str, tuple[str, str, str], tuple[float, float, float, float]], None] | None = None,
        include_checkpoints: bool = False,
        variant: int = 0,
    ) -> bool:
        entries = [
            e for e in pool.entries
            if e.opponent.type != "nn_checkpoint" or include_checkpoints
        ]

        if not entries:
            pool.learner_elo = float(anchor_elo)
            return True

        # Create temporary learner entry for calibration
        learner_opponent = LeagueOpponent(name="learner", type="nn_checkpoint", path=model_path)
        learner_entry = LeaguePoolEntry(opponent=learner_opponent, elo=1500.0)
        entries.append(learner_entry)

        if len(entries) == 1:
            entries[0].elo = float(anchor_elo)
            return True

        if n_games_per_pair <= 0:
            return False

        anchor_idx = 0
        if anchor_name is not None:
            for i, entry in enumerate(entries):
                if entry.opponent.name == anchor_name:
                    anchor_idx = i
                    break

        # Number of total seats: 4 for FourPlayer (variant=0), 3 for ThreePlayer (variant=1).
        n_seats = 3 if int(variant) == 1 else 4
        n_filler = n_seats - 1

        place_by_name: dict[str, float] = {}
        total_runs = len(entries)
        for run_idx, entry in enumerate(entries, start=1):
            target_name = entry.opponent.name
            other_tokens = [
                e.opponent.seat_token()
                for e in entries
                if e.opponent.name != target_name
            ]
            filler = _pick_n_opponents(other_tokens, n_filler)
            seat_config = ",".join((entry.opponent.seat_token(),) + filler)
            filler_label = ",".join(filler)

            if on_pair_start is not None:
                on_pair_start(run_idx, total_runs, target_name, filler_label)

            raw = selfplay.run(
                model_path=model_path,
                n_games=n_games_per_pair,
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
            if scores is None or len(scores) == 0:
                continue

            placements = _avg_placements(scores, session_size=session_size)
            if placements is None:
                continue

            place_by_name[target_name] = placements[0]
            if on_mixed_result is not None:
                # Pad filler to length 3 for the legacy callback signature.
                padded = (filler + ("",) * 3)[:3]
                placements_padded = (placements + (0.0,) * 4)[:4]
                on_mixed_result(run_idx, total_runs, target_name, padded, placements_padded)
            if on_pair_done is not None:
                on_pair_done(run_idx, total_runs, target_name, filler_label, 0, 0, 0, 0.0)

        anchor_entry = entries[anchor_idx]
        anchor_place = place_by_name.get(anchor_entry.opponent.name)
        if anchor_place is None:
            for entry in entries:
                entry.elo = float(anchor_elo)
            learner_elo = float(anchor_elo) - 100.0
            pool.learner_elo = learner_elo
            return True

        for entry in entries:
            p = place_by_name.get(entry.opponent.name)
            if p is None:
                entry.elo = float(anchor_elo)
                continue
            entry.elo = float(anchor_elo) + _ELO_PER_PLACEMENT * (anchor_place - p)
        
        # Extract learner Elo and set it 100 points below
        learner_elo_raw = place_by_name.get("learner")
        if learner_elo_raw is not None:
            learner_elo = float(anchor_elo) + _ELO_PER_PLACEMENT * (anchor_place - learner_elo_raw)
            pool.learner_elo = learner_elo - 100.0
        
        return True
