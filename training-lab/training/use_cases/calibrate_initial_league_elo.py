"""Use case: bootstrap opponent Elo from greedy bot-vs-bot calibration games."""

from __future__ import annotations

from itertools import cycle
from typing import Callable

from training.entities.league import LeagueOpponent, LeaguePool, LeaguePoolEntry
from training.ports.selfplay_port import SelfPlayPort


_ELO_PER_PLACEMENT = 250.0


def _placement_from_scores(score_vec: list[float]) -> tuple[float, float, float, float]:
    out = [1.0, 1.0, 1.0, 1.0]
    for i in range(4):
        out[i] = 1.0 + float(sum(1 for v in score_vec if v > score_vec[i]))
    return out[0], out[1], out[2], out[3]


def _avg_placements(scores: list, session_size: int) -> tuple[float, float, float, float] | None:
    if scores is None or len(scores) == 0:
        return None

    units: list[list[float]] = []
    n_games = len(scores)
    if session_size > 1 and n_games >= session_size:
        n_sessions = n_games // session_size
        used_games = n_sessions * session_size
        for start in range(0, used_games, session_size):
            total = [0.0, 0.0, 0.0, 0.0]
            for g in range(start, start + session_size):
                row = scores[g]
                for s in range(4):
                    total[s] += float(row[s])
            units.append(total)
    else:
        units = [[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in scores]

    if not units:
        return None

    sums = [0.0, 0.0, 0.0, 0.0]
    for u in units:
        p0, p1, p2, p3 = _placement_from_scores(u)
        sums[0] += p0
        sums[1] += p1
        sums[2] += p2
        sums[3] += p3
    n = float(len(units))
    return sums[0] / n, sums[1] / n, sums[2] / n, sums[3] / n


def _pick_three_opponents(tokens: list[str]) -> tuple[str, str, str]:
    if not tokens:
        return "nn", "nn", "nn"
    cyc = cycle(tokens)
    return next(cyc), next(cyc), next(cyc)


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
    ) -> bool:
        entries = [
            e for e in pool.entries
            if e.opponent.type != "nn_checkpoint" or include_checkpoints
        ]
        
        # Create temporary learner entry for calibration
        learner_opponent = LeagueOpponent(name="learner", type="nn_checkpoint", path=model_path)
        learner_entry = LeaguePoolEntry(opponent=learner_opponent, elo=1500.0)
        entries.append(learner_entry)
        
        if not entries:
            return False

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

        place_by_name: dict[str, float] = {}
        total_runs = len(entries)
        for run_idx, entry in enumerate(entries, start=1):
            target_name = entry.opponent.name
            other_tokens = [
                e.opponent.seat_token()
                for e in entries
                if e.opponent.name != target_name
            ]
            seat1, seat2, seat3 = _pick_three_opponents(other_tokens)
            seat_config = f"{entry.opponent.seat_token()},{seat1},{seat2},{seat3}"

            if on_pair_start is not None:
                on_pair_start(run_idx, total_runs, target_name, f"{seat1},{seat2},{seat3}")

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
            )
            scores = raw.get("scores")
            if scores is None or len(scores) == 0:
                continue

            placements = _avg_placements(scores, session_size=session_size)
            if placements is None:
                continue

            place_by_name[target_name] = placements[0]
            if on_mixed_result is not None:
                on_mixed_result(run_idx, total_runs, target_name, (seat1, seat2, seat3), placements)
            if on_pair_done is not None:
                on_pair_done(run_idx, total_runs, target_name, f"{seat1},{seat2},{seat3}", 0, 0, 0, 0.0)

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
