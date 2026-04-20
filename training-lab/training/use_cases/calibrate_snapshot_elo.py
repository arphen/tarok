"""Use case: calibrate newly admitted snapshot Elo from greedy matchup results."""

from __future__ import annotations

from itertools import cycle
from typing import Callable

from training.entities.league import LeaguePool
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


class CalibrateSnapshotElo:
    def execute(
        self,
        pool: LeaguePool,
        snapshot_name: str,
        selfplay: SelfPlayPort,
        model_path: str,
        n_games_per_opponent: int,
        concurrency: int,
        session_size: int,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        on_pair_result: Callable[[str, str, int, int, int, float, float, float], None] | None = None,
        on_match_setup: Callable[[str, str, int], None] | None = None,
    ) -> bool:
        if n_games_per_opponent <= 0:
            return False

        snapshot = next((e for e in pool.entries if e.opponent.name == snapshot_name), None)
        if snapshot is None:
            return False

        opponents = [e for e in pool.entries if e.opponent.name != snapshot_name]
        if not opponents:
            return False

        snap_tok = snapshot.opponent.seat_token()
        implied_elos: list[float] = []

        for target_idx, target in enumerate(opponents):
            # Fill remaining 2 seats with other opponents (cycling).
            filler_indices = [i for i in range(len(opponents)) if i != target_idx]
            if not filler_indices:
                filler_indices = [target_idx]
            cyc = cycle(filler_indices)
            f1, f2 = opponents[next(cyc)], opponents[next(cyc)]

            seat_config = (
                f"{snap_tok},"
                f"{target.opponent.seat_token()},"
                f"{f1.opponent.seat_token()},"
                f"{f2.opponent.seat_token()}"
            )

            if on_match_setup is not None:
                on_match_setup(snapshot_name, seat_config, n_games_per_opponent)

            raw = selfplay.run(
                model_path=model_path,
                n_games=n_games_per_opponent,
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

            places = _avg_placements(scores, session_size=session_size)
            if places is None:
                continue

            snapshot_place = places[0]
            target_place = places[1]
            implied = target.elo + _ELO_PER_PLACEMENT * (target_place - snapshot_place)
            implied_elos.append(implied)

            if on_pair_result is not None:
                on_pair_result(
                    snapshot_name,
                    target.opponent.name,
                    0,
                    0,
                    0,
                    snapshot_place,
                    implied,
                    target.elo,
                )

        if not implied_elos:
            return False

        # Median — robust to outlier opponents with inflated base Elo.
        implied_elos.sort()
        mid = len(implied_elos) // 2
        if len(implied_elos) % 2 == 0:
            snapshot.elo = (implied_elos[mid - 1] + implied_elos[mid]) / 2
        else:
            snapshot.elo = implied_elos[mid]
        return True
