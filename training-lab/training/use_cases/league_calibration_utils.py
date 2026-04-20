"""Shared helpers for league Elo calibration use cases."""

from __future__ import annotations

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
