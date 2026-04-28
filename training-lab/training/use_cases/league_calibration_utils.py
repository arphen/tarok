"""Shared helpers for league Elo calibration use cases."""

from __future__ import annotations

_ELO_PER_PLACEMENT = 250.0


def _placement_from_scores(score_vec: list[float]) -> tuple[float, ...]:
    n = len(score_vec)
    out = [1.0] * n
    for i in range(n):
        out[i] = 1.0 + float(sum(1 for v in score_vec if v > score_vec[i]))
    return tuple(out)


def _avg_placements(scores: list, session_size: int) -> tuple[float, ...] | None:
    if scores is None or len(scores) == 0:
        return None

    # Detect seat count from first row; supports 3 or 4 seats.
    first = scores[0]
    n_seats = len(first)
    if n_seats <= 0:
        return None

    units: list[list[float]] = []
    n_games = len(scores)
    if session_size > 1 and n_games >= session_size:
        n_sessions = n_games // session_size
        used_games = n_sessions * session_size
        for start in range(0, used_games, session_size):
            total = [0.0] * n_seats
            for g in range(start, start + session_size):
                row = scores[g]
                for s in range(n_seats):
                    total[s] += float(row[s])
            units.append(total)
    else:
        units = [[float(r[s]) for s in range(n_seats)] for r in scores]

    if not units:
        return None

    sums = [0.0] * n_seats
    for u in units:
        placements = _placement_from_scores(u)
        for s in range(n_seats):
            sums[s] += placements[s]
    n = float(len(units))
    return tuple(s / n for s in sums)
