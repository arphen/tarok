"""Use case: update Elo ratings in the league pool after one iteration."""

from __future__ import annotations

from training.entities.league import LeaguePool

_K = 32  # Elo K-factor


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


class UpdateLeagueElo:
    """Pure logic — mutates ``pool.entries`` in place.

    ``seat_outcomes`` maps each opponent seat index to a
    ``(learner_wins, opponent_wins, draws)`` triple computed per-game.
    This replaces the old mean-score comparison which was biased by
    multi-NN seat averaging.

    For each of seats 1–3 that maps to a named pool entry, we do a pairwise
    Elo update using the aggregated per-game win rate.  If the same opponent
    occupies multiple seats it is updated independently per seat.
    """

    def execute(
        self,
        pool: LeaguePool,
        seat_config_used: str,
        seat_outcomes: dict[int, tuple[int, int, int]],
    ) -> None:
        labels = [s.strip() for s in seat_config_used.split(",")]
        if len(labels) != 4:
            return

        # Build a token→entry map for fast lookup
        token_to_entry: dict[str, list] = {}
        for entry in pool.entries:
            tok = entry.opponent.seat_token()
            token_to_entry.setdefault(tok, []).append(entry)

        for seat_idx in range(1, 4):
            token = labels[seat_idx]
            if token == "nn":
                continue  # learner vs learner — skip
            entries_for_token = token_to_entry.get(token)
            if not entries_for_token:
                continue
            outcomes = seat_outcomes.get(seat_idx)
            if outcomes is None:
                continue

            learner_wins, opp_wins, draws = outcomes
            n_games = learner_wins + opp_wins + draws
            if n_games == 0:
                continue

            # Aggregate outcome as a fraction: 1.0 = opponent won all,
            # 0.0 = learner won all, draws count as 0.5 each.
            opp_outcome = (opp_wins + 0.5 * draws) / n_games

            for entry in entries_for_token:
                entry.games_played += n_games
                entry.wins += opp_wins

                opp_elo = entry.elo
                learner_elo_ref = 1500.0
                e_opp = 1.0 - _elo_expected(learner_elo_ref, opp_elo)
                entry.elo += _K * (opp_outcome - e_opp)
