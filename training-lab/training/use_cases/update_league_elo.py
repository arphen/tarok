"""Use case: update Elo ratings in the league pool after one iteration."""

from __future__ import annotations

from training.entities.league import LeaguePool

_K = 32  # Elo K-factor


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


class UpdateLeagueElo:
    """Pure logic — mutates ``pool.entries`` in place.

    ``seat_outcomes`` maps each opponent seat index to a
    ``(learner_outplaces, opponent_outplaces, draws)`` triple computed
    from comparison units (single games or aggregated sessions):

    * learner_outplaces: ``learner_score > opponent_score``
    * opponent_outplaces: ``learner_score < opponent_score``
    * draws: ``learner_score == opponent_score``

    This is intentionally pairwise per comparison unit. It replaces the older
    mean-score comparison, which could be biased by multi-NN seat averaging.
    The Elo update uses aggregate outcome over those pairwise outcomes.

    For each of seats 1–3 that maps to a named pool entry, we compute expected
    score against that fixed yardstick and update only the learner Elo.
    Opponent/snapshot Elo values remain fixed.
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

        # Session-based outplacing produces fewer but higher-signal outcomes;
        # this weight scales K so one unit can carry more Elo impact.
        k_weight = max(1.0, float(pool.config.elo_outplace_unit_weight))
        k_factor = _K * k_weight

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

            learner_outplaces, opp_outplaces, draws = outcomes
            n_games = learner_outplaces + opp_outplaces + draws
            if n_games == 0:
                continue

            # Aggregate outcome as a fraction: 1.0 = opponent won all,
            # 0.0 = learner won all, draws count as 0.5 each.
            opp_outcome = (opp_outplaces + 0.5 * draws) / n_games

            learner_outcome = 1.0 - opp_outcome

            for entry in entries_for_token:
                entry.games_played += n_games
                entry.learner_outplaces += learner_outplaces

                opp_elo = entry.elo
                e_learner = _elo_expected(pool.learner_elo, opp_elo)
                pool.learner_elo += k_factor * (learner_outcome - e_learner)
