"""Use case: update Elo ratings in the league pool after one iteration."""

from __future__ import annotations

from training.entities.league import LeaguePool

# ── Elo K-factor ───────────────────────────────────────────────────────
# Standard chess K=32 updates a rating by at most 32 points per decisive
# outcome. We deliberately scale K by ``pool.config.elo_outplace_unit_weight``
# so that one *session* aggregating N greedy games carries the same rating
# impact as N individual games would have. With the default value of 50
# (aligned to ``outplace_session_size``) the effective K becomes 1600, which
# means a single lopsided iteration can shift learner Elo by several hundred
# points — correct in expectation over many iterations, but intentionally
# aggressive and thus "high variance" early in training.
#
# Implications for tuning:
#   * ``elo_outplace_unit_weight = outplace_session_size`` (current default)
#     treats a session as a single high-signal comparison unit. Use this
#     when your schedules are Elo-driven and you want swift convergence.
#   * ``elo_outplace_unit_weight = 1`` falls back to per-session K=32, which
#     decays a lot more slowly and is less noisy — preferable if Elo is a
#     monitoring signal rather than a scheduling signal.
#   * Per-pairwise tapering (K / sqrt(games_seen)) is not implemented here;
#     if you want it, replace ``k_factor`` inside ``execute`` with a decayed
#     variant based on ``entry.games_played``.
_K = 32  # Elo K-factor (scaled below by elo_outplace_unit_weight)


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
        opponent_outcomes: dict[str, tuple[int, int, int]] | None = None,
    ) -> None:
        for entry in pool.entries:
            entry.recent_outplace_rate = None
            entry.recent_outplace_samples = 0

        # Build a token→entry map for fast lookup (used by both paths).
        token_to_entry: dict[str, list] = {}
        for entry in pool.entries:
            tok = entry.opponent.seat_token()
            token_to_entry.setdefault(tok, []).append(entry)

        # Session-based outplacing produces fewer but higher-signal outcomes;
        # this weight scales K so one unit can carry more Elo impact.
        k_weight = max(1.0, float(pool.config.elo_outplace_unit_weight))
        k_factor = _K * k_weight

        # Duplicate-mode path: outcomes are already bucketed by opponent
        # league token (learner rotates through all 4 seats pod-by-pod, so
        # seat-index-keyed ``seat_outcomes`` cannot represent them). When
        # this dict is non-empty we prefer it and skip the seat-indexed
        # fallback so the same comparison is never counted twice.
        if opponent_outcomes:
            for token, outcomes in opponent_outcomes.items():
                entries_for_token = token_to_entry.get(token)
                if not entries_for_token:
                    continue
                self._apply_outcomes(
                    pool=pool,
                    entries_for_token=entries_for_token,
                    outcomes=outcomes,
                    k_factor=k_factor,
                )
            return

        labels = [s.strip() for s in seat_config_used.split(",")]
        if len(labels) != 4:
            return

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
            self._apply_outcomes(
                pool=pool,
                entries_for_token=entries_for_token,
                outcomes=outcomes,
                k_factor=k_factor,
            )

    @staticmethod
    def _apply_outcomes(
        *,
        pool: LeaguePool,
        entries_for_token: list,
        outcomes: tuple[int, int, int],
        k_factor: float,
    ) -> None:
        learner_outplaces, opp_outplaces, draws = outcomes
        n_games = learner_outplaces + opp_outplaces + draws
        if n_games == 0:
            return

        recent_outplace_rate = learner_outplaces / n_games

        # Aggregate outcome as a fraction: 1.0 = opponent won all,
        # 0.0 = learner won all, draws count as 0.5 each.
        opp_outcome = (opp_outplaces + 0.5 * draws) / n_games
        learner_outcome = 1.0 - opp_outcome

        for entry in entries_for_token:
            entry.games_played += n_games
            entry.learner_outplaces += learner_outplaces
            entry.recent_outplace_rate = recent_outplace_rate
            entry.recent_outplace_samples = n_games

            opp_elo = entry.elo
            e_learner = _elo_expected(pool.learner_elo, opp_elo)
            pool.learner_elo += k_factor * (learner_outcome - e_learner)
