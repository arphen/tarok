"""Entity: per-iteration duplicate-RL statistics.

Captures, for one training iteration's pod batch:

* Per-opponent outplace outcomes (learner_score vs. each opponent seat,
  bucketed by opponent label) — the duplicate analogue of
  ``seat_outcomes`` in regular self-play; consumed by
  :class:`training.use_cases.update_league_elo.UpdateLeagueElo` to update
  the league Elo ratings for tokens the learner actually met this iter.
* Mean duplicate advantage (learner_score − shadow_score per matched
  deck, averaged over all active games) and its sample std. This is the
  strongest "am I improving?" signal available in duplicate mode — deal
  variance is cancelled, so the only source of iteration-to-iteration
  noise is the learner vs. shadow policy gap.

The entity is a pure value object: no numpy / torch imports, no adapter
types, safe to reference from ``training.use_cases``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DuplicateIterationStats:
    """Per-iteration duplicate stats.

    Attributes
    ----------
    opponent_outcomes
        ``{opponent_token: (learner_outplaces, opponent_outplaces, draws)}``
        aggregated across every active game and every opponent seat the
        learner faced. ``opponent_token`` is the league-pool seat token
        (e.g. ``"bot_v5"``, ``"Lustrek"``, or a checkpoint path).
    opponent_games
        ``{opponent_token: n_comparisons}`` — the number of learner-vs-
        opponent pairwise comparisons summed into ``opponent_outcomes``.
        Equal to ``learner_outplaces + opponent_outplaces + draws`` for
        each token; exposed separately so presenters can render "games
        vs X" without re-summing.
    mean_advantage
        Mean of ``(learner_score − shadow_score) / 100`` across matched
        active/shadow game pairs (same deck, same learner seat). The /100
        scale matches the PPO reward so presenter output and the training
        gradient speak the same units.
    advantage_std
        Sample standard deviation of the same per-game advantage, on the
        /100 scale. ``0.0`` when ``n_active_games < 2``.
    n_active_games
        Count of active games that contributed to ``mean_advantage``.
    """

    opponent_outcomes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    opponent_games: dict[str, int] = field(default_factory=dict)
    mean_advantage: float = 0.0
    advantage_std: float = 0.0
    n_active_games: int = 0
