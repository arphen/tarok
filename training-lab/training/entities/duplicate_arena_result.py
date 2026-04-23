"""Entity: DuplicateArenaResult — outcome of a head-to-head duplicate match.

Produced by :class:`training.use_cases.run_duplicate_arena.RunDuplicateArena`
and consumed by the arena CLI / HTTP layer for display.

Numbers are stored as plain floats so this entity stays free of numerical
library imports under the import-linter ``entities`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DuplicateArenaResult:
    """Per-match summary statistics for a duplicate arena run.

    Attributes
    ----------
    boards_played
        Number of paired active/shadow games (one per (pod, rotation)).
    challenger_mean_score
        Mean raw game score of the challenger (learner seat) across all
        boards.
    defender_mean_score
        Mean raw game score of the defender (shadow seat) across the same
        boards.
    mean_duplicate_advantage
        Mean of ``challenger_score - defender_score`` across boards.
        Positive → challenger is stronger on this sample.
    duplicate_advantage_std
        Sample standard deviation of the per-board advantage.
    ci_low_95
        Lower bound of the 95% bootstrap confidence interval on
        ``mean_duplicate_advantage``.
    ci_high_95
        Upper bound of the 95% bootstrap confidence interval.
    imps_per_board
        International-match-point-style scaled advantage. For the default
        ``ShadowScoreRewardAdapter`` accounting this is simply
        ``mean_duplicate_advantage / score_scale`` (score_scale defaults to
        100); alternative reward models may redefine it.
    """

    boards_played: int
    challenger_mean_score: float
    defender_mean_score: float
    mean_duplicate_advantage: float
    duplicate_advantage_std: float
    ci_low_95: float
    ci_high_95: float
    imps_per_board: float
