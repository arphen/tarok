"""Entity: DuplicateConfig — the ``duplicate:`` YAML block, parsed.

Added to :class:`training.entities.training_config.TrainingConfig` as an
optional ``duplicate`` field. When ``enabled=False`` (the default),
everything about the training pipeline behaves exactly as it did before
duplicate RL was introduced — see ``test_duplicate_disabled_is_noop``.
"""

from __future__ import annotations

from dataclasses import dataclass


_VALID_PAIRINGS: frozenset[str] = frozenset({
    "rotation_8game",
    "rotation_4game",
    "single_seat_2game",
})

_VALID_SHADOW_SOURCES: frozenset[str] = frozenset({
    "previous_iteration",
    "league_pool_role",
})

_VALID_REWARD_MODELS: frozenset[str] = frozenset({
    "shadow_score_diff",
    "imps",
    "ranking",
})


@dataclass(frozen=True)
class DuplicateConfig:
    """Parsed ``duplicate:`` block from the YAML config.

    All fields have defaults that make ``DuplicateConfig()`` equivalent to
    the feature being disabled. ``enabled=True, actor_only=False`` enables
    the conservative mode (reward source only, GAE retained). Setting
    ``actor_only=True`` additionally drops the critic/oracle/GAE path — see
    ``docs/double_rl.md`` §2.7.
    """

    enabled: bool = False
    actor_only: bool = False
    pairing: str = "rotation_8game"
    pods_per_iteration: int = 400
    shadow_source: str = "previous_iteration"
    apply_shaped_bonuses: bool = False
    reward_model: str = "shadow_score_diff"
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.actor_only and not self.enabled:
            raise ValueError(
                "duplicate.actor_only=true requires duplicate.enabled=true"
            )
        if self.pairing not in _VALID_PAIRINGS:
            raise ValueError(
                f"duplicate.pairing must be one of {sorted(_VALID_PAIRINGS)}, "
                f"got {self.pairing!r}"
            )
        if self.shadow_source not in _VALID_SHADOW_SOURCES:
            raise ValueError(
                f"duplicate.shadow_source must be one of {sorted(_VALID_SHADOW_SOURCES)}, "
                f"got {self.shadow_source!r}"
            )
        if self.reward_model not in _VALID_REWARD_MODELS:
            raise ValueError(
                f"duplicate.reward_model must be one of {sorted(_VALID_REWARD_MODELS)}, "
                f"got {self.reward_model!r}"
            )
        if self.pods_per_iteration < 0:
            raise ValueError(
                f"duplicate.pods_per_iteration must be >= 0, got {self.pods_per_iteration}"
            )

    @property
    def games_per_pod(self) -> int:
        """Total games (active + shadow) produced by one pod under the chosen pairing."""
        if self.pairing == "rotation_8game":
            return 8
        if self.pairing == "rotation_4game":
            return 4
        # single_seat_2game
        return 2

    @property
    def active_games_per_pod(self) -> int:
        """Learner-seat (active-group) games per pod."""
        return self.games_per_pod // 2
