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
    "trailing",
    "relative_trailing",
    "league_pool",
    "best_snapshot",
    "weakest_snapshot",
}) | frozenset({
    # Heuristic-bot shadow sources: the shadow seat is played by a
    # hand-coded bot with no NN involved. Kept in sync with
    # ``HEURISTIC_SHADOW_BOT_LABELS`` in
    # ``training.adapters.duplicate.shadow_sources`` and
    # ``SUPPORTED_BOT_SEAT_LABELS`` in ``engine-rs/src/player_bot.rs``.
    "bot_lapajne",
    "bot_lustrek",
    "bot_v1",
    "bot_v3",
    "bot_v5",
    "bot_v6",
    "bot_m6",
    "bot_m8",
    "bot_m9",
    "bot_pozrl",
})

_VALID_REWARD_MODELS: frozenset[str] = frozenset({
    "shadow_score_diff",
    "imps",
    "ranking",
})

_VALID_LEARNER_SEAT_TOKENS: frozenset[str] = frozenset({
    "nn",
    "centaur",
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
    # Performance knob: caps how many distinct ordered opponent triplets
    # the pairing adapter samples per iteration. Fewer triplets → fewer
    # distinct seatings → fewer `run_self_play` invocations in the Rust
    # engine → fewer TorchScript model reloads. Duplicate-invariant
    # semantics are unaffected (the deck seed still drives dealing). Set
    # to None to restore the legacy per-pod independent sampling.
    max_opponent_triplets: int | None = 4
    shadow_source: str = "previous_iteration"
    # Only used when ``shadow_source == "trailing"``: the shadow TorchScript
    # file is refreshed every ``shadow_refresh_interval`` iterations, so
    # the baseline lags the learner by up to N-1 iterations. Must be >= 1.
    # Typical values: 5 (matches snapshot_interval) or 10.
    shadow_refresh_interval: int = 1
    apply_shaped_bonuses: bool = False
    reward_model: str = "shadow_score_diff"
    rng_seed: int = 0
    # Which engine-side player type sits at the learner position in each pod.
    # "nn" (default) keeps the existing pure-NN learner; "centaur" routes the
    # learner through the NN+endgame-solver hybrid so tricks >= handoff_trick
    # are decided by PIMC/alpha-mu. Opponent tokens come from the league.
    learner_seat_token: str = "nn"

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
        if self.shadow_refresh_interval < 1:
            raise ValueError(
                f"duplicate.shadow_refresh_interval must be >= 1, "
                f"got {self.shadow_refresh_interval}"
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
        if self.max_opponent_triplets is not None and self.max_opponent_triplets < 1:
            raise ValueError(
                f"duplicate.max_opponent_triplets must be >= 1 or null, "
                f"got {self.max_opponent_triplets}"
            )
        if self.learner_seat_token not in _VALID_LEARNER_SEAT_TOKENS:
            raise ValueError(
                f"duplicate.learner_seat_token must be one of "
                f"{sorted(_VALID_LEARNER_SEAT_TOKENS)}, got {self.learner_seat_token!r}"
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
