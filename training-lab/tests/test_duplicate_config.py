"""Tests for DuplicateConfig dataclass."""

from __future__ import annotations

import pytest

from training.entities.duplicate_config import DuplicateConfig


def test_default_is_disabled() -> None:
    cfg = DuplicateConfig()
    assert cfg.enabled is False
    assert cfg.actor_only is False
    assert cfg.pairing == "rotation_8game"
    assert cfg.games_per_pod == 8
    assert cfg.active_games_per_pod == 4


def test_actor_only_requires_enabled() -> None:
    with pytest.raises(ValueError, match="actor_only"):
        DuplicateConfig(enabled=False, actor_only=True)


def test_actor_only_with_enabled_is_allowed() -> None:
    cfg = DuplicateConfig(enabled=True, actor_only=True)
    assert cfg.actor_only is True


def test_invalid_pairing_rejected() -> None:
    with pytest.raises(ValueError, match="pairing"):
        DuplicateConfig(pairing="rotation_16game")


def test_invalid_reward_model_rejected() -> None:
    with pytest.raises(ValueError, match="reward_model"):
        DuplicateConfig(reward_model="matchpoints")


def test_invalid_shadow_source_rejected() -> None:
    with pytest.raises(ValueError, match="shadow_source"):
        DuplicateConfig(shadow_source="best_ghost_ever")


def test_valid_shadow_sources_accepted() -> None:
    for src in (
        "previous_iteration",
        "trailing",
        "relative_trailing",
        "league_pool",
        "best_snapshot",
        "weakest_snapshot",
    ):
        cfg = DuplicateConfig(shadow_source=src)
        assert cfg.shadow_source == src


def test_heuristic_bot_shadow_sources_accepted() -> None:
    # Heuristic-bot seat labels are valid shadow_source values. They route
    # through ``HeuristicBotShadowSource`` in the factory and make the
    # engine play a hand-coded bot at the shadow seat (no NN).
    for src in (
        "bot_v3",
        "bot_v5",
        "bot_v6",
        "bot_m6",
        "bot_lustrek",
        "bot_lapajne",
    ):
        cfg = DuplicateConfig(shadow_source=src)
        assert cfg.shadow_source == src


def test_invalid_shadow_refresh_interval_rejected() -> None:
    with pytest.raises(ValueError, match="shadow_refresh_interval"):
        DuplicateConfig(shadow_refresh_interval=0)


def test_negative_pods_per_iteration_rejected() -> None:
    with pytest.raises(ValueError, match="pods_per_iteration"):
        DuplicateConfig(pods_per_iteration=-1)


def test_games_per_pod_by_pairing() -> None:
    assert DuplicateConfig(pairing="rotation_8game").games_per_pod == 8
    assert DuplicateConfig(pairing="rotation_4game").games_per_pod == 4
    assert DuplicateConfig(pairing="single_seat_2game").games_per_pod == 2


def test_learner_seat_token_defaults_to_nn() -> None:
    assert DuplicateConfig().learner_seat_token == "nn"


def test_learner_seat_token_accepts_centaur() -> None:
    cfg = DuplicateConfig(learner_seat_token="centaur")
    assert cfg.learner_seat_token == "centaur"


def test_invalid_learner_seat_token_rejected() -> None:
    with pytest.raises(ValueError, match="learner_seat_token"):
        DuplicateConfig(learner_seat_token="bot_v5")
