"""Tests for ShadowScoreRewardAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from training.adapters.duplicate.shadow_score_reward import ShadowScoreRewardAdapter
from training.entities.duplicate_pod import DuplicatePod


def _make_pod(seed: int = 0) -> DuplicatePod:
    return DuplicatePod(
        deck_seed=seed,
        opponents=("bot_v5", "bot_v6", "bot_m6"),
        active_seatings=(("nn", "bot_v5", "bot_v6", "bot_m6"),),
        shadow_seatings=(("shadow", "bot_v5", "bot_v6", "bot_m6"),),
        learner_positions=(0,),
    )


def test_terminal_reward_equals_score_diff_over_100() -> None:
    # Single active game, trajectory length 3 (3 steps for player 0 in game 0).
    # Learner final score = 70; shadow final score = 50 → reward = 0.20.
    active_raw = {
        "game_ids": np.array([0, 0, 0], dtype=np.uint32),
        "players": np.array([0, 0, 0], dtype=np.uint8),
        "scores": np.array([[70, -30, -30, -30]], dtype=np.int32),
        "pod_ids": np.array([0, 0, 0], dtype=np.int64),
        "learner_positions_flat": np.array([0, 0, 0], dtype=np.int64),
        "game_idx_within_pod": np.array([0, 0, 0], dtype=np.int64),
    }
    shadow_scores = np.array([[[50, -20, -20, -20]]], dtype=np.int32)  # (1 pod, 1 game, 4 seats)
    adapter = ShadowScoreRewardAdapter()
    rewards = adapter.compute_rewards(active_raw, shadow_scores, [_make_pod()])
    assert rewards.shape == (3,)
    # Non-terminal zeros.
    assert rewards[0] == 0.0
    assert rewards[1] == 0.0
    # Terminal = (70 - 50) / 100 = 0.20.
    np.testing.assert_allclose(rewards[2], 0.20, atol=1e-6)


def test_negative_score_diff_is_negative_reward() -> None:
    active_raw = {
        "game_ids": np.array([0, 0], dtype=np.uint32),
        "players": np.array([0, 0], dtype=np.uint8),
        "scores": np.array([[-40, 0, 0, 0]], dtype=np.int32),
        "pod_ids": np.array([0, 0], dtype=np.int64),
        "learner_positions_flat": np.array([0, 0], dtype=np.int64),
        "game_idx_within_pod": np.array([0, 0], dtype=np.int64),
    }
    shadow_scores = np.array([[[30, 0, 0, 0]]], dtype=np.int32)
    adapter = ShadowScoreRewardAdapter()
    rewards = adapter.compute_rewards(active_raw, shadow_scores, [_make_pod()])
    np.testing.assert_allclose(rewards[-1], -1.40, atol=1e-6)


def test_custom_score_scale() -> None:
    active_raw = {
        "game_ids": np.array([0], dtype=np.uint32),
        "players": np.array([0], dtype=np.uint8),
        "scores": np.array([[50, 0, 0, 0]], dtype=np.int32),
        "pod_ids": np.array([0], dtype=np.int64),
        "learner_positions_flat": np.array([0], dtype=np.int64),
        "game_idx_within_pod": np.array([0], dtype=np.int64),
    }
    shadow_scores = np.array([[[10, 0, 0, 0]]], dtype=np.int32)
    adapter = ShadowScoreRewardAdapter(score_scale=40.0)
    rewards = adapter.compute_rewards(active_raw, shadow_scores, [_make_pod()])
    np.testing.assert_allclose(rewards[-1], 1.0, atol=1e-6)


def test_empty_input_returns_empty_array() -> None:
    active_raw = {
        "game_ids": np.array([], dtype=np.uint32),
        "players": np.array([], dtype=np.uint8),
        "scores": np.array([[0, 0, 0, 0]], dtype=np.int32),
        "pod_ids": np.array([], dtype=np.int64),
        "learner_positions_flat": np.array([], dtype=np.int64),
        "game_idx_within_pod": np.array([], dtype=np.int64),
    }
    shadow_scores = np.array([[[0, 0, 0, 0]]], dtype=np.int32)
    adapter = ShadowScoreRewardAdapter()
    rewards = adapter.compute_rewards(active_raw, shadow_scores, [_make_pod()])
    assert rewards.shape == (0,)


def test_multi_game_multi_player() -> None:
    # Two games, each with trajectory for one learner seat (but at different seats).
    # Game 0: learner at seat 0, game 1: learner at seat 2 (rotation).
    active_raw = {
        "game_ids": np.array([0, 0, 1, 1], dtype=np.uint32),
        "players": np.array([0, 0, 2, 2], dtype=np.uint8),
        "scores": np.array(
            [
                [100, 0, 0, 0],   # game 0
                [0, 0, 80, 0],    # game 1
            ],
            dtype=np.int32,
        ),
        "pod_ids": np.array([0, 0, 0, 0], dtype=np.int64),
        "learner_positions_flat": np.array([0, 0, 2, 2], dtype=np.int64),
        "game_idx_within_pod": np.array([0, 0, 1, 1], dtype=np.int64),
    }
    # Shadow: (1 pod, 2 games_within_pod, 4 seats).
    shadow_scores = np.array(
        [
            [
                [50, 0, 0, 0],   # shadow game 0: shadow at seat 0, score 50
                [0, 0, 60, 0],   # shadow game 1: shadow at seat 2, score 60
            ],
        ],
        dtype=np.int32,
    )
    adapter = ShadowScoreRewardAdapter()
    rewards = adapter.compute_rewards(active_raw, shadow_scores, [_make_pod()])
    # Terminals are indices 1 and 3.
    np.testing.assert_allclose(rewards[0], 0.0)
    np.testing.assert_allclose(rewards[1], (100 - 50) / 100.0, atol=1e-6)
    np.testing.assert_allclose(rewards[2], 0.0)
    np.testing.assert_allclose(rewards[3], (80 - 60) / 100.0, atol=1e-6)


def test_invalid_shadow_scores_shape_rejected() -> None:
    active_raw = {
        "game_ids": np.array([0], dtype=np.uint32),
        "players": np.array([0], dtype=np.uint8),
        "scores": np.array([[0, 0, 0, 0]], dtype=np.int32),
        "pod_ids": np.array([0], dtype=np.int64),
        "learner_positions_flat": np.array([0], dtype=np.int64),
        "game_idx_within_pod": np.array([0], dtype=np.int64),
    }
    bad_shadow = np.zeros((1, 1), dtype=np.int32)  # 2-D instead of 3-D
    adapter = ShadowScoreRewardAdapter()
    with pytest.raises(ValueError, match="shadow_scores"):
        adapter.compute_rewards(active_raw, bad_shadow, [_make_pod()])


def test_invalid_score_scale_rejected() -> None:
    with pytest.raises(ValueError, match="score_scale"):
        ShadowScoreRewardAdapter(score_scale=0.0)


def test_negative_reward_multiplier_is_configurable() -> None:
    active_raw = {
        "game_ids": np.array([0], dtype=np.uint32),
        "players": np.array([0], dtype=np.uint8),
        "scores": np.array([[-40, 0, 0, 0]], dtype=np.int32),
        "pod_ids": np.array([0], dtype=np.int64),
        "learner_positions_flat": np.array([0], dtype=np.int64),
        "game_idx_within_pod": np.array([0], dtype=np.int64),
    }
    shadow_scores = np.array([[[30, 0, 0, 0]]], dtype=np.int32)

    # Baseline multiplier=1.0 preserves the original score difference.
    adapter_linear = ShadowScoreRewardAdapter(negative_reward_multiplier=1.0)
    rewards_linear = adapter_linear.compute_rewards(active_raw, shadow_scores, [_make_pod()])
    np.testing.assert_allclose(rewards_linear[-1], -0.70, atol=1e-6)

    # Stronger multiplier applies only to negative rewards.
    adapter_scaled = ShadowScoreRewardAdapter(negative_reward_multiplier=3.0)
    rewards_scaled = adapter_scaled.compute_rewards(active_raw, shadow_scores, [_make_pod()])
    np.testing.assert_allclose(rewards_scaled[-1], -2.10, atol=1e-6)


def test_invalid_negative_reward_multiplier_rejected() -> None:
    with pytest.raises(ValueError, match="negative_reward_multiplier"):
        ShadowScoreRewardAdapter(negative_reward_multiplier=0.0)


def test_berac_bid_penalty_applies_at_bid_step() -> None:
    # One trajectory (game 0, seat 0): BID then CARD_PLAY.
    # Base score diff is zero; only the explicit Berač bid penalty should remain.
    active_raw = {
        "game_ids": np.array([0, 0], dtype=np.uint32),
        "players": np.array([0, 0], dtype=np.uint8),
        "scores": np.array([[0, 0, 0, 0]], dtype=np.int32),
        "pod_ids": np.array([0, 0], dtype=np.int64),
        "learner_positions_flat": np.array([0, 0], dtype=np.int64),
        "game_idx_within_pod": np.array([0, 0], dtype=np.int64),
        "decision_types": np.array([0, 3], dtype=np.int8),  # BID, CARD_PLAY
        "bid_contracts": np.array([[8, -1, -1, -1]], dtype=np.int8),  # learner bid Berač
    }
    shadow_scores = np.array([[[0, 0, 0, 0]]], dtype=np.int32)

    adapter = ShadowScoreRewardAdapter(
        negative_reward_multiplier=1.0,
        berac_bid_penalty=-10.0,
    )
    rewards = adapter.compute_rewards(active_raw, shadow_scores, [_make_pod()])

    # Penalty is moved to / kept on bid-phase step; cardplay terminal stays zero.
    np.testing.assert_allclose(rewards[0], -10.0, atol=1e-6)
    np.testing.assert_allclose(rewards[1], 0.0, atol=1e-6)
