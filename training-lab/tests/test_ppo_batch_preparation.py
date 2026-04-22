"""Unit tests for PPO batch preparation (terminal masking, normalisation, invariants)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarok_model.encoding import CARD_ACTION_SIZE, STATE_SIZE
from training.adapters.ppo.ppo_batch_preparation import prepare_batched


def _make_raw(
    n: int,
    *,
    game_ids: np.ndarray | None = None,
    players: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    values: np.ndarray | None = None,
    oracle_states: np.ndarray | None = None,
    behavioral_clone_mask: np.ndarray | None = None,
    traces: list[dict] | None = None,
    declarers: np.ndarray | None = None,
    partners: np.ndarray | None = None,
) -> dict:
    states = np.zeros((n, STATE_SIZE), dtype=np.float32)
    actions = np.arange(n, dtype=np.int64) % CARD_ACTION_SIZE
    log_probs = np.zeros(n, dtype=np.float32)
    _values = np.zeros(n, dtype=np.float32) if values is None else values.astype(np.float32)
    decision_types = np.full(n, 3, dtype=np.int8)  # DT_CARD_PLAY
    game_modes = np.full(n, 2, dtype=np.int8)
    _game_ids = np.zeros(n, dtype=np.int64) if game_ids is None else game_ids.astype(np.int64)
    _players = np.zeros(n, dtype=np.int8) if players is None else players.astype(np.int8)
    n_games = int(_game_ids.max()) + 1 if n > 0 else 0
    _scores = (
        np.zeros((max(n_games, 1), 4), dtype=np.float32) if scores is None else scores.astype(np.float32)
    )
    legal_masks = np.ones((n, CARD_ACTION_SIZE), dtype=np.float32)
    raw: dict = {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "values": _values,
        "decision_types": decision_types,
        "game_modes": game_modes,
        "legal_masks": legal_masks,
        "game_ids": _game_ids,
        "players": _players,
        "scores": _scores,
        "oracle_states": oracle_states,
        "behavioral_clone_mask": behavioral_clone_mask,
        "traces": traces,
        "declarers": np.full((max(n_games, 1),), -1, dtype=np.int8)
        if declarers is None
        else declarers.astype(np.int8),
        "partners": np.full((max(n_games, 1),), -1, dtype=np.int8)
        if partners is None
        else partners.astype(np.int8),
    }
    return raw


def test_prepare_batched_returns_tensors_with_expected_shapes() -> None:
    raw = _make_raw(4)
    out = prepare_batched(raw)

    assert isinstance(out["states"], torch.Tensor)
    assert out["states"].shape == (4, STATE_SIZE)
    assert out["actions"].dtype == torch.int64
    assert out["vad"].shape == (4, 3)  # old_values, advantages, returns
    assert out["legal_masks"].shape == (4, CARD_ACTION_SIZE)


def test_prepare_batched_normalises_advantages_to_zero_mean_unit_std() -> None:
    # Two games, each with a positive reward on the final step so GAE produces
    # a real advantage signal after terminal-masking.
    n = 8
    game_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    players = np.zeros(n, dtype=np.int8)
    scores = np.array([[30.0, -10.0, -10.0, -10.0], [60.0, -20.0, -20.0, -20.0]], dtype=np.float32)
    values = np.linspace(0.0, 1.0, n, dtype=np.float32)

    out = prepare_batched(_make_raw(n, game_ids=game_ids, players=players, scores=scores, values=values))

    advantages = out["vad"][:, 1].numpy()
    assert advantages.mean() == pytest.approx(0.0, abs=1e-5)
    assert advantages.std() == pytest.approx(1.0, abs=1e-4)


def test_prepare_batched_zeroes_non_terminal_rewards() -> None:
    # Three-step trajectory of a single (game, player): only the last step sees
    # the final score. With gamma=1, lambda=1 the returns should equal the score
    # on every step, not scale with step count (would indicate reward leakage).
    n = 3
    game_ids = np.zeros(n, dtype=np.int64)
    players = np.zeros(n, dtype=np.int8)
    scores = np.array([[50.0, -50.0, 0.0, 0.0]], dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)

    out = prepare_batched(
        _make_raw(n, game_ids=game_ids, players=players, scores=scores, values=values),
        gamma=1.0,
        gae_lambda=1.0,
    )

    returns = out["vad"][:, 2].numpy()
    expected = 50.0 / 100.0  # rewards are divided by 100
    assert returns == pytest.approx(np.full(n, expected), abs=1e-6)


def test_prepare_batched_raises_when_game_ids_exceed_scores_rows() -> None:
    # Rust adapter bug: game_ids references games past the scores table.
    # Must raise, not silently wrap via modulo.
    n = 2
    game_ids = np.array([0, 5], dtype=np.int64)
    scores = np.zeros((2, 4), dtype=np.float32)

    raw = _make_raw(n, game_ids=game_ids, scores=scores)

    with pytest.raises(ValueError, match="game_ids contain id"):
        prepare_batched(raw)


def test_prepare_batched_rejects_negative_game_ids() -> None:
    raw = _make_raw(2, game_ids=np.array([0, -1], dtype=np.int64))
    with pytest.raises(ValueError, match="game_ids must be non-negative"):
        prepare_batched(raw)


def test_prepare_batched_defaults_behavioral_clone_mask_to_false() -> None:
    out = prepare_batched(_make_raw(5))
    mask = out["behavioral_clone_mask"]
    assert mask.shape == (5,)
    assert mask.dtype == torch.bool
    assert not mask.any()


def test_prepare_batched_preserves_explicit_behavioral_clone_mask() -> None:
    mask = np.array([True, False, True, False], dtype=bool)
    out = prepare_batched(_make_raw(4, behavioral_clone_mask=mask))
    assert out["behavioral_clone_mask"].dtype == torch.bool
    assert out["behavioral_clone_mask"].tolist() == mask.tolist()


def test_prepare_batched_oracle_states_optional() -> None:
    raw = _make_raw(3)
    assert raw["oracle_states"] is None
    out_none = prepare_batched(raw)
    assert out_none["oracle_states"] is None

    from tarok_model.encoding import ORACLE_STATE_SIZE

    oracle_states = np.random.default_rng(0).standard_normal((3, ORACLE_STATE_SIZE), dtype=np.float32)
    out = prepare_batched(_make_raw(3, oracle_states=oracle_states))
    assert isinstance(out["oracle_states"], torch.Tensor)
    assert out["oracle_states"].shape == (3, ORACLE_STATE_SIZE)


def test_shaped_rewards_skis_beats_mond_in_same_trick() -> None:
    # One-step trajectories for each player so returns mirror terminal reward.
    raw = _make_raw(
        4,
        game_ids=np.zeros(4, dtype=np.int64),
        players=np.array([0, 1, 2, 3], dtype=np.int8),
        scores=np.zeros((1, 4), dtype=np.float32),
        values=np.zeros(4, dtype=np.float32),
        traces=[
            {
                "dealer": 0,
                # Trick order: (player, card_idx)
                # Player 1: Mond (20), Player 2: Skis (21)
                "cards_played": [(1, 20), (2, 21), (3, 22), (0, 23)],
            }
        ],
    )

    out = prepare_batched(raw, gamma=1.0, gae_lambda=1.0)
    returns = out["vad"][:, 2].numpy()

    mond_return = float(returns[1])
    skis_return = float(returns[2])
    assert skis_return > 0.0
    assert mond_return < 0.0
    assert skis_return > mond_return


def test_shaped_rewards_pagat_beats_skis_and_mond_in_same_trick() -> None:
    raw = _make_raw(
        4,
        game_ids=np.zeros(4, dtype=np.int64),
        players=np.array([0, 1, 2, 3], dtype=np.int8),
        scores=np.zeros((1, 4), dtype=np.float32),
        values=np.zeros(4, dtype=np.float32),
        traces=[
            {
                "dealer": 0,
                # Mond + Skis + Pagat in one trick.
                "cards_played": [(1, 20), (2, 21), (3, 0), (0, 22)],
            }
        ],
    )

    out = prepare_batched(raw, gamma=1.0, gae_lambda=1.0)
    returns = out["vad"][:, 2].numpy()

    pagat_return = float(returns[3])
    mond_return = float(returns[1])
    skis_return = float(returns[2])
    assert pagat_return > 0.0
    assert mond_return < 0.0
    assert skis_return < 0.0


def test_shaped_rewards_capture_opponent_mond_is_positive() -> None:
    # Declarer team is players 0 and 2. Player 1 (opponent) loses Mond to player 0.
    raw = _make_raw(
        4,
        game_ids=np.zeros(4, dtype=np.int64),
        players=np.array([0, 1, 2, 3], dtype=np.int8),
        scores=np.zeros((1, 4), dtype=np.float32),
        values=np.zeros(4, dtype=np.float32),
        declarers=np.array([0], dtype=np.int8),
        partners=np.array([2], dtype=np.int8),
        traces=[
            {
                "dealer": 0,
                # Lead is suit; player 0 plays tarok Skis and wins, capturing player 1's Mond.
                "cards_played": [(1, 20), (2, 22), (3, 23), (0, 21)],
            }
        ],
    )

    out = prepare_batched(raw, gamma=1.0, gae_lambda=1.0)
    returns = out["vad"][:, 2].numpy()

    winner_return = float(returns[0])
    mond_owner_return = float(returns[1])
    assert winner_return > 0.0
    assert mond_owner_return < 0.0


# ---------------------------------------------------------------------------
# Duplicate RL: precomputed_rewards fallback (docs/double_rl.md §4.1).
# ---------------------------------------------------------------------------


def test_precomputed_rewards_override_replaces_score_extraction() -> None:
    """When ``precomputed_rewards`` is present, ``scores`` is ignored entirely."""
    n = 3
    game_ids = np.zeros(n, dtype=np.int64)
    players = np.zeros(n, dtype=np.int8)
    # Rust-computed scores that would normally drive reward...
    scores = np.array([[999.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    raw = _make_raw(n, game_ids=game_ids, players=players, scores=scores, values=values)

    # ...but the duplicate adapter injects its own reward array whose terminal
    # entry is the duplicate advantage. Non-terminals are zero.
    duplicate_reward = np.array([0.0, 0.0, 0.42], dtype=np.float32)
    raw["precomputed_rewards"] = duplicate_reward

    out = prepare_batched(raw, gamma=1.0, gae_lambda=1.0)
    returns = out["vad"][:, 2].numpy()
    # With gamma=1, lambda=1, the constant-return = terminal reward = 0.42.
    assert returns == pytest.approx(np.full(n, 0.42), abs=1e-6)


def test_precomputed_rewards_suppresses_special_shaped_bonuses() -> None:
    """Duplicate mode disables trick-shaping; both tables saw the same trick."""
    raw = _make_raw(
        4,
        game_ids=np.zeros(4, dtype=np.int64),
        players=np.array([0, 1, 2, 3], dtype=np.int8),
        scores=np.zeros((1, 4), dtype=np.float32),
        values=np.zeros(4, dtype=np.float32),
        traces=[
            {
                "dealer": 0,
                # A Mond-loses-to-Skis trick that would normally shape rewards.
                "cards_played": [(1, 20), (2, 21), (3, 22), (0, 23)],
            }
        ],
    )
    raw["precomputed_rewards"] = np.zeros(4, dtype=np.float32)

    out = prepare_batched(raw, gamma=1.0, gae_lambda=1.0)
    returns = out["vad"][:, 2].numpy()
    # Bonuses would make Mond-owner negative and Skis-owner positive. With
    # duplicate rewards injected, they should all be zero.
    assert returns == pytest.approx(np.zeros(4), abs=1e-6)


def test_precomputed_rewards_shape_mismatch_raises() -> None:
    raw = _make_raw(3)
    raw["precomputed_rewards"] = np.zeros(5, dtype=np.float32)  # wrong length
    with pytest.raises(ValueError, match="precomputed_rewards shape"):
        prepare_batched(raw)


def test_missing_precomputed_rewards_uses_legacy_path() -> None:
    """``precomputed_rewards=None`` preserves the current behaviour exactly."""
    n = 3
    game_ids = np.zeros(n, dtype=np.int64)
    players = np.zeros(n, dtype=np.int8)
    scores = np.array([[80.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)

    raw = _make_raw(n, game_ids=game_ids, players=players, scores=scores, values=values)
    assert "precomputed_rewards" not in raw

    out = prepare_batched(raw, gamma=1.0, gae_lambda=1.0)
    returns = out["vad"][:, 2].numpy()
    assert returns == pytest.approx(np.full(n, 0.80), abs=1e-6)
