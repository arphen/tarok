"""Tests for Phase 3 actor-only mode in ppo_batch_preparation.

Covers:
- ``_broadcast_terminal_advantage`` helper (pure numpy)
- ``prepare_batched`` with ``raw["actor_only"] = True`` branch
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarok_model.encoding import CARD_ACTION_SIZE, STATE_SIZE
from training.adapters.ppo.ppo_batch_preparation import (
    _broadcast_terminal_advantage,
    prepare_batched,
)


def test_broadcast_terminal_advantage_single_trajectory_no_discount():
    # 3 steps, all same (game=0, player=0), terminal reward = 1.0 at step 3.
    terminal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gids = np.array([0, 0, 0], dtype=np.int64)
    plrs = np.array([0, 0, 0], dtype=np.int8)
    out = _broadcast_terminal_advantage(terminal, gids, plrs, gamma=1.0)
    assert out.tolist() == [1.0, 1.0, 1.0]


def test_broadcast_terminal_advantage_single_trajectory_with_discount():
    terminal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gids = np.array([0, 0, 0], dtype=np.int64)
    plrs = np.array([0, 0, 0], dtype=np.int8)
    out = _broadcast_terminal_advantage(terminal, gids, plrs, gamma=0.9)
    # step 0: 0.9^2 * 1 = 0.81; step 1: 0.9 * 1 = 0.9; step 2: 1.0
    assert out[0] == pytest.approx(0.81, abs=1e-5)
    assert out[1] == pytest.approx(0.9, abs=1e-5)
    assert out[2] == pytest.approx(1.0, abs=1e-5)


def test_broadcast_terminal_advantage_multiple_trajectories():
    # game 0 / player 0: 2 steps, terminal adv = 2.0
    # game 0 / player 1: 1 step,  terminal adv = -3.0
    # game 1 / player 0: 3 steps, terminal adv = 5.0
    terminal = np.array([0.0, 2.0, -3.0, 0.0, 0.0, 5.0], dtype=np.float32)
    gids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    plrs = np.array([0, 0, 1, 0, 0, 0], dtype=np.int8)
    out = _broadcast_terminal_advantage(terminal, gids, plrs, gamma=1.0)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(-3.0)
    assert out[3] == pytest.approx(5.0)
    assert out[4] == pytest.approx(5.0)
    assert out[5] == pytest.approx(5.0)


def test_broadcast_terminal_advantage_empty():
    out = _broadcast_terminal_advantage(
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int8),
        gamma=0.99,
    )
    assert out.shape == (0,)


def test_broadcast_terminal_advantage_not_sorted_input():
    # Steps for game 0/player 0 interleaved with game 1/player 0.
    # Engine may not guarantee contiguous layout — the helper must still work.
    terminal = np.array([0.0, 0.0, 0.0, 10.0, 7.0, 0.0], dtype=np.float32)
    gids = np.array([0, 1, 0, 0, 1, 1], dtype=np.int64)
    plrs = np.array([0, 0, 0, 0, 0, 0], dtype=np.int8)
    # game 0/player 0 rows: indices [0, 2, 3], terminal adv = 10.0 (at idx 3)
    # game 1/player 0 rows: indices [1, 4, 5], terminal adv = 7.0 (at idx 4)
    # But the helper uses the *last* row per trajectory in original order as
    # terminal; let's confirm behaviour by constructing an explicit case.
    # In our engine, rows are emitted in chronological order per trajectory
    # but not necessarily contiguous. We test: within each trajectory, in
    # engine-emission order, terminal is the last such row.
    # traj (0,0) rows in emission order: [0, 2, 3] -> terminal at 3 (val=10.0)
    # traj (1,0) rows in emission order: [1, 4, 5] -> terminal at 5 (val=0.0)
    # Hmm, that breaks our assumption. The reward adapter guarantees exactly
    # one nonzero entry per trajectory AT its terminal row — so the "terminal
    # value" extracted from the sorted array equals that nonzero entry.
    # For this test we adjust the setup: put the nonzero at the last row.
    terminal = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 7.0], dtype=np.float32)
    out = _broadcast_terminal_advantage(terminal, gids, plrs, gamma=1.0)
    # traj (0,0) rows [0,2,3] all get 10.0
    assert out[0] == pytest.approx(10.0)
    assert out[2] == pytest.approx(10.0)
    assert out[3] == pytest.approx(10.0)
    # traj (1,0) rows [1,4,5] all get 7.0
    assert out[1] == pytest.approx(7.0)
    assert out[4] == pytest.approx(7.0)
    assert out[5] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# prepare_batched actor-only branch
# ---------------------------------------------------------------------------


def _make_raw_actor_only(n: int, game_ids, players, terminal_rewards) -> dict:
    n_games = int(max(game_ids)) + 1 if n > 0 else 0
    return {
        "states": np.zeros((n, STATE_SIZE), dtype=np.float32),
        "actions": np.arange(n, dtype=np.int64) % CARD_ACTION_SIZE,
        "log_probs": np.zeros(n, dtype=np.float32),
        "values": np.zeros(n, dtype=np.float32),
        "decision_types": np.full(n, 3, dtype=np.int8),
        "game_modes": np.full(n, 2, dtype=np.int8),
        "legal_masks": np.ones((n, CARD_ACTION_SIZE), dtype=np.float32),
        "game_ids": np.asarray(game_ids, dtype=np.int64),
        "players": np.asarray(players, dtype=np.int8),
        "scores": np.zeros((max(n_games, 1), 4), dtype=np.float32),
        "declarers": np.full((max(n_games, 1),), -1, dtype=np.int8),
        "partners": np.full((max(n_games, 1),), -1, dtype=np.int8),
        "oracle_states": None,
        "behavioral_clone_mask": None,
        "traces": None,
        "precomputed_rewards": np.asarray(terminal_rewards, dtype=np.float32),
        "actor_only": True,
    }


def test_prepare_batched_actor_only_broadcasts_advantage():
    # One trajectory, 3 steps, terminal adv = 2.0
    raw = _make_raw_actor_only(
        n=3,
        game_ids=[0, 0, 0],
        players=[0, 0, 0],
        terminal_rewards=[0.0, 0.0, 2.0],
    )
    out = prepare_batched(raw, gamma=1.0)
    # After per-batch normalisation, all equal → (0, 0, 0) adv; but we
    # check the shape and that values column is zero (critic dropped).
    assert out["vad"].shape == (3, 3)
    assert torch.all(out["vad"][:, 0] == 0.0)  # old_values zero
    # col 1 = advantages, col 2 = returns; both equal (no bootstrap).
    assert torch.allclose(out["vad"][:, 1], out["vad"][:, 2])


def test_prepare_batched_actor_only_requires_precomputed_rewards():
    raw = _make_raw_actor_only(
        n=3,
        game_ids=[0, 0, 0],
        players=[0, 0, 0],
        terminal_rewards=[0.0, 0.0, 1.0],
    )
    raw["precomputed_rewards"] = None
    with pytest.raises(ValueError, match="precomputed_rewards"):
        prepare_batched(raw)


def test_prepare_batched_actor_only_two_trajectories_have_distinct_advantages():
    # Traj A: terminal adv = +1.0; Traj B: terminal adv = -1.0.
    # After normalisation, the two groups must have opposite signs.
    raw = _make_raw_actor_only(
        n=4,
        game_ids=[0, 0, 1, 1],
        players=[0, 0, 0, 0],
        terminal_rewards=[0.0, 1.0, 0.0, -1.0],
    )
    out = prepare_batched(raw, gamma=1.0)
    adv = out["vad"][:, 1].detach().cpu().numpy()
    assert adv[0] > 0 and adv[1] > 0  # traj A positive
    assert adv[2] < 0 and adv[3] < 0  # traj B negative
    # Mean is ~0 after normalisation.
    assert abs(adv.mean()) < 1e-5
