"""Unit tests for NumpyDuplicateArenaStats."""

from __future__ import annotations

import math

import numpy as np

from training.adapters.duplicate.numpy_arena_stats import NumpyDuplicateArenaStats
from training.entities.duplicate_run_result import DuplicateRunResult


def _make_result(
    *,
    challenger_scores: list[list[int]],
    defender_scores: list[list[int]],
    learner_seats: list[list[int]],
) -> DuplicateRunResult:
    """Build a synthetic DuplicateRunResult for N pods * G games/group.

    All arrays are sized so that ``active["scores"]`` contains one row per
    active game (n_pods * G rows), and ``shadow_scores`` is
    (n_pods, G, 4) with the defender's score placed at the same seat as
    the learner played in the paired active game.
    """
    n_pods = len(challenger_scores)
    G = len(challenger_scores[0])
    assert len(defender_scores) == n_pods and len(learner_seats) == n_pods
    assert all(len(row) == G for row in defender_scores)
    assert all(len(row) == G for row in learner_seats)

    n_games = n_pods * G
    scores = np.zeros((n_games, 4), dtype=np.int32)
    shadow = np.zeros((n_pods, G, 4), dtype=np.int32)

    active_game_ids = np.zeros((n_pods, G), dtype=np.int64)
    learner_positions = np.zeros((n_pods, G), dtype=np.int64)

    for p in range(n_pods):
        for g in range(G):
            gid = p * G + g
            seat = learner_seats[p][g]
            scores[gid, seat] = challenger_scores[p][g]
            shadow[p, g, seat] = defender_scores[p][g]
            active_game_ids[p, g] = gid
            learner_positions[p, g] = seat

    active = {"scores": scores}
    return DuplicateRunResult(
        active=active,
        shadow_scores=shadow,
        pod_ids=np.array([], dtype=np.int64),  # not used by stats adapter
        learner_positions=learner_positions,
        active_game_ids=active_game_ids,
    )


def test_zero_boards_returns_empty():
    stats = NumpyDuplicateArenaStats()
    result = DuplicateRunResult(
        active={"scores": np.zeros((0, 4), dtype=np.int32)},
        shadow_scores=np.zeros((0, 0, 4), dtype=np.int32),
        pod_ids=np.array([], dtype=np.int64),
        learner_positions=np.zeros((0, 0), dtype=np.int64),
        active_game_ids=np.zeros((0, 0), dtype=np.int64),
    )
    out = stats.compute(result)
    assert out.boards_played == 0
    assert out.mean_duplicate_advantage == 0.0


def test_challenger_wins_every_board():
    stats = NumpyDuplicateArenaStats()
    run = _make_result(
        challenger_scores=[[50, 50, 50, 50]],
        defender_scores=[[10, 10, 10, 10]],
        learner_seats=[[0, 1, 2, 3]],
    )
    out = stats.compute(run, bootstrap_samples=0)

    assert out.boards_played == 4
    assert out.challenger_mean_score == 50.0
    assert out.defender_mean_score == 10.0
    assert out.mean_duplicate_advantage == 40.0
    assert out.duplicate_advantage_std == 0.0
    assert out.imps_per_board == 0.4


def test_defender_wins_means_negative_advantage():
    stats = NumpyDuplicateArenaStats()
    run = _make_result(
        challenger_scores=[[-20, -30]],
        defender_scores=[[10, 20]],
        learner_seats=[[0, 1]],
    )
    out = stats.compute(run, bootstrap_samples=0)
    assert out.boards_played == 2
    assert out.mean_duplicate_advantage == -40.0


def test_imps_per_board_uses_score_scale():
    stats = NumpyDuplicateArenaStats()
    run = _make_result(
        challenger_scores=[[60, 40]],
        defender_scores=[[10, 10]],
        learner_seats=[[0, 1]],
    )
    out = stats.compute(run, score_scale=20.0, bootstrap_samples=0)
    # mean adv = ((60-10) + (40-10))/2 = 40; imps = 40 / 20 = 2.0
    assert out.imps_per_board == 2.0


def test_learner_seat_rotation_extracts_from_correct_seat():
    """Regression: the adapter must index shadow/challenger scores by the
    learner's seat for each game, not by seat 0 always."""
    stats = NumpyDuplicateArenaStats()
    run = _make_result(
        challenger_scores=[[100, 200, 300, 400]],
        defender_scores=[[10, 20, 30, 40]],
        learner_seats=[[0, 1, 2, 3]],
    )
    out = stats.compute(run, bootstrap_samples=0)
    assert out.challenger_mean_score == (100 + 200 + 300 + 400) / 4
    assert out.defender_mean_score == (10 + 20 + 30 + 40) / 4


def test_bootstrap_ci_brackets_mean_for_wide_sample():
    stats = NumpyDuplicateArenaStats()
    # 8 boards with advantage [-10, -5, 0, 5, 10, 15, 20, 25] → mean = 7.5
    ch = [[-5, 0, 5, 10, 15, 20, 25, 30]]
    df = [[5, 5, 5, 5, 5, 5, 5, 5]]
    seats = [[0, 1, 2, 3, 0, 1, 2, 3]]
    run = _make_result(
        challenger_scores=ch, defender_scores=df, learner_seats=seats,
    )
    out = stats.compute(run, bootstrap_samples=500, rng_seed=42)
    assert out.boards_played == 8
    assert out.mean_duplicate_advantage == 7.5
    # 95% CI must contain the mean
    assert out.ci_low_95 < out.mean_duplicate_advantage < out.ci_high_95


def test_bootstrap_ci_is_reproducible():
    stats = NumpyDuplicateArenaStats()
    run = _make_result(
        challenger_scores=[[10, 20, 30, 40]],
        defender_scores=[[5, 10, 15, 20]],
        learner_seats=[[0, 1, 2, 3]],
    )
    out1 = stats.compute(run, bootstrap_samples=200, rng_seed=123)
    out2 = stats.compute(run, bootstrap_samples=200, rng_seed=123)
    assert out1.ci_low_95 == out2.ci_low_95
    assert out1.ci_high_95 == out2.ci_high_95


def test_zero_bootstrap_samples_returns_nan_ci():
    stats = NumpyDuplicateArenaStats()
    run = _make_result(
        challenger_scores=[[50]],
        defender_scores=[[10]],
        learner_seats=[[0]],
    )
    out = stats.compute(run, bootstrap_samples=0)
    assert math.isnan(out.ci_low_95)
    assert math.isnan(out.ci_high_95)


def test_rejects_malformed_shadow_shape():
    stats = NumpyDuplicateArenaStats()
    bad = DuplicateRunResult(
        active={"scores": np.zeros((4, 4), dtype=np.int32)},
        shadow_scores=np.zeros((1, 4), dtype=np.int32),  # wrong rank
        pod_ids=np.array([], dtype=np.int64),
        learner_positions=np.zeros((1, 4), dtype=np.int64),
        active_game_ids=np.zeros((1, 4), dtype=np.int64),
    )
    try:
        stats.compute(bad)
    except ValueError as e:
        assert "shadow_scores" in str(e)
    else:
        raise AssertionError("expected ValueError for bad shadow_scores shape")
