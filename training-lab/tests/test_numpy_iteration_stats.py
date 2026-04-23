"""Unit tests for NumpyDuplicateIterationStats adapter."""

from __future__ import annotations

import numpy as np

from training.adapters.duplicate.numpy_iteration_stats import NumpyDuplicateIterationStats
from training.entities.duplicate_pod import DuplicatePod


def _make_pod(learner_positions: tuple[int, ...], opponents=("bot_v5", "bot_m6", "bot_v6")):
    seatings: list[tuple[str, str, str, str]] = []
    for lp in learner_positions:
        row = [None, None, None, None]
        row[lp] = "nn"
        opp_iter = iter(opponents)
        for i in range(4):
            if i == lp:
                continue
            row[i] = next(opp_iter)
        seatings.append(tuple(row))  # type: ignore[arg-type]
    shadow = tuple(tuple("shadow" if s == "nn" else s for s in row) for row in seatings)
    return DuplicatePod(
        deck_seed=123,
        opponents=opponents,
        active_seatings=tuple(seatings),
        shadow_seatings=shadow,
        learner_positions=learner_positions,
    )


def test_compute_returns_empty_stats_for_no_pods():
    port = NumpyDuplicateIterationStats()
    stats = port.compute(
        active_raw={"scores": np.zeros((0, 4), dtype=np.int32)},
        shadow_scores=np.zeros((0, 0, 4), dtype=np.int32),
        pods=[],
        pod_ids=np.zeros(0, dtype=np.int64),
        learner_positions=np.zeros((0, 0), dtype=np.int8),
        active_game_ids=np.zeros((0, 0), dtype=np.int64),
    )
    assert stats.n_active_games == 0
    assert stats.opponent_outcomes == {}
    assert stats.mean_advantage == 0.0


def test_compute_buckets_per_opponent_and_counts_correctly():
    # One pod, 1 game. Learner sits at seat 0. Opponents fill the remaining
    # three seats in ascending order:
    #   seat 1 = opponents[0]="A",
    #   seat 2 = opponents[1]="B",
    #   seat 3 = opponents[2]="C"
    # (per RotationPairingAdapter: opp_idx = j if j < learner_pos else j-1)
    pod = _make_pod(learner_positions=(0,), opponents=("A", "B", "C"))
    active_scores = np.array([[40, 30, 50, 40]], dtype=np.int32)  # (n_games=1, 4 seats)
    shadow_scores = np.array([[[10, 0, 0, 0]]], dtype=np.int32)  # (pods=1, games=1, seats=4)
    learner_positions = np.array([[0]], dtype=np.int8)
    active_game_ids = np.array([[0]], dtype=np.int64)

    port = NumpyDuplicateIterationStats()
    stats = port.compute(
        active_raw={"scores": active_scores},
        shadow_scores=shadow_scores,
        pods=[pod],
        pod_ids=np.zeros(1, dtype=np.int64),
        learner_positions=learner_positions,
        active_game_ids=active_game_ids,
    )

    # learner score = 40 at seat 0.
    # vs A (seat 1, score=30): 40 > 30 → learner wins
    # vs B (seat 2, score=50): 40 < 50 → learner loses
    # vs C (seat 3, score=40): tie → draw
    assert stats.opponent_outcomes == {
        "A": (1, 0, 0),
        "B": (0, 1, 0),
        "C": (0, 0, 1),
    }
    assert stats.opponent_games == {"A": 1, "B": 1, "C": 1}

    # duplicate advantage = (40 - 10)/100 = 0.30, only 1 sample → std=0.
    assert stats.n_active_games == 1
    assert abs(stats.mean_advantage - 0.3) < 1e-6
    assert stats.advantage_std == 0.0


def test_compute_learner_at_different_seats_still_buckets_by_opponent_label():
    # Two games in one pod: learner at seat 0, then seat 2. Same opponent triple.
    pod = _make_pod(learner_positions=(0, 2), opponents=("A", "B", "C"))
    # Game 0: learner@seat0, seats = [learner, A, B, C]
    # Game 1: learner@seat2, seats = [A, B, learner, C]  (cyclic fill skips lp)
    active_scores = np.array(
        [
            [10, 20, 30, 40],  # game 0
            [20, 30, 10, 40],  # game 1
        ],
        dtype=np.int32,
    )
    shadow_scores = np.array(
        [[[0, 0, 0, 0], [0, 0, 0, 0]]],  # all zeros
        dtype=np.int32,
    )
    learner_positions = np.array([[0, 2]], dtype=np.int8)
    active_game_ids = np.array([[0, 1]], dtype=np.int64)

    port = NumpyDuplicateIterationStats()
    stats = port.compute(
        active_raw={"scores": active_scores},
        shadow_scores=shadow_scores,
        pods=[pod],
        pod_ids=np.zeros(2, dtype=np.int64),
        learner_positions=learner_positions,
        active_game_ids=active_game_ids,
    )

    # Each opponent faced the learner twice (once per game). Total 6 comparisons.
    total = sum(lo + oo + d for lo, oo, d in stats.opponent_outcomes.values())
    assert total == 6
    # Each opponent label faced the learner exactly twice.
    for token in ("A", "B", "C"):
        lo, oo, d = stats.opponent_outcomes[token]
        assert lo + oo + d == 2

    # Game 0 advantage = 10/100 = 0.10, game 1 advantage = 10/100 = 0.10.
    assert stats.n_active_games == 2
    assert abs(stats.mean_advantage - 0.1) < 1e-6


def test_compute_raises_on_shadow_shape_mismatch():
    pod = _make_pod(learner_positions=(0,))
    port = NumpyDuplicateIterationStats()
    try:
        port.compute(
            active_raw={"scores": np.zeros((1, 4), dtype=np.int32)},
            shadow_scores=np.zeros((1, 1, 3), dtype=np.int32),  # wrong last dim
            pods=[pod],
            pod_ids=np.zeros(1, dtype=np.int64),
            learner_positions=np.array([[0]], dtype=np.int8),
            active_game_ids=np.array([[0]], dtype=np.int64),
        )
    except ValueError as e:
        assert "shadow_scores shape" in str(e)
    else:
        raise AssertionError("expected ValueError on shape mismatch")
