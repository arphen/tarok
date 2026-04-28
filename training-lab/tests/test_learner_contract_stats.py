"""Tests for per-contract learner diagnostics."""

from __future__ import annotations

import numpy as np

from training.use_cases.learner_contract_stats import compute_learner_contract_stats


def test_bid_reward_average_is_computed_per_bid_type() -> None:
    # Two games, learner at seat 0.
    # Game 0: learner bids berac, final contract solo_one, learner reward +40.
    # Game 1: learner bids berac, final contract berac, learner reward -60.
    raw = {
        "contracts": np.array([6, 8], dtype=np.int8),
        "declarers": np.array([2, 0], dtype=np.int8),
        "bid_contracts": np.array(
            [
                [8, -1, 6, -1],
                [8, -1, -1, -1],
            ],
            dtype=np.int8,
        ),
        "scores": np.array(
            [
                [10, 0, 20, 0],
                [-30, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        "reward_scores": np.array(
            [
                [40, 0, 10, 0],
                [-60, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    }

    stats = compute_learner_contract_stats(raw, learner_seats=[0])

    berac = stats["berac"]
    assert int(berac["bids_made"]) == 2
    assert int(berac["bid_reward_count"]) == 2
    assert float(berac["bid_reward_sum"]) == -20.0


def test_klop_played_and_averages_are_reported() -> None:
    # Three games, learner at seat 0 in all games.
    # Games 0-1 are final klop; game 2 final berac.
    raw = {
        "contracts": np.array([0, 0, 8], dtype=np.int8),
        "declarers": np.array([-1, -1, 0], dtype=np.int8),
        "bid_contracts": np.array(
            [
                [0, -1, -1, -1],
                [0, -1, -1, -1],
                [8, -1, -1, -1],
            ],
            dtype=np.int8,
        ),
        "scores": np.array(
            [
                [-20, 10, 5, 5],
                [30, -10, -10, -10],
                [50, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        "reward_scores": np.array(
            [
                [-20, 10, 5, 5],
                [30, -10, -10, -10],
                [40, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    }

    stats = compute_learner_contract_stats(raw, learner_seats=[0])

    klop = stats["klop"]
    assert int(klop["played_count"]) == 2
    assert float(klop["played_score_sum"]) == 10.0  # -20 + 30
    assert float(klop["played_reward_sum"]) == 10.0  # -20 + 30


def test_per_game_learner_seat_is_used_for_duplicate_rotation() -> None:
    # Two games with rotating learner seat. Both are klop finals.
    # Learner seat game0=0, game1=2.
    raw = {
        "contracts": np.array([0, 0], dtype=np.int8),
        "declarers": np.array([-1, -1], dtype=np.int8),
        "bid_contracts": np.array(
            [
                [0, -1, -1, -1],
                [-1, -1, 0, -1],
            ],
            dtype=np.int8,
        ),
        "scores": np.array(
            [
                [10, 20, 30, 40],
                [1, 2, 3, 4],
            ],
            dtype=np.int32,
        ),
        "reward_scores": np.array(
            [
                [11, 21, 31, 41],
                [5, 6, 7, 8],
            ],
            dtype=np.int32,
        ),
    }

    stats = compute_learner_contract_stats(
        raw,
        learner_seats=[0, 1, 2, 3],
        learner_seat_per_game=np.array([0, 2], dtype=np.int8),
    )

    klop = stats["klop"]
    # Pull only seat0 from game0 and seat2 from game1.
    assert int(klop["played_count"]) == 2
    assert float(klop["played_score_sum"]) == 13.0  # 10 + 3
    assert float(klop["played_reward_sum"]) == 18.0  # 11 + 7


def test_pass_count_and_average_reward_are_reported() -> None:
    # Learner seat 0 passes twice; pass row should carry both count and
    # average bid reward from learner reward_scores.
    raw = {
        "contracts": np.array([6, 6], dtype=np.int8),
        "declarers": np.array([1, 2], dtype=np.int8),
        "bid_contracts": np.array(
            [
                [-1, 6, -1, -1],
                [-1, -1, 6, -1],
            ],
            dtype=np.int8,
        ),
        "scores": np.array(
            [
                [5, 10, 0, 0],
                [-3, 0, 8, 0],
            ],
            dtype=np.int32,
        ),
        "reward_scores": np.array(
            [
                [12, 7, 0, 0],
                [-4, 0, 9, 0],
            ],
            dtype=np.int32,
        ),
    }

    stats = compute_learner_contract_stats(raw, learner_seats=[0])

    passed = stats["pass"]
    assert int(passed["bids_made"]) == 2
    assert int(passed["bid_reward_count"]) == 2
    assert float(passed["bid_reward_sum"]) == 8.0  # 12 + (-4)
