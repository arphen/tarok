"""Unit tests for RustSelfPlay run-stat aggregation semantics."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def rust_self_play_module(monkeypatch: pytest.MonkeyPatch):
    """Import adapter module with a lightweight tarok_engine stub."""
    stub = types.SimpleNamespace(
        CONTRACT_OFFSET=0,
        CONTRACT_SIZE=10,
        run_self_play=lambda **_: {},
    )
    monkeypatch.setitem(sys.modules, "tarok_engine", stub)
    return importlib.import_module("training.adapters.self_play.rust_self_play_adapter")


def test_compute_run_stats_extracts_pairwise_outplaces(rust_self_play_module) -> None:
    adapter = rust_self_play_module.RustSelfPlay()

    raw = {
        "players": np.array(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            dtype=np.int32,
        ),
        "scores": np.array(
            [
                [10, 5, 10, 0],
                [0, 0, 5, 2],
                [3, 4, 3, 3],
            ],
            dtype=np.int32,
        ),
    }

    n_learner, mean_scores, seat_outcomes = adapter.compute_run_stats(
        raw=raw,
        seat_labels=["nn", "bot_v5", "bot_v6", "bot_m6"],
    )

    assert n_learner == 3
    assert mean_scores == pytest.approx((13 / 3, 3.0, 6.0, 5 / 3))

    # seat 1 comparisons: [10>5, 0==0, 3<4] -> (1, 1, 1)
    assert seat_outcomes[1] == (1, 1, 1)
    # seat 2 comparisons: [10==10, 0<5, 3==3] -> (0, 1, 2)
    assert seat_outcomes[2] == (0, 1, 2)
    # seat 3 comparisons: [10>0, 0<2, 3==3] -> (1, 1, 1)
    assert seat_outcomes[3] == (1, 1, 1)


def test_compute_run_stats_handles_missing_scores_with_warning(
    rust_self_play_module,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = rust_self_play_module.RustSelfPlay()

    raw = {
        "players": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "scores": np.array([], dtype=np.float32),
    }

    n_learner, mean_scores, seat_outcomes = adapter.compute_run_stats(
        raw=raw,
        seat_labels=["nn", "bot_v5", "bot_v6", "bot_m6"],
    )

    assert n_learner == 1
    assert mean_scores == (0.0, 0.0, 0.0, 0.0)
    assert seat_outcomes == {}
    assert "league Elo cannot be updated" in caplog.text


def test_compute_run_stats_uses_session_cumulative_scores(rust_self_play_module) -> None:
    adapter = rust_self_play_module.RustSelfPlay()

    raw = {
        "players": np.array(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            dtype=np.int32,
        ),
        # Per-game vs seat1: loss, win, loss, win -> 2W/2L
        # Session_size=2 makes sessions:
        #   s0 totals learner=0, opp1=2 (loss)
        #   s1 totals learner=0, opp1=2 (loss)
        # => learner_outplaces=0, opponent_outplaces=2
        "scores": np.array(
            [
                [1, 2, 0, 0],
                [-1, 0, 0, 0],
                [1, 2, 0, 0],
                [-1, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    }

    _n_learner, _mean_scores, seat_outcomes = adapter.compute_run_stats(
        raw=raw,
        seat_labels=["nn", "bot_v5", "bot_v6", "bot_m6"],
        session_size=2,
    )

    assert seat_outcomes[1] == (0, 2, 0)


def test_compute_run_stats_drops_leftover_games_for_outplace_units(
    rust_self_play_module,
) -> None:
    adapter = rust_self_play_module.RustSelfPlay()

    raw = {
        "players": np.array(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            dtype=np.int32,
        ),
        # First 4 games produce two draw sessions with session_size=2.
        # Last game would be learner win, but must be dropped for outplace units.
        "scores": np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [2, 2, 0, 0],
                [2, 2, 0, 0],
                [9, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    }

    _n_learner, _mean_scores, seat_outcomes = adapter.compute_run_stats(
        raw=raw,
        seat_labels=["nn", "bot_v5", "bot_v6", "bot_m6"],
        session_size=2,
    )

    assert seat_outcomes[1] == (0, 0, 2)


def test_compute_run_stats_accumulates_across_all_nn_seats(rust_self_play_module) -> None:
    """When multiple seats are labelled 'nn', each contributes to the outplace count."""
    adapter = rust_self_play_module.RustSelfPlay()

    # Seats 0 and 2 are nn; seats 1 and 3 are bots.
    raw = {
        "players": np.array(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            dtype=np.int32,
        ),
        "scores": np.array(
            [
                [5, 3, 2, 7],  # seat0 beats bot_v5; seat2 loses to bot_v5
                [4, 4, 6, 1],  # seat0 ties  bot_v5; seat2 beats bot_v5
                [1, 6, 3, 2],  # seat0 loses bot_v5; seat2 loses to bot_v5
            ],
            dtype=np.int32,
        ),
    }

    _n_learner, _mean_scores, seat_outcomes = adapter.compute_run_stats(
        raw=raw,
        seat_labels=["nn", "bot_v5", "nn", "bot_m6"],
        session_size=1,  # per-game comparison units to keep arithmetic clear
    )

    # bot_v5 (seat 1):
    #   seat0 vs seat1: [5>3, 4==4, 1<6] -> (1W, 1L, 1D)
    #   seat2 vs seat1: [2<3, 6>4, 3<6] -> (1W, 2L, 0D)
    #   total: (2W, 3L, 1D)
    assert seat_outcomes[1] == (2, 3, 1)

    # bot_m6 (seat 3):
    #   seat0 vs seat3: [5<7, 4>1, 1<2] -> (1W, 2L, 0D)
    #   seat2 vs seat3: [2<7, 6>1, 3>2] -> (2W, 1L, 0D)
    #   total: (3W, 3L, 0D)
    assert seat_outcomes[3] == (3, 3, 0)
