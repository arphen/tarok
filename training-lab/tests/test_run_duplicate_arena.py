"""Unit tests for RunDuplicateArena use case."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from training.adapters.duplicate.numpy_arena_stats import NumpyDuplicateArenaStats
from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
from training.entities.duplicate_run_result import DuplicateRunResult
from training.use_cases.run_duplicate_arena import RunDuplicateArena


def _fake_selfplay_with_scores(
    challenger_per_game: int, defender_per_game: int,
) -> MagicMock:
    """Return a selfplay mock whose run_seeded_pods produces a
    DuplicateRunResult with the given challenger/defender scores on every
    board."""
    selfplay = MagicMock()

    def _run_seeded_pods(*, learner_path, shadow_path, pods, **_kwargs):
        n_pods = len(pods)
        games_per_group = 4  # rotation_8game
        n_games = n_pods * games_per_group
        scores = np.zeros((n_games, 4), dtype=np.int32)
        shadow = np.zeros((n_pods, games_per_group, 4), dtype=np.int32)
        active_game_ids = np.zeros((n_pods, games_per_group), dtype=np.int64)
        learner_positions = np.zeros((n_pods, games_per_group), dtype=np.int64)
        for p in range(n_pods):
            for g in range(games_per_group):
                gid = p * games_per_group + g
                seat = g  # rotate learner seat 0..3 across pod
                scores[gid, seat] = challenger_per_game
                shadow[p, g, seat] = defender_per_game
                active_game_ids[p, g] = gid
                learner_positions[p, g] = seat
        return DuplicateRunResult(
            active={"scores": scores},
            shadow_scores=shadow,
            pod_ids=np.array([], dtype=np.int64),
            learner_positions=learner_positions,
            active_game_ids=active_game_ids,
        )

    selfplay.run_seeded_pods.side_effect = _run_seeded_pods
    return selfplay


def test_execute_builds_pods_plays_and_aggregates():
    selfplay = _fake_selfplay_with_scores(challenger_per_game=80, defender_per_game=20)
    uc = RunDuplicateArena(
        selfplay=selfplay,
        pairing=RotationPairingAdapter(pairing="rotation_8game"),
        stats=NumpyDuplicateArenaStats(),
    )

    result = uc.execute(
        challenger_path="/tmp/challenger.ts",
        defender_path="/tmp/defender.ts",
        n_boards=8,
        rng_seed=0,
        concurrency=1,
        bootstrap_samples=0,
    )

    # 8 boards // 4 = 2 pods, 4 games each → 8 boards
    assert result.boards_played == 8
    assert result.challenger_mean_score == 80.0
    assert result.defender_mean_score == 20.0
    assert result.mean_duplicate_advantage == 60.0
    assert result.imps_per_board == 0.6

    # Challenger/defender paths forwarded correctly
    call = selfplay.run_seeded_pods.call_args
    assert call.kwargs["learner_path"] == "/tmp/challenger.ts"
    assert call.kwargs["shadow_path"] == "/tmp/defender.ts"


def test_execute_rejects_nonpositive_n_boards():
    uc = RunDuplicateArena(
        selfplay=MagicMock(),
        pairing=RotationPairingAdapter(),
        stats=NumpyDuplicateArenaStats(),
    )
    with pytest.raises(ValueError, match="n_boards"):
        uc.execute(
            challenger_path="/a.ts", defender_path="/b.ts", n_boards=0,
        )


def test_execute_rejects_negative_explore_rate():
    uc = RunDuplicateArena(
        selfplay=MagicMock(),
        pairing=RotationPairingAdapter(),
        stats=NumpyDuplicateArenaStats(),
    )
    with pytest.raises(ValueError, match="explore_rate"):
        uc.execute(
            challenger_path="/a.ts",
            defender_path="/b.ts",
            n_boards=8,
            explore_rate=-0.1,
        )


def test_small_n_boards_still_produces_at_least_one_pod():
    selfplay = _fake_selfplay_with_scores(challenger_per_game=30, defender_per_game=10)
    uc = RunDuplicateArena(
        selfplay=selfplay,
        pairing=RotationPairingAdapter(),
        stats=NumpyDuplicateArenaStats(),
    )
    result = uc.execute(
        challenger_path="/a.ts",
        defender_path="/b.ts",
        n_boards=1,  # still allocates 1 pod = 4 boards
        bootstrap_samples=0,
    )
    assert result.boards_played == 4
