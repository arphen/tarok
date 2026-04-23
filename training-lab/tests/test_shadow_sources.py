"""Unit tests for DuplicateShadowSourcePort adapters."""

from __future__ import annotations

import pytest

from training.adapters.duplicate.shadow_sources import (
    BestSnapshotShadowSource,
    LeaguePoolShadowSource,
    PreviousIterationShadowSource,
    create_shadow_source,
)
from training.entities.league import (
    LeagueConfig,
    LeagueOpponent,
    LeaguePool,
    LeaguePoolEntry,
)


LEARNER_PATH = "/tmp/learner.ts"


def _make_pool(entries: list[LeaguePoolEntry], learner_elo: float = 1500.0) -> LeaguePool:
    pool = LeaguePool(config=LeagueConfig(), entries=list(entries), learner_elo=learner_elo)
    return pool


def _nn_entry(name: str, path: str, elo: float, games_played: int = 0) -> LeaguePoolEntry:
    return LeaguePoolEntry(
        opponent=LeagueOpponent(name=name, type="nn_checkpoint", path=path),
        elo=elo,
        games_played=games_played,
    )


def _bot_entry(name: str, type_: str, elo: float = 1500.0) -> LeaguePoolEntry:
    return LeaguePoolEntry(
        opponent=LeagueOpponent(name=name, type=type_),  # type: ignore[arg-type]
        elo=elo,
    )


# ---------------------------------------------------------------------------
# PreviousIterationShadowSource
# ---------------------------------------------------------------------------


def test_previous_iteration_returns_learner_ts_path_on_iter_0():
    src = PreviousIterationShadowSource()
    assert src.resolve(iteration=0, learner_ts_path=LEARNER_PATH, pool=None) == LEARNER_PATH


def test_previous_iteration_returns_learner_ts_path_on_iter_n():
    src = PreviousIterationShadowSource()
    # The learner ts file on disk at iter start still holds previous iter's weights
    # because ExportModel overwrites at END of RunIteration.execute.
    assert src.resolve(iteration=42, learner_ts_path=LEARNER_PATH, pool=None) == LEARNER_PATH


def test_previous_iteration_ignores_pool():
    src = PreviousIterationShadowSource()
    pool = _make_pool([_nn_entry("snap-1", "/tmp/snap1.ts", 1600.0)])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == LEARNER_PATH


# ---------------------------------------------------------------------------
# LeaguePoolShadowSource
# ---------------------------------------------------------------------------


def test_league_pool_falls_back_when_pool_is_none():
    src = LeaguePoolShadowSource(rng_seed=0)
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=None) == LEARNER_PATH


def test_league_pool_falls_back_when_no_nn_checkpoints():
    src = LeaguePoolShadowSource(rng_seed=0)
    pool = _make_pool([_bot_entry("bv5", "bot_v5"), _bot_entry("bv6", "bot_v6")])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == LEARNER_PATH


def test_league_pool_samples_nn_checkpoint_path():
    src = LeaguePoolShadowSource(rng_seed=0)
    pool = _make_pool([
        _bot_entry("bv5", "bot_v5"),
        _nn_entry("snap-A", "/tmp/A.ts", 1500.0),
    ])
    # Only one nn_checkpoint candidate → must pick it.
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == "/tmp/A.ts"


def test_league_pool_seeded_rng_is_reproducible():
    pool = _make_pool([
        _nn_entry("A", "/tmp/A.ts", 1400.0),
        _nn_entry("B", "/tmp/B.ts", 1500.0),
        _nn_entry("C", "/tmp/C.ts", 1600.0),
        _nn_entry("D", "/tmp/D.ts", 1700.0),
    ])
    s1 = LeaguePoolShadowSource(rng_seed=123)
    s2 = LeaguePoolShadowSource(rng_seed=123)
    seq1 = [s1.resolve(iteration=i, learner_ts_path=LEARNER_PATH, pool=pool) for i in range(8)]
    seq2 = [s2.resolve(iteration=i, learner_ts_path=LEARNER_PATH, pool=pool) for i in range(8)]
    assert seq1 == seq2


def test_league_pool_gaussian_biases_near_learner_elo():
    # Many entries spread across Elo range; learner_elo centred on B.
    pool = _make_pool(
        entries=[
            _nn_entry("Far1", "/tmp/far1.ts", 500.0),
            _nn_entry("Near", "/tmp/near.ts", 1500.0),
            _nn_entry("Far2", "/tmp/far2.ts", 2500.0),
        ],
        learner_elo=1500.0,
    )
    src = LeaguePoolShadowSource(rng_seed=0)
    N = 200
    picks = [
        src.resolve(iteration=i, learner_ts_path=LEARNER_PATH, pool=pool) for i in range(N)
    ]
    near_count = picks.count("/tmp/near.ts")
    # With sigma=200 Elo, 'near' should dominate heavily vs the 1000-Elo-away peers.
    assert near_count > 0.7 * N


# ---------------------------------------------------------------------------
# BestSnapshotShadowSource
# ---------------------------------------------------------------------------


def test_best_snapshot_falls_back_when_pool_is_none():
    src = BestSnapshotShadowSource()
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=None) == LEARNER_PATH


def test_best_snapshot_falls_back_when_no_nn_checkpoints():
    src = BestSnapshotShadowSource()
    pool = _make_pool([_bot_entry("bv5", "bot_v5")])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == LEARNER_PATH


def test_best_snapshot_picks_highest_elo():
    src = BestSnapshotShadowSource()
    pool = _make_pool([
        _nn_entry("A", "/tmp/A.ts", 1400.0),
        _nn_entry("B", "/tmp/B.ts", 1800.0),  # best
        _nn_entry("C", "/tmp/C.ts", 1600.0),
    ])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == "/tmp/B.ts"


def test_best_snapshot_ties_broken_by_games_played():
    src = BestSnapshotShadowSource()
    pool = _make_pool([
        _nn_entry("A", "/tmp/A.ts", 1700.0, games_played=10),
        _nn_entry("B", "/tmp/B.ts", 1700.0, games_played=50),  # tie on Elo, more games
        _nn_entry("C", "/tmp/C.ts", 1700.0, games_played=20),
    ])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == "/tmp/B.ts"


def test_best_snapshot_ignores_non_nn_entries():
    src = BestSnapshotShadowSource()
    pool = _make_pool([
        _bot_entry("bvbig", "bot_v5", elo=9999.0),  # should be ignored despite high Elo
        _nn_entry("A", "/tmp/A.ts", 1500.0),
    ])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == "/tmp/A.ts"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_creates_previous_iteration():
    src = create_shadow_source("previous_iteration")
    assert isinstance(src, PreviousIterationShadowSource)


def test_factory_creates_league_pool():
    src = create_shadow_source("league_pool", rng_seed=7)
    assert isinstance(src, LeaguePoolShadowSource)


def test_factory_creates_best_snapshot():
    src = create_shadow_source("best_snapshot")
    assert isinstance(src, BestSnapshotShadowSource)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="shadow_source"):
        create_shadow_source("does_not_exist")
