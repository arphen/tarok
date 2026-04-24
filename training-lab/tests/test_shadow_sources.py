"""Unit tests for DuplicateShadowSourcePort adapters."""

from __future__ import annotations

import pytest

from training.adapters.duplicate.shadow_sources import (
    BestSnapshotShadowSource,
    HeuristicBotShadowSource,
    LeaguePoolShadowSource,
    PreviousIterationShadowSource,
    RelativeTrailingShadowSource,
    TrailingShadowSource,
    WeakestSnapshotShadowSource,
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


def test_weakest_snapshot_falls_back_when_pool_is_none():
    src = WeakestSnapshotShadowSource()
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=None) == LEARNER_PATH


def test_weakest_snapshot_falls_back_when_no_nn_checkpoints():
    src = WeakestSnapshotShadowSource()
    pool = _make_pool([_bot_entry("bv5", "bot_v5")])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == LEARNER_PATH


def test_weakest_snapshot_picks_lowest_elo():
    src = WeakestSnapshotShadowSource()
    pool = _make_pool([
        _nn_entry("A", "/tmp/A.ts", 1400.0),  # weakest
        _nn_entry("B", "/tmp/B.ts", 1800.0),
        _nn_entry("C", "/tmp/C.ts", 1600.0),
    ])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == "/tmp/A.ts"


def test_weakest_snapshot_ties_broken_by_games_played():
    src = WeakestSnapshotShadowSource()
    pool = _make_pool([
        _nn_entry("A", "/tmp/A.ts", 1300.0, games_played=10),
        _nn_entry("B", "/tmp/B.ts", 1300.0, games_played=50),  # tie on Elo, more games
        _nn_entry("C", "/tmp/C.ts", 1300.0, games_played=20),
    ])
    assert src.resolve(iteration=1, learner_ts_path=LEARNER_PATH, pool=pool) == "/tmp/B.ts"


def test_weakest_snapshot_ignores_non_nn_entries():
    src = WeakestSnapshotShadowSource()
    pool = _make_pool([
        _bot_entry("bv5", "bot_v5", elo=-9999.0),  # ignored despite very low Elo
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


def test_factory_creates_weakest_snapshot():
    src = create_shadow_source("weakest_snapshot")
    assert isinstance(src, WeakestSnapshotShadowSource)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="shadow_source"):
        create_shadow_source("does_not_exist")


# ---------------------------------------------------------------------------
# HeuristicBotShadowSource
# ---------------------------------------------------------------------------


def test_heuristic_shadow_resolve_returns_learner_path_placeholder():
    src = HeuristicBotShadowSource(seat_token="bot_v3")
    # Placeholder path: engine never dereferences it because no ``nn`` seat
    # sits in the shadow seating; heuristic bot occupies that slot.
    assert src.resolve(iteration=0, learner_ts_path=LEARNER_PATH, pool=None) == LEARNER_PATH
    assert src.resolve(iteration=5, learner_ts_path=LEARNER_PATH, pool=None) == LEARNER_PATH


def test_heuristic_shadow_exposes_seat_token():
    src = HeuristicBotShadowSource(seat_token="bot_v3")
    assert src.seat_token == "bot_v3"


def test_heuristic_shadow_rejects_unknown_label():
    with pytest.raises(ValueError, match="heuristic shadow bot label"):
        HeuristicBotShadowSource(seat_token="not_a_bot")


@pytest.mark.parametrize(
    "label",
    ["bot_v3", "bot_v5", "bot_v6", "bot_m6", "bot_lustrek", "bot_lapajne"],
)
def test_factory_creates_heuristic_shadow(label):
    src = create_shadow_source(label)
    assert isinstance(src, HeuristicBotShadowSource)
    assert src.seat_token == label


# ---------------------------------------------------------------------------
# TrailingShadowSource
# ---------------------------------------------------------------------------


def test_trailing_rejects_invalid_interval():
    with pytest.raises(ValueError, match="refresh_interval"):
        TrailingShadowSource(refresh_interval=0)


def test_trailing_falls_back_when_learner_file_missing(tmp_path):
    learner = str(tmp_path / "learner.pt")  # does not exist
    src = TrailingShadowSource(refresh_interval=5)
    # No file yet → fall back to learner path (iteration 0 bootstrap).
    assert src.resolve(iteration=0, learner_ts_path=learner, pool=None) == learner


def test_trailing_snapshots_and_holds_between_refreshes(tmp_path):
    learner = tmp_path / "learner.pt"
    learner.write_bytes(b"v0")

    src = TrailingShadowSource(refresh_interval=5)
    cached = TrailingShadowSource._cached_path(str(learner))

    # Iter 0: first refresh → snapshots v0.
    out0 = src.resolve(iteration=0, learner_ts_path=str(learner), pool=None)
    assert out0 == cached
    assert open(cached, "rb").read() == b"v0"

    # Mutate learner file; shadow must still reflect v0 for 4 more iterations.
    learner.write_bytes(b"v1")
    out1 = src.resolve(iteration=1, learner_ts_path=str(learner), pool=None)
    out4 = src.resolve(iteration=4, learner_ts_path=str(learner), pool=None)
    assert out1 == cached and out4 == cached
    assert open(cached, "rb").read() == b"v0", "shadow must lag during interval"

    # Iter 5: refresh boundary reached → picks up v1.
    out5 = src.resolve(iteration=5, learner_ts_path=str(learner), pool=None)
    assert out5 == cached
    assert open(cached, "rb").read() == b"v1"

    # And holds v1 for the next interval.
    learner.write_bytes(b"v2")
    out6 = src.resolve(iteration=6, learner_ts_path=str(learner), pool=None)
    assert open(cached, "rb").read() == b"v1"
    assert out6 == cached


def test_trailing_refreshes_when_cached_file_deleted(tmp_path):
    """Defensive: if the cached file disappears mid-run, the next resolve
    must rebuild it rather than returning a non-existent path."""
    learner = tmp_path / "learner.pt"
    learner.write_bytes(b"v0")
    src = TrailingShadowSource(refresh_interval=100)
    cached = src.resolve(iteration=0, learner_ts_path=str(learner), pool=None)
    import os as _os
    _os.remove(cached)
    # Still inside the interval, but file is gone → must rebuild.
    out = src.resolve(iteration=1, learner_ts_path=str(learner), pool=None)
    assert _os.path.exists(out)


def test_factory_creates_trailing(tmp_path):
    src = create_shadow_source("trailing", refresh_interval=7)
    assert isinstance(src, TrailingShadowSource)
    assert src._interval == 7


def test_factory_creates_relative_trailing():
    src = create_shadow_source("relative_trailing", refresh_interval=6)
    assert isinstance(src, RelativeTrailingShadowSource)
    assert src._lag == 6


def test_relative_trailing_rejects_invalid_lag():
    with pytest.raises(ValueError, match="lag_iterations"):
        RelativeTrailingShadowSource(lag_iterations=0)


def test_relative_trailing_uses_exact_fixed_lag_each_iteration(tmp_path):
    learner = tmp_path / "learner.pt"
    src = RelativeTrailingShadowSource(lag_iterations=3)

    for i in range(1, 8):
        # At start of iteration i, learner file contains weights from i-1.
        learner.write_bytes(f"w{i-1}".encode("ascii"))
        out = src.resolve(iteration=i, learner_ts_path=str(learner), pool=None)
        expected_iter = max(i - 3, 0)
        assert open(out, "rb").read() == f"w{expected_iter}".encode("ascii")
        assert src.last_target_iteration == expected_iter
