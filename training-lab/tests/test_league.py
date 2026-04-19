"""Tests for the League Play domain logic and use cases."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from training.entities.league import (
    LeagueConfig,
    LeagueOpponent,
    LeaguePool,
    LeaguePoolEntry,
)
from training.adapters.persistence import JsonLeagueStatePersistence
from training.use_cases.sample_league_seats import SampleLeagueSeats
from training.use_cases.update_league_elo import UpdateLeagueElo


# ---------------------------------------------------------------------------
# LeagueOpponent.seat_token
# ---------------------------------------------------------------------------


def test_league_opponent_seat_token_bot() -> None:
    bot = LeagueOpponent(name="V5", type="bot_v5")
    assert bot.seat_token() == "bot_v5"


def test_league_opponent_seat_token_checkpoint() -> None:
    ckpt = LeagueOpponent(name="Iter10", type="nn_checkpoint", path="checkpoints/iter_10.pt")
    assert ckpt.seat_token() == "checkpoints/iter_10.pt"


def test_league_opponent_seat_token_checkpoint_no_path_raises() -> None:
    bad = LeagueOpponent(name="Bad", type="nn_checkpoint", path=None)
    with pytest.raises(ValueError, match="has no path"):
        bad.seat_token()


# ---------------------------------------------------------------------------
# LeaguePool initialisation
# ---------------------------------------------------------------------------


def test_league_pool_creates_entries_from_config() -> None:
    cfg = LeagueConfig(
        enabled=True,
        opponents=(
            LeagueOpponent(name="V5", type="bot_v5"),
            LeagueOpponent(name="M6", type="bot_m6"),
        ),
    )
    pool = LeaguePool(config=cfg)

    assert len(pool.entries) == 2
    assert pool.entries[0].opponent.name == "V5"
    assert pool.entries[0].elo == 1500.0


def test_league_pool_uses_opponent_initial_elo() -> None:
    cfg = LeagueConfig(
        enabled=True,
        opponents=(
            LeagueOpponent(name="StockSkis", type="bot_v1", initial_elo=800.0),
            LeagueOpponent(name="M6", type="bot_m6", initial_elo=2000.0),
        ),
    )
    pool = LeaguePool(config=cfg)

    assert [e.elo for e in pool.entries] == [800.0, 2000.0]


def test_league_pool_entry_outplace_rate_default_is_half() -> None:
    # With 0 games played, outplace_rate defaults to 0.5 (neutral prior).
    entry = LeaguePoolEntry(opponent=LeagueOpponent("A", "bot_v1"))
    assert entry.outplace_rate == 0.5


def test_league_pool_add_snapshot() -> None:
    pool = LeaguePool(config=LeagueConfig())
    pool.learner_elo = 1600.0
    pool.add_snapshot("snap1", "checkpoints/snap1.pt")

    assert len(pool.entries) == 1
    assert pool.entries[0].elo == 1600.0
    assert pool.entries[0].opponent.path == "checkpoints/snap1.pt"


def test_league_pool_save_and_restore_preserves_snapshot_elos(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "league_pool" / "iter_005.pt"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(b"checkpoint")

    cfg = LeagueConfig(
        enabled=True,
        opponents=(LeagueOpponent(name="Anchor", type="bot_v1", initial_elo=900.0),),
    )
    pool = LeaguePool(config=cfg)
    pool.learner_elo = 1325.0
    pool.entries[0].games_played = 8
    pool.entries[0].learner_outplaces = 5
    pool.add_snapshot("snapshot_iter_005", str(snapshot_path))
    pool.entries[1].elo = 1280.0
    pool.entries[1].games_played = 4
    pool.entries[1].learner_outplaces = 2

    state_path = tmp_path / "league_pool" / "state.json"
    persistence = JsonLeagueStatePersistence()
    persistence.save(pool, state_path)

    restored = LeaguePool(config=cfg)
    assert persistence.restore(restored, state_path) is True

    assert restored.learner_elo == pytest.approx(1325.0)
    assert len(restored.entries) == 2
    assert restored.entries[0].opponent.name == "Anchor"
    assert restored.entries[0].games_played == 8
    assert restored.entries[0].learner_outplaces == 5
    assert restored.entries[1].opponent.name == "snapshot_iter_005"
    assert restored.entries[1].elo == pytest.approx(1280.0)
    assert restored.entries[1].opponent.path == str(snapshot_path)


def test_league_pool_restore_skips_missing_snapshot_files(tmp_path: Path) -> None:
    cfg = LeagueConfig(enabled=True, opponents=(LeagueOpponent(name="Anchor", type="bot_v1"),))
    pool = LeaguePool(config=cfg)
    pool.add_snapshot("snapshot_iter_001", str(tmp_path / "missing.pt"))
    state_path = tmp_path / "league_pool" / "state.json"
    persistence = JsonLeagueStatePersistence()
    persistence.save(pool, state_path)

    restored = LeaguePool(config=cfg)
    persistence.restore(restored, state_path)

    assert [entry.opponent.name for entry in restored.entries] == ["Anchor"]


# ---------------------------------------------------------------------------
# LeaguePool.sampling_weights
# ---------------------------------------------------------------------------


def test_sampling_weights_uniform() -> None:
    cfg = LeagueConfig(sampling="uniform")
    pool = LeaguePool(config=cfg)
    pool.entries = [
        LeaguePoolEntry(opponent=LeagueOpponent("A", "bot_v1"), elo=1200.0),
        LeaguePoolEntry(opponent=LeagueOpponent("B", "bot_v5"), elo=1600.0),
    ]
    # uniform returns 1.0 per entry (not normalised to 0.5 — random.choices handles that)
    weights = pool.sampling_weights()
    assert weights == [1.0, 1.0]


def test_sampling_weights_hardest() -> None:
    cfg = LeagueConfig(sampling="hardest")
    pool = LeaguePool(config=cfg)
    pool.entries = [
        LeaguePoolEntry(opponent=LeagueOpponent("Weak", "bot_v1"), elo=1400.0),
        LeaguePoolEntry(opponent=LeagueOpponent("Strong", "bot_v5"), elo=1600.0),
        LeaguePoolEntry(opponent=LeagueOpponent("TiedStrong", "bot_v6"), elo=1600.0),
    ]
    weights = pool.sampling_weights()
    assert weights == [0.0, 1.0, 1.0]


def test_sampling_weights_pfsp_ordering() -> None:
    # Higher Elo → higher weight; weights normalise to 1.0.
    cfg = LeagueConfig(sampling="pfsp", pfsp_alpha=1.5)
    pool = LeaguePool(config=cfg)
    pool.entries = [
        LeaguePoolEntry(opponent=LeagueOpponent("Weak", "bot_v1"), elo=1100.0),
        LeaguePoolEntry(opponent=LeagueOpponent("Base", "bot_v5"), elo=1500.0),
        LeaguePoolEntry(opponent=LeagueOpponent("Boss", "bot_m6"), elo=1900.0),
    ]
    weights = pool.sampling_weights()

    assert weights[2] > weights[1] > weights[0]
    assert sum(weights) == pytest.approx(1.0)


def test_sampling_weights_pfsp_math() -> None:
    # Verify the exact formula: w_i = exp(alpha*(elo_i-1500)/400), then normalise.
    alpha = 1.5
    cfg = LeagueConfig(sampling="pfsp", pfsp_alpha=alpha)
    pool = LeaguePool(config=cfg)
    pool.entries = [
        LeaguePoolEntry(opponent=LeagueOpponent("A", "bot_v1"), elo=1300.0),
        LeaguePoolEntry(opponent=LeagueOpponent("B", "bot_v5"), elo=1700.0),
    ]
    raw = [math.exp(alpha * (e.elo - 1500.0) / 400.0) for e in pool.entries]
    expected = [w / sum(raw) for w in raw]

    weights = pool.sampling_weights()
    assert weights == pytest.approx(expected, rel=1e-6)


def test_sampling_weights_matchmaking_prefers_near_learner() -> None:
    cfg = LeagueConfig(sampling="matchmaking")
    pool = LeaguePool(config=cfg)
    pool.learner_elo = 1200.0
    pool.entries = [
        LeaguePoolEntry(opponent=LeagueOpponent("Near", "bot_v1"), elo=1200.0),
        LeaguePoolEntry(opponent=LeagueOpponent("Mid", "bot_v5"), elo=1400.0),
        LeaguePoolEntry(opponent=LeagueOpponent("Far", "bot_m6"), elo=2000.0),
    ]

    weights = pool.sampling_weights()
    assert weights[0] > weights[1] > weights[2]
    assert sum(weights) == pytest.approx(1.0)


def test_sampling_weights_matchmaking_math() -> None:
    cfg = LeagueConfig(sampling="matchmaking")
    pool = LeaguePool(config=cfg)
    pool.learner_elo = 1500.0
    pool.entries = [
        LeaguePoolEntry(opponent=LeagueOpponent("A", "bot_v1"), elo=1500.0),
        LeaguePoolEntry(opponent=LeagueOpponent("B", "bot_v5"), elo=1700.0),
    ]

    window = 200.0
    raw = [math.exp(-(((e.elo - pool.learner_elo) ** 2) / (2 * (window ** 2)))) for e in pool.entries]
    expected = [w / sum(raw) for w in raw]

    weights = pool.sampling_weights()
    assert weights == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# SampleLeagueSeats
# ---------------------------------------------------------------------------


def test_sample_league_seats_empty_pool() -> None:
    pool = LeaguePool(config=LeagueConfig())
    result = SampleLeagueSeats().execute(pool)
    assert result == "nn,nn,nn,nn"


@patch("training.use_cases.sample_league_seats.random.choices")
def test_sample_league_seats_basic(mock_choices) -> None:
    bot = LeaguePoolEntry(opponent=LeagueOpponent("V5", "bot_v5"))
    pool = LeaguePool(config=LeagueConfig(min_nn_per_game=1))
    pool.entries = [bot]
    mock_choices.return_value = [bot, bot, bot]

    result = SampleLeagueSeats().execute(pool)
    assert result == "nn,bot_v5,bot_v5,bot_v5"


@patch("training.use_cases.sample_league_seats.random.choices")
def test_sample_league_seats_enforces_min_nn(mock_choices) -> None:
    bot = LeaguePoolEntry(opponent=LeagueOpponent("V5", "bot_v5"))
    pool = LeaguePool(config=LeagueConfig(min_nn_per_game=3))
    pool.entries = [bot]
    # All 3 non-learner seats sampled as bot, but we need 2 more nn seats.
    mock_choices.return_value = [bot, bot, bot]

    result = SampleLeagueSeats().execute(pool)
    # Right-to-left replacement fills the deficit: seats 3 and 2 become nn.
    assert result == "nn,bot_v5,nn,nn"


@patch("training.use_cases.sample_league_seats.random.choices")
def test_sample_league_seats_nn_checkpoint_does_not_count_as_nn(mock_choices) -> None:
    # nn_checkpoint seat tokens are paths, not the literal string "nn".
    # The min_nn_per_game check strictly compares token == "nn", so checkpoints
    # do NOT count and will be overridden like any other non-nn token.
    ckpt = LeaguePoolEntry(
        opponent=LeagueOpponent("Old", "nn_checkpoint", "checkpoints/old.pt")
    )
    pool = LeaguePool(config=LeagueConfig(min_nn_per_game=2))
    pool.entries = [ckpt]
    mock_choices.return_value = [ckpt, ckpt, ckpt]

    result = SampleLeagueSeats().execute(pool)
    # min_nn_per_game=2 → need 1 extra nn beyond seat 0.  Rightmost becomes nn.
    assert result == "nn,checkpoints/old.pt,checkpoints/old.pt,nn"


@patch("training.use_cases.sample_league_seats.random.choices")
def test_sample_league_seats_no_override_when_quota_met(mock_choices) -> None:
    bot = LeaguePoolEntry(opponent=LeagueOpponent("V5", "bot_v5"))
    pool = LeaguePool(config=LeagueConfig(min_nn_per_game=1))
    pool.entries = [bot]
    mock_choices.return_value = [bot, bot, bot]

    result = SampleLeagueSeats().execute(pool)
    # Seat 0 already satisfies min_nn_per_game=1; no overrides needed.
    assert result == "nn,bot_v5,bot_v5,bot_v5"


# ---------------------------------------------------------------------------
# UpdateLeagueElo
# ---------------------------------------------------------------------------


def _elo_expected(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def test_update_league_elo_learner_wins_all() -> None:
    v5 = LeagueOpponent("V5", "bot_v5")
    pool = LeaguePool(config=LeagueConfig())
    pool.entries = [LeaguePoolEntry(opponent=v5, elo=1500.0)]
    pool.learner_elo = 1500.0

    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,bot_v5,nn,nn",
        seat_outcomes={1: (10, 0, 0)},
    )

    assert pool.learner_elo > 1500.0
    assert pool.entries[0].elo == 1500.0
    assert pool.entries[0].games_played == 10
    assert pool.entries[0].learner_outplaces == 10
    assert pool.entries[0].outplace_rate == 1.0


def test_update_league_elo_learner_loses_to_multiple_seats() -> None:
    v5 = LeagueOpponent("V5", "bot_v5")
    m6 = LeagueOpponent("M6", "bot_m6")
    pool = LeaguePool(config=LeagueConfig())
    pool.entries = [
        LeaguePoolEntry(opponent=v5, elo=1500.0),
        LeaguePoolEntry(opponent=m6, elo=1500.0),
    ]
    pool.learner_elo = 1500.0

    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,bot_v5,bot_m6,nn",
        seat_outcomes={1: (0, 5, 0), 2: (0, 10, 0)},
    )

    assert pool.entries[0].games_played == 5
    assert pool.entries[1].games_played == 10
    assert pool.entries[0].outplace_rate == 0.0
    assert pool.entries[1].outplace_rate == 0.0
    assert pool.entries[0].elo == 1500.0
    assert pool.entries[1].elo == 1500.0
    assert pool.learner_elo < 1500.0


def test_update_league_elo_ignores_nn_vs_nn_seats() -> None:
    pool = LeaguePool(config=LeagueConfig())
    pool.learner_elo = 1500.0

    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,nn,nn,nn",
        seat_outcomes={1: (10, 0, 0), 2: (0, 10, 0), 3: (5, 5, 0)},
    )

    # No pool entries → learner Elo unchanged.
    assert pool.learner_elo == 1500.0


def test_update_league_elo_draw_is_half_point() -> None:
    v5 = LeagueOpponent("V5", "bot_v5")
    pool = LeaguePool(config=LeagueConfig())
    pool.entries = [LeaguePoolEntry(opponent=v5, elo=1500.0)]
    pool.learner_elo = 1500.0

    # 0 wins, 0 losses, 10 draws — expected score is 0.5 for both sides.
    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,bot_v5,nn,nn",
        seat_outcomes={1: (0, 0, 10)},
    )

    # At equal Elo a draw is the expected outcome, so neither rating should change.
    assert pool.learner_elo == pytest.approx(1500.0, abs=1e-6)
    assert pool.entries[0].elo == pytest.approx(1500.0, abs=1e-6)


def test_update_league_elo_exact_k_factor() -> None:
    # Manual verification of one update step.
    # K=32, equal Elo → e_learner = 0.5.
    # Learner wins all 4 games → learner_outcome = 1.0.
    # delta = 32 * (1.0 - 0.5) = 16.0
    v5 = LeagueOpponent("V5", "bot_v5")
    pool = LeaguePool(config=LeagueConfig())
    pool.entries = [LeaguePoolEntry(opponent=v5, elo=1500.0)]
    pool.learner_elo = 1500.0

    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,bot_v5,nn,nn",
        seat_outcomes={1: (4, 0, 0)},
    )

    assert pool.learner_elo == pytest.approx(1516.0, abs=1e-4)
    assert pool.entries[0].elo == pytest.approx(1500.0, abs=1e-4)


def test_update_league_elo_uses_weighted_k_factor() -> None:
    # With elo_outplace_unit_weight=5, effective K is 32*5.
    v5 = LeagueOpponent("V5", "bot_v5")
    pool = LeaguePool(config=LeagueConfig(elo_outplace_unit_weight=5.0))
    pool.entries = [LeaguePoolEntry(opponent=v5, elo=1500.0)]
    pool.learner_elo = 1500.0

    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,bot_v5,nn,nn",
        seat_outcomes={1: (4, 0, 0)},
    )

    # learner_outcome=1.0, expected=0.5, delta = 32*5*(1-0.5)=80
    assert pool.learner_elo == pytest.approx(1580.0, abs=1e-4)


def test_update_league_elo_mixed_outcomes_vs_stronger_opponent() -> None:
    # 10 games versus 1700-Elo opponent, learner starts at 1500:
    # learner_outcome = (6 wins + 0.5*1 draw) / 10 = 0.65
    # expected(1500 vs 1700) ~= 0.240253...
    # delta = 32 * (0.65 - expected)
    v5 = LeagueOpponent("V5", "bot_v5")
    pool = LeaguePool(config=LeagueConfig())
    pool.entries = [LeaguePoolEntry(opponent=v5, elo=1700.0)]
    pool.learner_elo = 1500.0

    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,bot_v5,nn,nn",
        seat_outcomes={1: (6, 3, 1)},
    )

    expected_score = _elo_expected(1500.0, 1700.0)
    expected_delta = 32.0 * (0.65 - expected_score)

    assert pool.learner_elo == pytest.approx(1500.0 + expected_delta, abs=1e-6)
    assert pool.entries[0].games_played == 10
    assert pool.entries[0].learner_outplaces == 6


def test_update_league_elo_ignores_unknown_token() -> None:
    # A seat token not in the pool is silently skipped.
    pool = LeaguePool(config=LeagueConfig())
    pool.learner_elo = 1500.0

    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,bot_v9,nn,nn",
        seat_outcomes={1: (10, 0, 0)},
    )

    assert pool.learner_elo == 1500.0


def test_update_league_elo_zero_games_skipped() -> None:
    v5 = LeagueOpponent("V5", "bot_v5")
    pool = LeaguePool(config=LeagueConfig())
    pool.entries = [LeaguePoolEntry(opponent=v5, elo=1500.0)]
    pool.learner_elo = 1500.0

    UpdateLeagueElo().execute(
        pool,
        seat_config_used="nn,bot_v5,nn,nn",
        seat_outcomes={1: (0, 0, 0)},  # n_games == 0
    )

    assert pool.learner_elo == 1500.0
    assert pool.entries[0].elo == 1500.0
