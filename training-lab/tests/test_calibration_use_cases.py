"""Tests for league calibration use cases."""

from __future__ import annotations

from training.entities.league import LeagueConfig, LeagueOpponent, LeaguePool
from training.use_cases.calibrate_initial_league_elo import (
    CalibrateInitialLeagueElo,
    _avg_placements as initial_avg_placements,
    _pick_three_opponents,
)
from training.use_cases.calibrate_snapshot_elo import (
    CalibrateSnapshotElo,
    _avg_placements as snapshot_avg_placements,
)


class _FakeSelfPlay:
    def __init__(self, by_seat0: dict[str, list[list[float]]]) -> None:
        self._by_seat0 = by_seat0
        self.calls: list[str] = []

    def run(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        explore_rate: float,
        concurrency: int,
        include_replay_data: bool = False,
        include_oracle_states: bool = False,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        variant: int = 0,
    ) -> dict[str, list[list[float]]]:
        del model_path
        del n_games
        del explore_rate
        del concurrency
        del include_replay_data
        del include_oracle_states
        del lapajne_mc_worlds
        del lapajne_mc_sims
        del variant
        self.calls.append(seat_config)
        seat0 = seat_config.split(",")[0]
        return {"scores": self._by_seat0.get(seat0, [])}


def test_pick_three_opponents_cycles_tokens() -> None:
    assert _pick_three_opponents([]) == ("nn", "nn", "nn")
    assert _pick_three_opponents(["bot_v1"]) == ("bot_v1", "bot_v1", "bot_v1")
    assert _pick_three_opponents(["a", "b"]) == ("a", "b", "a")


def test_avg_placements_session_rollup_for_both_calibrators() -> None:
    # 4 games rolled up into 2 sessions. Seat 0 wins session 1, loses session 2.
    scores = [
        [10, 0, -1, -2],
        [11, 0, -1, -2],
        [0, 10, 1, -1],
        [0, 10, 1, -1],
    ]
    got_initial = initial_avg_placements(scores, session_size=2)
    got_snapshot = snapshot_avg_placements(scores, session_size=2)
    assert got_initial == got_snapshot
    assert got_initial is not None
    assert got_initial[0] == 2.0


def test_initial_calibration_sets_anchor_and_learner_relative_elo() -> None:
    model_path = "data/checkpoints/hall_of_fame/eva_golob.pt"
    pool = LeaguePool(
        config=LeagueConfig(
            enabled=True,
            opponents=(
                LeagueOpponent(name="V3", type="bot_v3"),
                LeagueOpponent(name="V5", type="bot_v5"),
                LeagueOpponent(name="M6", type="bot_m6"),
            ),
        )
    )
    selfplay = _FakeSelfPlay(
        {
            "bot_v3": [[10, 0, -1, -2]],  # place 1
            "bot_v5": [[0, 10, -1, -2]],  # place 2
            "bot_m6": [[0, 1, 10, -2]],  # place 3
            model_path: [[-2, -1, 0, 10]],  # learner place 4
        }
    )

    mixed: list[tuple[str, tuple[float, float, float, float]]] = []
    done = CalibrateInitialLeagueElo().execute(
        pool=pool,
        selfplay=selfplay,  # type: ignore[arg-type]
        model_path=model_path,
        n_games_per_pair=100,
        concurrency=8,
        session_size=1,
        anchor_name="V3",
        anchor_elo=1500.0,
        on_mixed_result=lambda _i, _n, name, _opp, placements: mixed.append((name, placements)),
    )

    assert done is True
    by_name = {e.opponent.name: e.elo for e in pool.entries}
    assert by_name["V3"] == 1500.0
    assert by_name["V5"] == 1250.0
    assert by_name["M6"] == 1000.0
    assert pool.learner_elo == 650.0  # learner implied 750, then -100 per policy
    assert len(mixed) == 4  # includes temporary learner matchup


def test_initial_calibration_anchor_missing_falls_back_to_anchor_elo() -> None:
    model_path = "data/checkpoints/hall_of_fame/eva_golob.pt"
    pool = LeaguePool(
        config=LeagueConfig(
            enabled=True,
            opponents=(
                LeagueOpponent(name="V3", type="bot_v3"),
                LeagueOpponent(name="V5", type="bot_v5"),
            ),
        )
    )
    selfplay = _FakeSelfPlay(
        {
            "bot_v5": [[0, 10, -1, -2]],
            model_path: [[0, 1, 10, -2]],
            # no data for V3 => anchor placement is missing
        }
    )

    done = CalibrateInitialLeagueElo().execute(
        pool=pool,
        selfplay=selfplay,  # type: ignore[arg-type]
        model_path=model_path,
        n_games_per_pair=100,
        concurrency=8,
        session_size=1,
        anchor_name="V3",
        anchor_elo=1500.0,
    )

    assert done is True
    assert [e.elo for e in pool.entries] == [1500.0, 1500.0]
    assert pool.learner_elo == 1400.0


def test_snapshot_calibration_uses_median_implied_elo() -> None:
    snapshot = LeagueOpponent(name="snapshot_010", type="nn_checkpoint", path="/tmp/snap.pt")
    pool = LeaguePool(
        config=LeagueConfig(
            enabled=True,
            opponents=(
                snapshot,
                LeagueOpponent(name="A", type="bot_v1", initial_elo=1400.0),
                LeagueOpponent(name="B", type="bot_v3", initial_elo=1600.0),
                LeagueOpponent(name="C", type="bot_m6", initial_elo=1800.0),
            ),
        )
    )
    selfplay = _FakeSelfPlay(
        {
            "/tmp/snap.pt": [[10, 0, -1, -2]],
        }
    )

    # Different target seat(1) outcomes by target token.
    per_target_scores = {
        "bot_v1": [[10, 0, -1, -2]],  # snapshot 1, target 2 => 1650
        "bot_v3": [[0, 10, -1, -2]],  # snapshot 2, target 1 => 1350
        "bot_m6": [[10, 0, 1, -1]],  # snapshot 1, target 3 => 2300
    }

    def _run_with_target_split(*args, **kwargs):
        seat_config = kwargs["seat_config"]
        selfplay.calls.append(seat_config)
        target_tok = seat_config.split(",")[1]
        return {"scores": per_target_scores[target_tok]}

    selfplay.run = _run_with_target_split  # type: ignore[method-assign]

    setups: list[str] = []
    pairs: list[tuple[str, str, float]] = []
    done = CalibrateSnapshotElo().execute(
        pool=pool,
        snapshot_name="snapshot_010",
        selfplay=selfplay,  # type: ignore[arg-type]
        model_path="/tmp/model.pt",
        n_games_per_opponent=200,
        concurrency=8,
        session_size=1,
        on_match_setup=lambda _snap, seat_cfg, _n: setups.append(seat_cfg),
        on_pair_result=lambda snap, target, _w, _l, _d, _place, implied, _target_elo: pairs.append((snap, target, implied)),
    )

    assert done is True
    snap_entry = next(e for e in pool.entries if e.opponent.name == "snapshot_010")
    assert snap_entry.elo == 1650.0
    assert len(setups) == 3
    assert [p[1] for p in pairs] == ["A", "B", "C"]


def test_snapshot_calibration_returns_false_for_invalid_inputs() -> None:
    pool = LeaguePool(config=LeagueConfig(enabled=True, opponents=()))
    selfplay = _FakeSelfPlay({})

    assert CalibrateSnapshotElo().execute(
        pool=pool,
        snapshot_name="missing",
        selfplay=selfplay,  # type: ignore[arg-type]
        model_path="/tmp/model.pt",
        n_games_per_opponent=10,
        concurrency=1,
        session_size=1,
    ) is False

    assert CalibrateSnapshotElo().execute(
        pool=pool,
        snapshot_name="missing",
        selfplay=selfplay,  # type: ignore[arg-type]
        model_path="/tmp/model.pt",
        n_games_per_opponent=0,
        concurrency=1,
        session_size=1,
    ) is False