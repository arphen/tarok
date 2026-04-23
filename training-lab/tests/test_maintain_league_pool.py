from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from training.entities.iteration_result import IterationResult
from training.entities.league import LeagueConfig, LeagueOpponent, LeaguePool, LeaguePoolEntry
from training.use_cases.maintain_league_pool import MaintainLeaguePool
from training.use_cases.update_league_elo import UpdateLeagueElo


class _FakeSelfPlay:
    def __init__(self, scores: list[list[float]]) -> None:
        self.scores = scores
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
        centaur_handoff_trick: int | None = None,
        centaur_pimc_worlds: int | None = None,
        centaur_endgame_solver: str | None = None,
        centaur_alpha_mu_depth: int | None = None,
        centaur_deterministic_seed: int | None = None,
    ) -> dict[str, list[list[float]]]:
        del model_path
        del n_games
        del explore_rate
        del concurrency
        del include_replay_data
        del include_oracle_states
        del lapajne_mc_worlds
        del lapajne_mc_sims
        del centaur_handoff_trick
        del centaur_pimc_worlds
        del centaur_endgame_solver
        del centaur_alpha_mu_depth
        del centaur_deterministic_seed
        self.calls.append(seat_config)
        return {"scores": self.scores}


def _result() -> IterationResult:
    return IterationResult(
        iteration=1,
        placement=2.0,
        loss=0.1,
        policy_loss=0.1,
        value_loss=0.1,
        entropy=0.1,
        n_experiences=1,
        selfplay_time=0.0,
        ppo_time=0.0,
        bench_time=0.0,
        seat_config_used="nn,bot_v1,nn,nn",
        seat_outcomes={1: (1, 0, 0)},
    )


def test_maintain_league_pool_admits_calibrated_snapshot_and_evicts_weakest(tmp_path: Path) -> None:
    cfg = LeagueConfig(enabled=True, snapshot_interval=1, snapshot_elo_delta=50.0, max_active_snapshots=2)
    pool = LeaguePool(config=cfg)
    pool.entries = [
        LeaguePoolEntry(opponent=LeagueOpponent("Anchor", "bot_v1"), elo=1000.0),
        LeaguePoolEntry(
            opponent=LeagueOpponent("old-low", "nn_checkpoint", path=str(tmp_path / "old-low.pt")),
            elo=1200.0,
        ),
        LeaguePoolEntry(
            opponent=LeagueOpponent("old-high", "nn_checkpoint", path=str(tmp_path / "old-high.pt")),
            elo=1300.0,
        ),
    ]
    pool.learner_elo = 1250.0

    ts_path = tmp_path / "current.pt"
    ts_path.write_bytes(b"x")

    presenter = MagicMock()
    persistence = MagicMock()
    # seat0 wins -> implied candidate elo is target_elo + 250 for each opponent
    selfplay = _FakeSelfPlay(scores=[[10, 0, -1, -2]])
    use_case = MaintainLeaguePool(
        updater=UpdateLeagueElo(),
        presenter=presenter,
        persistence=persistence,
        selfplay=selfplay,
    )

    out = use_case.execute(
        pool=pool,
        result=_result(),
        iteration=7,
        ts_path=str(ts_path),
        league_pool_dir=tmp_path / "league_pool",
        last_snapshot_elo=None,
        concurrency=8,
        session_size=1,
    )

    names = [e.opponent.name for e in pool.entries]
    assert "old-low" not in names
    assert "old-high" in names
    assert "ghost@7" in names

    ghost = next(e for e in pool.entries if e.opponent.name == "ghost@7")
    assert ghost.elo == 1450.0
    assert out == 1450.0
    assert len(selfplay.calls) == 3


def test_maintain_league_pool_rejects_candidate_below_checkpoint_margin(tmp_path: Path) -> None:
    cfg = LeagueConfig(enabled=True, snapshot_interval=1, snapshot_elo_delta=50.0, max_active_snapshots=2)
    pool = LeaguePool(config=cfg)
    pool.entries = [
        LeaguePoolEntry(opponent=LeagueOpponent("Anchor", "bot_v1"), elo=1000.0),
        LeaguePoolEntry(
            opponent=LeagueOpponent("old-high", "nn_checkpoint", path=str(tmp_path / "old-high.pt")),
            elo=1300.0,
        ),
    ]

    ts_path = tmp_path / "current.pt"
    ts_path.write_bytes(b"x")

    presenter = MagicMock()
    persistence = MagicMock()
    # learner place 2 vs seat1 place 1 -> implied = target_elo - 250 (too low)
    selfplay = _FakeSelfPlay(scores=[[0, 10, -1, -2]])
    use_case = MaintainLeaguePool(
        updater=UpdateLeagueElo(),
        presenter=presenter,
        persistence=persistence,
        selfplay=selfplay,
    )

    out = use_case.execute(
        pool=pool,
        result=_result(),
        iteration=8,
        ts_path=str(ts_path),
        league_pool_dir=tmp_path / "league_pool",
        last_snapshot_elo=1300.0,
        concurrency=8,
        session_size=1,
    )

    assert [e.opponent.name for e in pool.entries] == ["Anchor", "old-high"]
    assert out == 1300.0
    assert len(selfplay.calls) == 2
