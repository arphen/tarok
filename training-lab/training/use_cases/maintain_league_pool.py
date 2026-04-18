"""Use case: maintain league Elo reporting and snapshot lifecycle per iteration."""

from __future__ import annotations

import shutil
from pathlib import Path

from training.entities.iteration_result import IterationResult
from training.entities.league import LeaguePool
from training.ports.presenter_port import PresenterPort
from training.use_cases.update_league_elo import UpdateLeagueElo


class MaintainLeaguePool:
    def __init__(self, updater: UpdateLeagueElo, presenter: PresenterPort):
        self._updater = updater
        self._presenter = presenter

    def initial_snapshot_elo(self, pool: LeaguePool) -> float | None:
        checkpoint_elos = [e.elo for e in pool.entries if e.opponent.type == "nn_checkpoint"]
        if checkpoint_elos:
            return max(checkpoint_elos)
        return None

    def execute(
        self,
        pool: LeaguePool,
        result: IterationResult,
        iteration: int,
        ts_path: str,
        league_pool_dir: Path,
        last_snapshot_elo: float | None,
    ) -> float | None:
        prev_elos = {e.opponent.name: e.elo for e in pool.entries}
        prev_learner_elo = pool.learner_elo
        self._updater.execute(pool, result.seat_config_used, result.seat_outcomes)
        elo_deltas = {
            e.opponent.name: e.elo - prev_elos.get(e.opponent.name, e.elo)
            for e in pool.entries
        }
        elo_deltas["__learner__"] = pool.learner_elo - prev_learner_elo
        self._presenter.on_league_elo_updated(pool, elo_deltas)

        if iteration % pool.config.snapshot_interval != 0:
            return last_snapshot_elo
        if pool.config.max_active_snapshots <= 0:
            return last_snapshot_elo
        if (
            last_snapshot_elo is not None
            and pool.learner_elo < last_snapshot_elo + pool.config.snapshot_elo_delta
        ):
            return last_snapshot_elo

        league_pool_dir.mkdir(parents=True, exist_ok=True)
        snap_path = str(league_pool_dir / f"iter_{iteration:03d}.pt")
        shutil.copy2(ts_path, snap_path)

        # Keep heuristic anchors and cap only nn_checkpoint snapshots.
        checkpoint_indices = [
            idx for idx, entry in enumerate(pool.entries)
            if entry.opponent.type == "nn_checkpoint"
        ]
        if len(checkpoint_indices) >= pool.config.max_active_snapshots:
            pool.entries.pop(checkpoint_indices[0])

        pool.add_snapshot(f"snapshot_iter_{iteration:03d}", snap_path)
        self._presenter.on_league_snapshot_added(iteration, snap_path)
        return pool.learner_elo
