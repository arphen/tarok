"""Use case: maintain league Elo reporting and snapshot lifecycle per iteration."""

from __future__ import annotations

import shutil
from pathlib import Path

from training.entities.iteration_result import IterationResult
from training.entities.league import LeaguePool
from training.ports.league_calibration_port import LeagueCalibrationPort
from training.ports.league_persistence_port import LeagueStatePersistencePort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.update_league_elo import UpdateLeagueElo


class MaintainLeaguePool:
    def __init__(
        self,
        updater: UpdateLeagueElo,
        presenter: PresenterPort,
        persistence: LeagueStatePersistencePort,
        selfplay: SelfPlayPort | None = None,
        league_calibration: LeagueCalibrationPort | None = None,
    ):
        self._updater = updater
        self._presenter = presenter
        self._persistence = persistence
        self._selfplay = selfplay
        self._league_calibration = league_calibration

    @staticmethod
    def state_path(league_pool_dir: Path) -> Path:
        return league_pool_dir / "state.json"

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
        concurrency: int,
        session_size: int,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
    ) -> float | None:
        prev_elos = {e.opponent.name: e.elo for e in pool.entries}
        prev_learner_elo = pool.learner_elo
        self._updater.execute(
            pool,
            result.seat_config_used,
            result.seat_outcomes,
            opponent_outcomes=result.opponent_outcomes,
        )
        elo_deltas = {
            e.opponent.name: e.elo - prev_elos.get(e.opponent.name, e.elo)
            for e in pool.entries
        }
        elo_deltas["__learner__"] = pool.learner_elo - prev_learner_elo
        self._presenter.on_league_elo_updated(pool, elo_deltas)
        self._persistence.save(pool, self.state_path(league_pool_dir))

        if iteration % pool.config.snapshot_interval != 0:
            return last_snapshot_elo
        if pool.config.max_active_snapshots <= 0:
            return last_snapshot_elo

        candidate_elo = self._calibrate_candidate_elo(
            pool=pool,
            model_path=ts_path,
            concurrency=concurrency,
            session_size=session_size,
            lapajne_mc_worlds=lapajne_mc_worlds,
            lapajne_mc_sims=lapajne_mc_sims,
        )
        if candidate_elo is None:
            return last_snapshot_elo

        checkpoint_elos = [
            entry.elo
            for entry in pool.entries
            if entry.opponent.type == "nn_checkpoint"
        ]
        if checkpoint_elos and candidate_elo < max(checkpoint_elos) + pool.config.snapshot_elo_delta:
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
            weakest_idx = min(checkpoint_indices, key=lambda idx: pool.entries[idx].elo)
            pool.entries.pop(weakest_idx)

        pool.add_snapshot(f"ghost@{iteration}", snap_path, initial_elo=candidate_elo)
        self._persistence.save(pool, self.state_path(league_pool_dir))
        self._presenter.on_league_snapshot_added(iteration, snap_path)
        return candidate_elo

    def _calibrate_candidate_elo(
        self,
        pool: LeaguePool,
        model_path: str,
        concurrency: int,
        session_size: int,
        lapajne_mc_worlds: int | None,
        lapajne_mc_sims: int | None,
    ) -> float | None:
        if self._league_calibration is None:
            # Back-compat: tests/wiring that pre-date the port simply keep the
            # current learner Elo as the candidate's implied Elo.
            return pool.learner_elo

        return self._league_calibration.calibrate_candidate(
            pool=pool,
            model_path=model_path,
            concurrency=concurrency,
            session_size=session_size,
            n_games_per_opponent=max(1, pool.config.elo_eval_games),
            lapajne_mc_worlds=lapajne_mc_worlds,
            lapajne_mc_sims=lapajne_mc_sims,
        )
