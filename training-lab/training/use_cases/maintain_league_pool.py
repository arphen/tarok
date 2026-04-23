"""Use case: maintain league Elo reporting and snapshot lifecycle per iteration."""

from __future__ import annotations

import shutil
from pathlib import Path
from itertools import cycle

from training.entities.iteration_result import IterationResult
from training.entities.league import LeaguePool
from training.ports.league_persistence_port import LeagueStatePersistencePort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.league_calibration_utils import _ELO_PER_PLACEMENT, _avg_placements
from training.use_cases.update_league_elo import UpdateLeagueElo


class MaintainLeaguePool:
    def __init__(
        self,
        updater: UpdateLeagueElo,
        presenter: PresenterPort,
        persistence: LeagueStatePersistencePort,
        selfplay: SelfPlayPort | None = None,
    ):
        self._updater = updater
        self._presenter = presenter
        self._persistence = persistence
        self._selfplay = selfplay

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
        if self._selfplay is None:
            # Keep compatibility in tests/wiring that do not provide selfplay.
            return pool.learner_elo

        opponents = list(pool.entries)
        if not opponents:
            return None

        n_games = max(1, pool.config.elo_eval_games)
        implied_elos: list[float] = []

        for target_idx, target in enumerate(opponents):
            filler_indices = [i for i in range(len(opponents)) if i != target_idx]
            if not filler_indices:
                filler_indices = [target_idx]
            cyc = cycle(filler_indices)
            f1 = opponents[next(cyc)]
            f2 = opponents[next(cyc)]

            seat_config = (
                f"{model_path},"
                f"{target.opponent.seat_token()},"
                f"{f1.opponent.seat_token()},"
                f"{f2.opponent.seat_token()}"
            )

            raw = self._selfplay.run(
                model_path=model_path,
                n_games=n_games,
                seat_config=seat_config,
                explore_rate=0.0,
                concurrency=concurrency,
                include_replay_data=False,
                include_oracle_states=False,
                lapajne_mc_worlds=lapajne_mc_worlds,
                lapajne_mc_sims=lapajne_mc_sims,
            )
            scores = raw.get("scores")
            places = _avg_placements(scores, session_size=session_size)
            if places is None:
                continue

            learner_place = places[0]
            target_place = places[1]
            implied = target.elo + _ELO_PER_PLACEMENT * (target_place - learner_place)
            implied_elos.append(implied)

        if not implied_elos:
            return None

        implied_elos.sort()
        mid = len(implied_elos) // 2
        if len(implied_elos) % 2 == 0:
            return (implied_elos[mid - 1] + implied_elos[mid]) / 2.0
        return implied_elos[mid]
