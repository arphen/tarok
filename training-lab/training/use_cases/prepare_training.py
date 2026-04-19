"""Use case: bootstrap everything required before the training loop starts."""

from __future__ import annotations

import time
from pathlib import Path

from training.entities.league import LeaguePool
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.entities.training_context import TrainingContext
from training.entities.training_run import TrainingRun
from training.ports.benchmark_port import BenchmarkPort
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.league_persistence_port import LeagueStatePersistencePort
from training.ports.model_port import ModelPort
from training.ports.presenter_port import PresenterPort
class PrepareTraining:
    """Set up the training context before the iteration loop begins.

    Single responsibility: validate preconditions, initialise infrastructure,
    run the initial benchmark, and restore persisted league state — then return
    a fully populated TrainingContext ready for AdvanceIteration.
    """

    def __init__(
        self,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
        iteration_runner: IterationRunnerPort,
        league_persistence: LeagueStatePersistencePort,
    ) -> None:
        self._benchmark = benchmark
        self._model = model
        self._presenter = presenter
        self._iteration_runner = iteration_runner
        self._league_persistence = league_persistence

    def execute(
        self,
        config: TrainingConfig,
        identity: ModelIdentity,
        weights: dict,
        device: str,
    ) -> TrainingContext:
        if config.league is None or not config.league.enabled:
            raise ValueError("TrainModel requires league.enabled=true")

        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts_path = str(save_dir / "_current.pt")

        self._model.export_for_inference(
            weights, identity.hidden_size, identity.oracle_critic, identity.model_arch, ts_path,
        )
        self._presenter.on_model_loaded(identity, str(save_dir))
        self._presenter.on_device_selected(device)

        self._iteration_runner.setup(weights, config, device)

        self._presenter.on_training_plan(config)
        t0 = time.time()
        initial = self._benchmark.measure_placement(
            ts_path, config.bench_games, config.effective_bench_seats,
            config.concurrency, session_size=config.outplace_session_size,
            lapajne_mc_worlds=config.lapajne_mc_worlds,
            lapajne_mc_sims=config.lapajne_mc_sims,
        )
        self._presenter.on_initial_benchmark(
            initial, config.bench_games, config.effective_bench_seats, time.time() - t0,
        )

        run = TrainingRun(
            config=config,
            identity=identity,
            initial_placement=initial,
            start_time=time.time(),
        )

        pool = LeaguePool(config=config.league)
        league_pool_dir = save_dir / "league_pool"
        state_path = league_pool_dir / "state.json"
        self._league_persistence.restore(pool, state_path)
        checkpoint_elos = [e.elo for e in pool.entries if e.opponent.type == "nn_checkpoint"]
        last_snapshot_elo = max(checkpoint_elos) if checkpoint_elos else None

        return TrainingContext(
            run=run,
            pool=pool,
            last_snapshot_elo=last_snapshot_elo,
            ts_path=ts_path,
            save_dir=save_dir,
            league_pool_dir=league_pool_dir,
        )
