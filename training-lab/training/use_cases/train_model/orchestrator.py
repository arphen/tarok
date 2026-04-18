"""Use case: full training loop orchestration."""

from __future__ import annotations

import shutil
import time
from dataclasses import replace as dc_replace
from pathlib import Path

from training.entities.league import LeaguePool
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.entities.training_run import TrainingRun
from training.ports.benchmark_port import BenchmarkPort
from training.ports.imitation_coef_policy_port import ImitationCoefPolicyPort
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.league_persistence_port import LeagueStatePersistencePort
from training.ports.learning_rate_policy_port import LearningRatePolicyPort
from training.ports.model_port import ModelPort
from training.ports.presenter_port import PresenterPort
from training.use_cases.maintain_league_pool import MaintainLeaguePool
from training.use_cases.sample_league_seats import SampleLeagueSeats
from training.use_cases.train_model.policies import (
    DefaultEntropyCoefPolicy,
    DefaultImitationCoefPolicy,
    DefaultLearningRatePolicy,
    EloDecayEntropyPolicy,
)
from training.use_cases.update_league_elo import UpdateLeagueElo


class TrainModel:
    def __init__(
        self,
        iteration_runner: IterationRunnerPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
        lr_policy: LearningRatePolicyPort | None = None,
        imitation_policy: ImitationCoefPolicyPort | None = None,
        entropy_policy: DefaultEntropyCoefPolicy | EloDecayEntropyPolicy | None = None,
        league_persistence: LeagueStatePersistencePort | None = None,
    ):
        self._iteration_runner = iteration_runner
        self._benchmark = benchmark
        self._model = model
        self._presenter = presenter
        self._lr_policy = lr_policy if lr_policy is not None else DefaultLearningRatePolicy()
        self._imitation_policy = (
            imitation_policy if imitation_policy is not None else DefaultImitationCoefPolicy()
        )
        self._entropy_policy = entropy_policy if entropy_policy is not None else DefaultEntropyCoefPolicy()
        if league_persistence is None:
            from training.adapters.league_persistence import JsonLeagueStatePersistence
            league_persistence = JsonLeagueStatePersistence()
        self._league_persistence: LeagueStatePersistencePort = league_persistence

    def execute(
        self,
        config: TrainingConfig,
        identity: ModelIdentity,
        weights: dict,
        device: str,
    ) -> TrainingRun:
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts_path = str(save_dir / "_current.pt")

        self._model.export_for_inference(
            weights, identity.hidden_size, identity.oracle_critic, identity.model_arch, ts_path,
        )
        self._presenter.on_model_loaded(identity, str(save_dir))
        self._presenter.on_device_selected(device)

        # Setup iteration runner (PPO/setup ownership lives at adapter layer)
        self._iteration_runner.setup(weights, config, device)

        # Initial benchmark
        self._presenter.on_training_plan(config)
        t0 = time.time()
        initial = self._benchmark.measure_placement(
            ts_path, config.bench_games, config.effective_bench_seats,
            config.concurrency, session_size=50,
        )
        self._presenter.on_initial_benchmark(
            initial, config.bench_games, config.effective_bench_seats, time.time() - t0,
        )

        self._presenter.on_training_loop_start(config)

        run = TrainingRun(
            config=config,
            identity=identity,
            initial_placement=initial,
            start_time=time.time(),
        )

        if config.league is None or not config.league.enabled:
            raise ValueError("TrainModel requires league.enabled=true")

        pool = LeaguePool(config=config.league)
        league_maintenance = MaintainLeaguePool(
            updater=UpdateLeagueElo(),
            presenter=self._presenter,
            persistence=self._league_persistence,
        )
        league_pool_dir = save_dir / "league_pool"
        self._league_persistence.restore(pool, league_maintenance.state_path(league_pool_dir))
        last_snapshot_elo: float | None = league_maintenance.initial_snapshot_elo(pool)

        sample_seats = SampleLeagueSeats()

        try:
            for i in range(1, config.iterations + 1):
                elapsed = time.time() - run.start_time
                self._presenter.on_iteration_start(i, config.iterations, elapsed)

                iter_lr = self._lr_policy.compute(
                    config=config,
                    iteration=i,
                    learner_elo=pool.learner_elo,
                )
                iter_imitation_coef = self._imitation_policy.compute(
                    config=config, iteration=i, learner_elo=pool.learner_elo,
                )
                iter_entropy_coef = self._entropy_policy.compute(
                    config=config, iteration=i, learner_elo=pool.learner_elo,
                )

                seats_override = sample_seats.execute(pool)

                prev = run.placements[-1]
                should_bench = config.should_benchmark_iteration(i)
                result = self._iteration_runner.run_iteration(
                    i, config, identity, ts_path, save_dir,
                    prev_placement=prev,
                    iter_lr=iter_lr,
                    iter_imitation_coef=iter_imitation_coef,
                    iter_entropy_coef=iter_entropy_coef,
                    seats_override=seats_override,
                    run_benchmark=should_bench,
                )
                run.results.append(result)
                self._presenter.on_iteration_done(prev, result.placement, result.total_time)

                result = dc_replace(result, learner_elo=pool.learner_elo)
                run.results[-1] = result

                last_snapshot_elo = league_maintenance.execute(
                    pool=pool,
                    result=result,
                    iteration=i,
                    ts_path=ts_path,
                    league_pool_dir=league_pool_dir,
                    last_snapshot_elo=last_snapshot_elo,
                )
        finally:
            self._iteration_runner.teardown()

        run.end_time = time.time()

        # Copy best checkpoint
        if run.results:
            if config.best_model_metric == "loss":
                best_iter = run.best_loss_iteration
            elif config.best_model_metric == "elo":
                best_iter = run.best_elo_iteration
            else:
                best_iter = run.best_iteration
            best_src = str(save_dir / f"iter_{best_iter:03d}.pt")
            best_dst = str(save_dir / "best.pt")
            self._model.copy_best(best_src, best_dst)

        self._presenter.on_training_complete(run)
        return run
