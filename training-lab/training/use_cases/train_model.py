"""Use case: full training loop."""

from __future__ import annotations

import time
from pathlib import Path

from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig, scheduled_lr
from training.entities.training_run import TrainingRun
from training.ports.benchmark_port import BenchmarkPort
from training.ports.model_port import ModelPort
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.run_iteration import RunIteration


class TrainModel:
    def __init__(
        self,
        selfplay: SelfPlayPort,
        ppo: PPOPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
    ):
        self._selfplay = selfplay
        self._ppo = ppo
        self._benchmark = benchmark
        self._model = model
        self._presenter = presenter

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
            weights, identity.hidden_size, identity.oracle_critic, ts_path,
        )
        self._presenter.on_model_loaded(identity, str(save_dir))
        self._presenter.on_device_selected(device)

        # Setup PPO
        self._ppo.setup(weights, config, device)

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

        run_iteration = RunIteration(
            self._selfplay, self._ppo, self._benchmark, self._model, self._presenter,
        )

        current_weights = weights
        for i in range(1, config.iterations + 1):
            elapsed = time.time() - run.start_time
            self._presenter.on_iteration_start(i, config.iterations, elapsed)

            iter_lr = scheduled_lr(
                i - 1, config.iterations,
                config.lr, config.effective_lr_min, config.lr_schedule,
            )

            prev = run.placements[-1]
            result, current_weights = run_iteration.execute(
                i, config, identity, ts_path, save_dir, iter_lr=iter_lr,
            )
            run.results.append(result)
            self._presenter.on_iteration_done(prev, result.placement, result.total_time)

        run.end_time = time.time()

        # Copy best checkpoint
        if run.improved:
            best_src = str(save_dir / f"iter_{run.best_iteration:03d}.pt")
            best_dst = str(save_dir / "best.pt")
            self._model.copy_best(best_src, best_dst)

        self._presenter.on_training_complete(run)
        return run
