"""Use case: run a single training iteration."""

from __future__ import annotations

import time
from pathlib import Path

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.ports.benchmark_port import BenchmarkPort
from training.ports.model_port import ModelPort
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort


class RunIteration:
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
        iteration: int,
        config: TrainingConfig,
        identity: ModelIdentity,
        ts_path: str,
        save_dir: Path,
        iter_lr: float | None = None,
    ) -> tuple[IterationResult, dict]:
        # Self-play
        self._presenter.on_selfplay_start(config)
        t0 = time.time()
        raw = self._selfplay.run(
            ts_path, config.games, config.seats, config.explore_rate, config.concurrency,
        )
        nn_seats = config.nn_seat_indices
        bot_seats = config.bot_seat_indices
        n_exps = len(raw["players"])
        sp_time = time.time() - t0
        self._presenter.on_selfplay_done(n_exps, sp_time)

        # PPO update
        if iter_lr is not None:
            self._ppo.set_lr(iter_lr)
        self._presenter.on_ppo_start(config, iter_lr=iter_lr)
        t0 = time.time()
        metrics, new_weights = self._ppo.update(raw, nn_seats, bot_seats)
        ppo_time = time.time() - t0
        self._presenter.on_ppo_done(metrics, ppo_time)

        # Export updated model
        self._model.export_for_inference(
            new_weights, identity.hidden_size, identity.oracle_critic, identity.model_arch, ts_path,
        )

        # Benchmark
        self._presenter.on_benchmark_start(config)
        t0 = time.time()
        placement = self._benchmark.measure_placement(
            ts_path, config.bench_games, config.effective_bench_seats,
            config.concurrency, session_size=50,
        )
        bench_time = time.time() - t0
        self._presenter.on_benchmark_done(placement, bench_time)

        # Save checkpoint
        ckpt_path = str(save_dir / f"iter_{iteration:03d}.pt")
        self._model.save_checkpoint(
            new_weights, identity.hidden_size, identity.oracle_critic, identity.model_arch,
            iteration, metrics["total_loss"], placement, ckpt_path,
        )

        result = IterationResult(
            iteration=iteration,
            placement=placement,
            loss=metrics["total_loss"],
            policy_loss=metrics["policy_loss"],
            value_loss=metrics["value_loss"],
            entropy=metrics["entropy"],
            n_experiences=n_exps,
            selfplay_time=sp_time,
            ppo_time=ppo_time,
            bench_time=bench_time,
        )
        return result, new_weights
