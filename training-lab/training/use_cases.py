"""Use cases — application-level orchestration.

Depends only on entities and ports. Zero framework imports.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from training.entities import (
    IterationResult,
    ModelIdentity,
    TrainingConfig,
    TrainingRun,
    name_from_checkpoint,
    random_slovenian_name,
    scheduled_lr,
)
from training.ports import (
    BenchmarkPort,
    ConfigPort,
    ModelPort,
    PPOPort,
    PresenterPort,
    SelfPlayPort,
)


# ── Load / create model ────────────────────────────────────────────

class ResolveModel:
    def __init__(self, model: ModelPort):
        self._model = model

    def from_checkpoint(self, path: str) -> tuple[ModelIdentity, dict]:
        weights, hidden, oracle = self._model.load_weights(path)
        name = name_from_checkpoint(path) or random_slovenian_name()
        identity = ModelIdentity(name=name, hidden_size=hidden, oracle_critic=oracle, is_new=False)
        return identity, weights

    def from_scratch(self, hidden_size: int = 256, oracle: bool = True) -> tuple[ModelIdentity, dict]:
        name = random_slovenian_name()
        weights = self._model.create_new(hidden_size, oracle)
        identity = ModelIdentity(name=name, hidden_size=hidden_size, oracle_critic=oracle, is_new=True)
        return identity, weights


# ── Build config from CLI + YAML ────────────────────────────────────

class ResolveConfig:
    def __init__(self, config_loader: ConfigPort):
        self._loader = config_loader

    def resolve(self, cli: dict[str, Any], config_path: str | None) -> TrainingConfig:
        base: dict[str, Any] = {}
        if config_path:
            base = self._loader.load(config_path)

        merged = {**base, **{k: v for k, v in cli.items() if v is not None}}

        return TrainingConfig(
            seats=merged.get("seats", "nn,bot_v5,bot_v5,bot_v5"),
            bench_seats=merged.get("bench_seats"),
            iterations=merged.get("iterations", 10),
            games=merged.get("games", 10_000),
            bench_games=merged.get("bench_games", 10_000),
            ppo_epochs=merged.get("ppo_epochs", 6),
            batch_size=merged.get("batch_size", 8192),
            lr=merged.get("lr", 3e-4),
            lr_schedule=merged.get("lr_schedule", "constant"),
            lr_min=merged.get("lr_min"),
            explore_rate=merged.get("explore_rate", 0.10),
            device=merged.get("device", "auto"),
            save_dir=merged.get("save_dir", "checkpoints/training_run"),
            concurrency=merged.get("concurrency", 128),
        )


# ── Single iteration ───────────────────────────────────────────────

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
        n_exps = sum(1 for p in raw["players"] if int(p) == 0)
        sp_time = time.time() - t0
        self._presenter.on_selfplay_done(n_exps, sp_time)

        # PPO update
        if iter_lr is not None:
            self._ppo.set_lr(iter_lr)
        self._presenter.on_ppo_start(config, iter_lr=iter_lr)
        t0 = time.time()
        metrics, new_weights = self._ppo.update(raw)
        ppo_time = time.time() - t0
        self._presenter.on_ppo_done(metrics, ppo_time)

        # Export updated model
        self._model.export_for_inference(
            new_weights, identity.hidden_size, identity.oracle_critic, ts_path,
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
            new_weights, identity.hidden_size, identity.oracle_critic,
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


# ── Full training loop ──────────────────────────────────────────────

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
