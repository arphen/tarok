"""Use case: full training loop."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import torch

from training.entities.league import LeaguePool
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig, scheduled_lr
from training.entities.training_run import TrainingRun
from training.ports.benchmark_port import BenchmarkPort
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.model_port import ModelPort
from training.ports.presenter_port import PresenterPort
from training.use_cases.sample_league_seats import SampleLeagueSeats
from training.use_cases.update_league_elo import UpdateLeagueElo


class TrainModel:
    def __init__(
        self,
        iteration_runner: IterationRunnerPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
    ):
        self._iteration_runner = iteration_runner
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

        # Build league pool (empty if league not enabled)
        pool: LeaguePool | None = None
        if config.league is not None and config.league.enabled:
            pool = LeaguePool(config=config.league)

        _sample_seats = SampleLeagueSeats()
        _update_elo = UpdateLeagueElo()
        league_pool_dir = save_dir / "league_pool"

        if config.memory_telemetry and not tracemalloc.is_tracing():
            tracemalloc.start(10)
        prev_mem_stats: dict[str, float] | None = None

        try:
            for i in range(1, config.iterations + 1):
                elapsed = time.time() - run.start_time
                self._presenter.on_iteration_start(i, config.iterations, elapsed)

                iter_lr = scheduled_lr(
                    i - 1, config.iterations,
                    config.lr, config.effective_lr_min, config.lr_schedule,
                )
                iter_imitation_coef = config.scheduled_imitation_coef(i)

                seats_override: str | None = None
                if pool is not None:
                    seats_override = _sample_seats.execute(pool)

                prev = run.placements[-1]
                should_bench = config.should_benchmark_iteration(i)
                result = self._iteration_runner.run_iteration(
                    i, config, identity, ts_path, save_dir,
                    prev_placement=prev,
                    iter_lr=iter_lr,
                    iter_imitation_coef=iter_imitation_coef,
                    seats_override=seats_override,
                    run_benchmark=should_bench,
                )
                run.results.append(result)
                self._presenter.on_iteration_done(prev, result.placement, result.total_time)

                if pool is not None:
                    prev_elos = {e.opponent.name: e.elo for e in pool.entries}
                    prev_learner_elo = pool.learner_elo
                    _update_elo.execute(pool, result.seat_config_used, result.seat_outcomes)
                    elo_deltas = {
                        e.opponent.name: e.elo - prev_elos.get(e.opponent.name, e.elo)
                        for e in pool.entries
                    }
                    elo_deltas["__learner__"] = pool.learner_elo - prev_learner_elo
                    self._presenter.on_league_elo_updated(pool, elo_deltas)

                    if i % config.league.snapshot_interval == 0:
                        league_pool_dir.mkdir(parents=True, exist_ok=True)
                        snap_path = str(league_pool_dir / f"iter_{i:03d}.pt")
                        shutil.copy2(ts_path, snap_path)
                        pool.add_snapshot(f"snapshot_iter_{i:03d}", snap_path)
                        self._presenter.on_league_snapshot_added(i, snap_path)

                if config.memory_telemetry and i % config.memory_telemetry_every == 0:
                    stats = _collect_memory_stats()
                    deltas = None
                    if prev_mem_stats is not None:
                        deltas = {
                            k: stats[k] - prev_mem_stats[k]
                            for k in stats.keys()
                            if k in prev_mem_stats
                        }
                    self._presenter.on_memory_stats(i, stats, deltas)
                    prev_mem_stats = stats
        finally:
            self._iteration_runner.teardown()

        run.end_time = time.time()

        # Copy best checkpoint
        if run.results:
            best_iter = run.best_loss_iteration if config.best_model_metric == "loss" else run.best_iteration
            best_src = str(save_dir / f"iter_{best_iter:03d}.pt")
            best_dst = str(save_dir / "best.pt")
            self._model.copy_best(best_src, best_dst)

        self._presenter.on_training_complete(run)
        return run


def _collect_memory_stats() -> dict[str, float]:
    stats: dict[str, float] = {}

    # Process RSS (best-effort, no external dependency)
    rss_mb = _read_process_rss_mb()
    if rss_mb is not None:
        stats["rss_mb"] = rss_mb
    stats.update(_read_process_platform_memory_stats_mb())

    # Python heap tracked by tracemalloc
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        stats["py_heap_mb"] = cur / (1024 * 1024)
        stats["py_heap_peak_mb"] = peak / (1024 * 1024)

    # CUDA allocator stats
    if torch.cuda.is_available():
        try:
            stats["cuda_alloc_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass

    # MPS allocator stats (Apple Silicon)
    if hasattr(torch, "mps"):
        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                stats["mps_alloc_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
            if hasattr(torch.mps, "driver_allocated_memory"):
                stats["mps_driver_mb"] = torch.mps.driver_allocated_memory() / (1024 * 1024)
        except Exception:
            pass

    return stats


def _read_process_rss_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        if not out:
            return None
        # rss from ps is in KiB on macOS.
        return float(out) / 1024.0
    except Exception:
        return None


def _read_process_platform_memory_stats_mb() -> dict[str, float]:
    if sys.platform == "darwin":
        return _read_macos_vmmap_stats_mb()
    return {}


def _read_macos_vmmap_stats_mb() -> dict[str, float]:
    try:
        out = subprocess.check_output(
            ["vmmap", "-summary", str(os.getpid())],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}

    stats: dict[str, float] = {}
    for line in out.splitlines():
        if line.startswith("Physical footprint:"):
            value = _parse_macos_size_mb(line.split(":", 1)[1])
            if value is not None:
                stats["footprint_mb"] = value
        elif line.startswith("Compressed:"):
            value = _parse_macos_size_mb(line.split(":", 1)[1])
            if value is not None:
                stats["compressed_mb"] = value
    return stats


def _parse_macos_size_mb(raw: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])", raw.strip())
    if match is None:
        return None

    value = float(match.group(1))
    unit = match.group(2)
    unit_scale = {
        "K": 1 / 1024,
        "M": 1.0,
        "G": 1024.0,
        "T": 1024.0 * 1024.0,
        "P": 1024.0 * 1024.0 * 1024.0,
    }
    return value * unit_scale[unit]
