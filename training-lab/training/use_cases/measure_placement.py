"""Use case: measure model placement via greedy benchmark self-play."""

from __future__ import annotations

import time

from training.entities.training_config import TrainingConfig
from training.ports.benchmark_port import BenchmarkPort
from training.ports.presenter_port import PresenterPort


class MeasurePlacement:
    """Run the greedy benchmark and return the placement score.

    Single responsibility: decide whether to run the benchmark or carry forward
    the previous score, then delegate to BenchmarkPort and report timing.
    Returns ``(placement, bench_time)`` — both scalars, no side-effects beyond
    presenter notifications.
    """

    def __init__(self, benchmark: BenchmarkPort, presenter: PresenterPort) -> None:
        self._benchmark = benchmark
        self._presenter = presenter

    def execute(
        self,
        iteration: int,
        config: TrainingConfig,
        ts_path: str,
        prev_placement: float,
        run_benchmark: bool,
    ) -> tuple[float, float]:
        if run_benchmark:
            self._presenter.on_benchmark_start(config)
            t0 = time.time()
            placement = self._benchmark.measure_placement(
                ts_path,
                config.bench_games,
                config.effective_bench_seats,
                config.concurrency,
                session_size=config.outplace_session_size,
                lapajne_mc_worlds=config.lapajne_mc_worlds,
            lapajne_mc_sims=config.lapajne_mc_sims,
            )
            bench_time = time.time() - t0
            self._presenter.on_benchmark_done(placement, bench_time)
        else:
            placement = prev_placement
            bench_time = 0.0
            self._presenter.on_benchmark_skipped(iteration, config)

        return placement, bench_time
