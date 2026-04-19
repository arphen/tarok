"""Adapter: spawn-based iteration runner.

Runs iterations in a worker subprocess. The worker is killed and respawned
every ``restart_every`` iterations, reclaiming any accumulated PyTorch memory.
Inter-process communication uses multiprocessing queues; presenter events are
forwarded through a lightweight proxy so the terminal UI stays in the parent.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.presenter_port import PresenterPort

log = logging.getLogger(__name__)


@dataclass
class _WorkerInit:
    """Sent as the first message to initialize PPO in a fresh worker."""

    weights: dict
    config: TrainingConfig
    device: str


@dataclass
class _PresenterEvent:
    method: str
    args: tuple
    kwargs: dict


@dataclass
class _IterationTask:
    i: int
    config: TrainingConfig
    identity: ModelIdentity
    ts_path: str
    save_dir: Path
    prev_placement: float
    iter_lr: float | None
    iter_imitation_coef: float | None
    iter_behavioral_clone_coef: float | None
    iter_entropy_coef: float | None
    seats_override: str | None
    run_benchmark: bool


@dataclass
class _IterationResult:
    result: IterationResult | None
    new_weights: dict | None
    error: str | None


class _QueuePresenter(PresenterPort):
    """Sends presenter calls from worker to parent via queue."""

    def __init__(self, queue: mp.Queue) -> None:
        self._q = queue

    def _send(self, method: str, *args: Any, **kwargs: Any) -> None:
        self._q.put(_PresenterEvent(method, args, kwargs))

    def on_model_loaded(self, identity, save_dir):
        self._send("on_model_loaded", identity, save_dir)

    def on_device_selected(self, device):
        self._send("on_device_selected", device)

    def on_training_plan(self, config):
        pass

    def on_initial_benchmark(self, placement, n_games, seats, elapsed):
        pass

    def on_training_loop_start(self, config):
        pass

    def on_iteration_start(self, iteration, total, elapsed):
        self._send("on_iteration_start", iteration, total, elapsed)

    def on_selfplay_start(self, config, effective_seats=None):
        self._send("on_selfplay_start", config, effective_seats=effective_seats)

    def on_selfplay_done(self, n_total, n_learner, elapsed):
        self._send("on_selfplay_done", n_total, n_learner, elapsed)

    def on_ppo_start(
        self,
        config,
        iter_lr=None,
        iter_imitation_coef=None,
        iter_behavioral_clone_coef=None,
        iter_entropy_coef=None,
    ):
        self._send(
            "on_ppo_start",
            config,
            iter_lr=iter_lr,
            iter_imitation_coef=iter_imitation_coef,
            iter_behavioral_clone_coef=iter_behavioral_clone_coef,
            iter_entropy_coef=iter_entropy_coef,
        )

    def on_ppo_done(self, metrics, elapsed):
        self._send("on_ppo_done", metrics, elapsed)

    def on_benchmark_start(self, config):
        self._send("on_benchmark_start", config)

    def on_benchmark_done(self, placement, elapsed):
        self._send("on_benchmark_done", placement, elapsed)

    def on_benchmark_skipped(self, iteration, config):
        self._send("on_benchmark_skipped", iteration, config)

    def on_iteration_done(self, prev_placement, curr_placement, elapsed):
        self._send("on_iteration_done", prev_placement, curr_placement, elapsed)

    def on_training_complete(self, run):
        pass


def _worker_main(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    event_queue: mp.Queue,
    adapter_factory: Callable,
) -> None:
    """Worker loop: receive init, then execute iterations until sentinel."""
    from training.use_cases.run_iteration import RunIteration

    selfplay, ppo, benchmark, model = adapter_factory()
    presenter = _QueuePresenter(event_queue)
    run_iteration = RunIteration(selfplay, ppo, benchmark, model, presenter)

    init: _WorkerInit = task_queue.get()
    ppo.setup(init.weights, init.config, init.device)

    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            result, new_weights = run_iteration.execute(
                task.i,
                task.config,
                task.identity,
                task.ts_path,
                task.save_dir,
                prev_placement=task.prev_placement,
                iter_lr=task.iter_lr,
                iter_imitation_coef=task.iter_imitation_coef,
                iter_behavioral_clone_coef=task.iter_behavioral_clone_coef,
                iter_entropy_coef=task.iter_entropy_coef,
                seats_override=task.seats_override,
                run_benchmark=task.run_benchmark,
            )
            result_queue.put(_IterationResult(result=result, new_weights=new_weights, error=None))
        except Exception:
            result_queue.put(_IterationResult(result=None, new_weights=None, error=traceback.format_exc()))


class SpawnIterationRunner(IterationRunnerPort):
    """Runner that restarts the worker process every N iterations.

    Restarting drops PyTorch allocator state in that process, which is useful
    when long runs exhibit process growth that is otherwise hard to reclaim.
    """

    def __init__(
        self,
        adapter_factory: Callable,
        presenter: PresenterPort,
        restart_every: int = 10,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._presenter = presenter
        self._restart_every = max(1, int(restart_every))

        self._device: str = "cpu"
        self._config: TrainingConfig | None = None
        self._current_weights: dict | None = None

        self._worker: mp.Process | None = None
        self._task_q: mp.Queue | None = None
        self._result_q: mp.Queue | None = None
        self._event_q: mp.Queue | None = None
        self._iters_in_current_worker = 0

    def setup(self, weights: dict, config: TrainingConfig, device: str) -> None:
        self._current_weights = weights
        self._config = config
        self._device = device

    def run_iteration(
        self,
        i: int,
        config: TrainingConfig,
        identity: ModelIdentity,
        ts_path: str,
        save_dir: Path,
        *,
        prev_placement: float,
        iter_lr: float | None,
        iter_imitation_coef: float | None,
        iter_behavioral_clone_coef: float | None,
        iter_entropy_coef: float | None,
        seats_override: str | None,
        run_benchmark: bool,
    ) -> IterationResult:
        if self._worker is None or not self._worker.is_alive():
            self._spawn_worker()

        assert self._task_q is not None
        assert self._result_q is not None

        self._task_q.put(
            _IterationTask(
                i=i,
                config=config,
                identity=identity,
                ts_path=ts_path,
                save_dir=save_dir,
                prev_placement=prev_placement,
                iter_lr=iter_lr,
                iter_imitation_coef=iter_imitation_coef,
                iter_behavioral_clone_coef=iter_behavioral_clone_coef,
                iter_entropy_coef=iter_entropy_coef,
                seats_override=seats_override,
                run_benchmark=run_benchmark,
            )
        )

        result_msg: _IterationResult | None = None
        while result_msg is None:
            try:
                result_msg = self._result_q.get(timeout=0.05)
            except Empty:
                pass
            self._flush_events()

        self._iters_in_current_worker += 1

        if result_msg.error is not None:
            raise RuntimeError(f"Worker iteration {i} failed:\n{result_msg.error}")

        if result_msg.new_weights is not None:
            self._current_weights = result_msg.new_weights

        if self._iters_in_current_worker >= self._restart_every:
            log.info(
                "SpawnIterationRunner: restarting worker after %d iterations (restart_every=%d)",
                self._iters_in_current_worker,
                self._restart_every,
            )
            self._kill_worker()

        assert result_msg.result is not None
        return result_msg.result

    def teardown(self) -> None:
        self._kill_worker()

    def _spawn_worker(self) -> None:
        if self._current_weights is None or self._config is None:
            raise RuntimeError("SpawnIterationRunner.setup() must be called before run_iteration().")

        ctx = mp.get_context("spawn")
        self._task_q = ctx.Queue()
        self._result_q = ctx.Queue()
        self._event_q = ctx.Queue()
        self._worker = ctx.Process(
            target=_worker_main,
            args=(self._task_q, self._result_q, self._event_q, self._adapter_factory),
            daemon=True,
        )
        self._worker.start()
        self._iters_in_current_worker = 0

        self._task_q.put(_WorkerInit(self._current_weights, self._config, self._device))
        log.info("SpawnIterationRunner: spawned worker pid=%s", self._worker.pid)

    def _kill_worker(self) -> None:
        if self._worker is None:
            return

        try:
            assert self._task_q is not None
            self._task_q.put(None)
            self._worker.join(timeout=10)
        finally:
            if self._worker.is_alive():
                self._worker.kill()
                self._worker.join()

        self._flush_events()
        self._worker = None
        self._task_q = None
        self._result_q = None
        self._event_q = None
        log.info("SpawnIterationRunner: worker stopped")

    def _flush_events(self) -> None:
        if self._event_q is None:
            return

        while True:
            try:
                event: _PresenterEvent = self._event_q.get_nowait()
            except Empty:
                break

            method = getattr(self._presenter, event.method, None)
            if method is None:
                continue
            try:
                method(*event.args, **event.kwargs)
            except Exception:
                # Presenter failures should not crash training.
                pass
