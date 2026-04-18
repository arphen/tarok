"""Adapter: config-driven iteration runner selector.

Selects and owns one concrete runner strategy after `setup(config=...)`:
- in-process
- spawn (restart worker process every N iterations)
"""

from __future__ import annotations

from pathlib import Path

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.ports.benchmark_port import BenchmarkPort
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.model_port import ModelPort
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort


class ConfigurableIterationRunner(IterationRunnerPort):
    """Chooses runner implementation from TrainingConfig at setup time."""

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
        self._delegate: IterationRunnerPort | None = None

    def setup(self, weights: dict, config: TrainingConfig, device: str) -> None:
        mode = (config.iteration_runner_mode or "in-process").strip().lower()

        if mode in {"spawn", "process", "subprocess"}:
            from training.adapters.iteration_runners.spawn import SpawnIterationRunner

            self._delegate = SpawnIterationRunner(
                adapter_factory=self._adapter_factory_for_spawn,
                presenter=self._presenter,
                restart_every=config.iteration_runner_restart_every,
            )
        else:
            from training.adapters.iteration_runners.in_process import InProcessIterationRunner

            self._delegate = InProcessIterationRunner(
                self._selfplay,
                self._ppo,
                self._benchmark,
                self._model,
                self._presenter,
            )

        self._delegate.setup(weights, config, device)

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
        iter_entropy_coef: float | None,
        seats_override: str | None,
        run_benchmark: bool,
    ) -> IterationResult:
        if self._delegate is None:
            raise RuntimeError("ConfigurableIterationRunner.setup() must be called first.")

        return self._delegate.run_iteration(
            i,
            config,
            identity,
            ts_path,
            save_dir,
            prev_placement=prev_placement,
            iter_lr=iter_lr,
            iter_imitation_coef=iter_imitation_coef,
            iter_entropy_coef=iter_entropy_coef,
            seats_override=seats_override,
            run_benchmark=run_benchmark,
        )

    def teardown(self) -> None:
        if self._delegate is not None:
            self._delegate.teardown()
            self._delegate = None

    def _adapter_factory_for_spawn(self) -> tuple[SelfPlayPort, PPOPort, BenchmarkPort, ModelPort]:
        # Worker must own fresh adapters in its own process.
        from training.adapters.evaluation import SessionBenchmark
        from training.adapters.modeling import TorchModelAdapter
        from training.adapters.ppo import PPOAdapter
        from training.adapters.self_play import RustSelfPlay

        return RustSelfPlay(), PPOAdapter(), SessionBenchmark(), TorchModelAdapter()
