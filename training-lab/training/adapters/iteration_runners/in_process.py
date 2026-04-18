"""Adapter: in-process iteration runner (default).

Runs every iteration in the same process. Fastest, but subject to
PyTorch memory accumulation over long runs.
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
from training.use_cases.run_iteration import RunIteration


class InProcessIterationRunner(IterationRunnerPort):
    def __init__(
        self,
        selfplay: SelfPlayPort,
        ppo: PPOPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
    ):
        self._ppo = ppo
        self._run_iteration = RunIteration(selfplay, ppo, benchmark, model, presenter)

    def setup(self, weights: dict, config: TrainingConfig, device: str) -> None:
        self._ppo.setup(weights, config, device)

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
        result, _ = self._run_iteration.execute(
            i, config, identity, ts_path, save_dir,
            prev_placement=prev_placement,
            iter_lr=iter_lr,
            iter_imitation_coef=iter_imitation_coef,
            iter_entropy_coef=iter_entropy_coef,
            seats_override=seats_override,
            run_benchmark=run_benchmark,
        )
        return result
