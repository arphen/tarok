"""Adapter: in-process iteration runner (default).

Runs every iteration in the same process. Fastest, but subject to
PyTorch memory accumulation over long runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.ports.benchmark_port import BenchmarkPort
from training.ports.duplicate_iteration_stats_port import DuplicateIterationStatsPort
from training.ports.duplicate_pairing_port import DuplicatePairingPort
from training.ports.duplicate_reward_port import DuplicateRewardPort
from training.ports.duplicate_shadow_source_port import DuplicateShadowSourcePort
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.model_port import ModelPort
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.run_iteration import RunIteration

if TYPE_CHECKING:
    from training.entities.league import LeaguePool


class InProcessIterationRunner(IterationRunnerPort):
    def __init__(
        self,
        selfplay: SelfPlayPort,
        ppo: PPOPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
        *,
        duplicate_pairing: DuplicatePairingPort | None = None,
        duplicate_reward: DuplicateRewardPort | None = None,
        duplicate_shadow_source: DuplicateShadowSourcePort | None = None,
        duplicate_iteration_stats: DuplicateIterationStatsPort | None = None,
    ):
        self._ppo = ppo
        self._run_iteration = RunIteration(
            selfplay, ppo, benchmark, model, presenter,
            duplicate_pairing=duplicate_pairing,
            duplicate_reward=duplicate_reward,
            duplicate_shadow_source=duplicate_shadow_source,
            duplicate_iteration_stats=duplicate_iteration_stats,
        )

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
        iter_behavioral_clone_coef: float | None,
        iter_entropy_coef: float | None,
        iter_explore_rate: float | None = None,
        seats_override: str | None,
        run_benchmark: bool,
        pool: "LeaguePool | None" = None,
    ) -> IterationResult:
        result, _ = self._run_iteration.execute(
            i, config, identity, ts_path, save_dir,
            prev_placement=prev_placement,
            iter_lr=iter_lr,
            iter_imitation_coef=iter_imitation_coef,
            iter_behavioral_clone_coef=iter_behavioral_clone_coef,
            iter_entropy_coef=iter_entropy_coef,
            iter_explore_rate=iter_explore_rate,
            seats_override=seats_override,
            run_benchmark=run_benchmark,
            pool=pool,
        )
        return result
