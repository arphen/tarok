"""Adapter: config-driven iteration runner selector.

Selects and owns one concrete runner strategy after `setup(config=...)`:
- in-process
- spawn (restart worker process every N iterations)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.ports.benchmark_port import BenchmarkPort
from training.ports.duplicate_pairing_port import DuplicatePairingPort
from training.ports.duplicate_reward_port import DuplicateRewardPort
from training.ports.duplicate_shadow_source_port import DuplicateShadowSourcePort
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.model_port import ModelPort
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort

if TYPE_CHECKING:
    from training.entities.league import LeaguePool


class ConfigurableIterationRunner(IterationRunnerPort):
    """Chooses runner implementation from TrainingConfig at setup time."""

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
    ):
        self._selfplay = selfplay
        self._ppo = ppo
        self._benchmark = benchmark
        self._model = model
        self._presenter = presenter
        self._duplicate_pairing = duplicate_pairing
        self._duplicate_reward = duplicate_reward
        self._duplicate_shadow_source = duplicate_shadow_source
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
                duplicate_pairing=self._duplicate_pairing,
                duplicate_reward=self._duplicate_reward,
                duplicate_shadow_source=self._duplicate_shadow_source,
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
        iter_behavioral_clone_coef: float | None,
        iter_entropy_coef: float | None,
        iter_explore_rate: float | None = None,
        seats_override: str | None,
        run_benchmark: bool,
        pool: "LeaguePool | None" = None,
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
            iter_behavioral_clone_coef=iter_behavioral_clone_coef,
            iter_entropy_coef=iter_entropy_coef,
            iter_explore_rate=iter_explore_rate,
            seats_override=seats_override,
            run_benchmark=run_benchmark,
            pool=pool,
        )

    def teardown(self) -> None:
        if self._delegate is not None:
            self._delegate.teardown()
            self._delegate = None

    def _adapter_factory_for_spawn(
        self, config: TrainingConfig
    ) -> tuple[
        SelfPlayPort,
        PPOPort,
        BenchmarkPort,
        ModelPort,
        DuplicatePairingPort | None,
        DuplicateRewardPort | None,
        DuplicateShadowSourcePort | None,
    ]:
        # Worker must own fresh adapters in its own process. Duplicate-RL
        # ports are constructed lazily, mirroring ``_default_iteration_runner``
        # in the composition root so spawn runs match in-process behaviour.
        from training.adapters.evaluation import SessionBenchmark
        from training.adapters.modeling import TorchModelAdapter
        from training.adapters.ppo import PPOAdapter
        from training.adapters.self_play import RustSelfPlay

        selfplay: SelfPlayPort = RustSelfPlay()
        duplicate_pairing: DuplicatePairingPort | None = None
        duplicate_reward: DuplicateRewardPort | None = None
        duplicate_shadow_source: DuplicateShadowSourcePort | None = None

        duplicate_cfg = getattr(config, "duplicate", None)
        if duplicate_cfg is not None and duplicate_cfg.enabled:
            from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
            from training.adapters.duplicate.seeded_self_play_adapter import SeededSelfPlayAdapter
            from training.adapters.duplicate.shadow_score_reward import ShadowScoreRewardAdapter
            from training.adapters.duplicate.shadow_sources import create_shadow_source

            selfplay = SeededSelfPlayAdapter(inner=selfplay)
            duplicate_pairing = RotationPairingAdapter(pairing=duplicate_cfg.pairing)
            duplicate_reward = ShadowScoreRewardAdapter()
            duplicate_shadow_source = create_shadow_source(
                duplicate_cfg.shadow_source, rng_seed=duplicate_cfg.rng_seed,
            )

        return (
            selfplay,
            PPOAdapter(),
            SessionBenchmark(),
            TorchModelAdapter(),
            duplicate_pairing,
            duplicate_reward,
            duplicate_shadow_source,
        )
