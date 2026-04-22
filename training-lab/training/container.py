"""Composition root — wire adapters into use cases.

This is the only place that knows about concrete adapter classes.
Swap any adapter here to change behavior (e.g. mock presenter for tests).
"""

from __future__ import annotations

from training.entities.training_config import TrainingConfig
from training.ports import (
    BenchmarkPort,
    ConfigPort,
    ExploreRatePolicyPort,
    ImitationCoefPolicyPort,
    IterationRunnerPort,
    LeagueStatePersistencePort,
    LearningRatePolicyPort,
    ModelPort,
    PPOPort,
    PresenterPort,
    SelfPlayPort,
)
from training.use_cases import ResolveConfig, ResolveModel, TrainModel


def _default_selfplay() -> SelfPlayPort:
    from training.adapters.self_play import RustSelfPlay
    return RustSelfPlay()


def _default_benchmark() -> BenchmarkPort:
    from training.adapters.evaluation import SessionBenchmark
    return SessionBenchmark()


def _default_ppo() -> PPOPort:
    from training.adapters.ppo import PPOAdapter
    return PPOAdapter()


def _default_model() -> ModelPort:
    from training.adapters.modeling import TorchModelAdapter
    return TorchModelAdapter()


def _default_config() -> ConfigPort:
    from training.adapters.configuration import YAMLConfigLoader
    return YAMLConfigLoader()


def _default_presenter() -> PresenterPort:
    from training.adapters.presentation import TerminalPresenter
    return TerminalPresenter()


def _default_entropy_policy(config: TrainingConfig | None = None):
    if config is not None and config.entropy_schedule == "elo":
        from training.use_cases.train_model.policies import EloDecayEntropyPolicy
        return EloDecayEntropyPolicy()
    from training.use_cases.train_model.policies import DefaultEntropyCoefPolicy
    return DefaultEntropyCoefPolicy()


def _default_explore_rate_policy(config: TrainingConfig | None = None) -> ExploreRatePolicyPort:
    if config is not None and config.explore_rate_schedule == "elo":
        from training.use_cases.train_model.policies import EloDecayExplorePolicy
        return EloDecayExplorePolicy()
    from training.use_cases.train_model.policies import DefaultExploreRatePolicy
    return DefaultExploreRatePolicy()


def _default_league_persistence() -> LeagueStatePersistencePort:
    from training.adapters.persistence import JsonLeagueStatePersistence
    return JsonLeagueStatePersistence()


def _default_imitation_policy(config: TrainingConfig | None = None) -> ImitationCoefPolicyPort:
    if config is not None and config.imitation_schedule == "gaussian_elo":
        from training.use_cases.train_model.policies import EloGaussianILPolicy
        return EloGaussianILPolicy()
    from training.adapters.policies import ScheduledImitationCoefPolicy
    return ScheduledImitationCoefPolicy()


def _default_lr_policy() -> LearningRatePolicyPort:
    from training.adapters.policies import LeagueEloLearningRatePolicy
    return LeagueEloLearningRatePolicy()


def _default_iteration_runner(
    selfplay: SelfPlayPort,
    ppo: PPOPort,
    benchmark: BenchmarkPort,
    model: ModelPort,
    presenter: PresenterPort,
    config: TrainingConfig | None = None,
) -> IterationRunnerPort:
    from training.adapters.iteration_runners import ConfigurableIterationRunner

    duplicate_pairing = None
    duplicate_reward = None
    if config is not None and getattr(config, "duplicate", None) is not None and config.duplicate.enabled:
        from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
        from training.adapters.duplicate.seeded_self_play_adapter import SeededSelfPlayAdapter
        from training.adapters.duplicate.shadow_score_reward import ShadowScoreRewardAdapter

        # Wrap the default self-play adapter so ``run_seeded_pods`` is
        # available; legacy ``run``/``compute_run_stats`` still delegate to
        # the inner Rust adapter.
        selfplay = SeededSelfPlayAdapter(inner=selfplay)
        duplicate_pairing = RotationPairingAdapter(pairing=config.duplicate.pairing)
        duplicate_reward = ShadowScoreRewardAdapter()

    return ConfigurableIterationRunner(
        selfplay, ppo, benchmark, model, presenter,
        duplicate_pairing=duplicate_pairing,
        duplicate_reward=duplicate_reward,
    )


class Container:
    """Lazy DI container. Build once, use everywhere.

    Pass explicit adapter instances to override defaults (e.g. for testing).
    Defaults are constructed lazily so heavy imports (torch, tarok_engine)
    only happen when the adapter is actually needed.
    """

    def __init__(
        self,
        selfplay: SelfPlayPort | None = None,
        benchmark: BenchmarkPort | None = None,
        ppo: PPOPort | None = None,
        iteration_runner: IterationRunnerPort | None = None,
        imitation_policy: ImitationCoefPolicyPort | None = None,
        lr_policy: LearningRatePolicyPort | None = None,
        model: ModelPort | None = None,
        config_loader: ConfigPort | None = None,
        presenter: PresenterPort | None = None,
        league_persistence: LeagueStatePersistencePort | None = None,
    ):
        self._selfplay = selfplay
        self._benchmark = benchmark
        self._ppo = ppo
        self._iteration_runner = iteration_runner
        self._imitation_policy = imitation_policy
        self._lr_policy = lr_policy
        self._model = model
        self._config_loader = config_loader
        self._presenter = presenter
        self._league_persistence = league_persistence

    @property
    def selfplay(self) -> SelfPlayPort:
        if self._selfplay is None:
            self._selfplay = _default_selfplay()
        return self._selfplay

    @property
    def benchmark(self) -> BenchmarkPort:
        if self._benchmark is None:
            self._benchmark = _default_benchmark()
        return self._benchmark

    @property
    def ppo(self) -> PPOPort:
        if self._ppo is None:
            self._ppo = _default_ppo()
        return self._ppo

    @property
    def iteration_runner(self) -> IterationRunnerPort:
        if self._iteration_runner is None:
            self._iteration_runner = _default_iteration_runner(
                self.selfplay,
                self.ppo,
                self.benchmark,
                self.model,
                self.presenter,
            )
        return self._iteration_runner

    def _iteration_runner_for(self, config: TrainingConfig | None) -> IterationRunnerPort:
        """Config-aware iteration runner: injects duplicate-RL adapters when
        ``config.duplicate.enabled`` is True. Falls back to the cached default
        runner when duplicate is disabled or config is None."""
        if self._iteration_runner is not None:
            return self._iteration_runner
        if config is not None and getattr(config, "duplicate", None) is not None and config.duplicate.enabled:
            self._iteration_runner = _default_iteration_runner(
                self.selfplay, self.ppo, self.benchmark, self.model, self.presenter,
                config=config,
            )
            return self._iteration_runner
        return self.iteration_runner

    @property
    def model(self) -> ModelPort:
        if self._model is None:
            self._model = _default_model()
        return self._model

    @property
    def lr_policy(self) -> LearningRatePolicyPort:
        if self._lr_policy is None:
            self._lr_policy = _default_lr_policy()
        return self._lr_policy

    @property
    def imitation_policy(self) -> ImitationCoefPolicyPort:
        if self._imitation_policy is None:
            self._imitation_policy = _default_imitation_policy()
        return self._imitation_policy

    @property
    def config_loader(self) -> ConfigPort:
        if self._config_loader is None:
            self._config_loader = _default_config()
        return self._config_loader

    @property
    def presenter(self) -> PresenterPort:
        if self._presenter is None:
            self._presenter = _default_presenter()
        return self._presenter

    @property
    def league_persistence(self) -> LeagueStatePersistencePort:
        if self._league_persistence is None:
            self._league_persistence = _default_league_persistence()
        return self._league_persistence

    def resolve_model(self) -> ResolveModel:
        return ResolveModel(self.model)

    def resolve_config(self) -> ResolveConfig:
        return ResolveConfig(self.config_loader)

    def train_model(self, config: TrainingConfig | None = None) -> TrainModel:
        return TrainModel(
            iteration_runner=self._iteration_runner_for(config),
            benchmark=self.benchmark,
            model=self.model,
            presenter=self.presenter,
            selfplay=self.selfplay,
            lr_policy=self.lr_policy,
            imitation_policy=self._imitation_policy or _default_imitation_policy(config),
            entropy_policy=_default_entropy_policy(config),
            explore_rate_policy=_default_explore_rate_policy(config),
            league_persistence=self.league_persistence,
        )
