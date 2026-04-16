"""Composition root — wire adapters into use cases.

This is the only place that knows about concrete adapter classes.
Swap any adapter here to change behavior (e.g. mock presenter for tests).
"""

from __future__ import annotations

from training.ports import (
    BenchmarkPort,
    ConfigPort,
    IterationRunnerPort,
    ModelPort,
    PPOPort,
    PresenterPort,
    SelfPlayPort,
)
from training.use_cases import ResolveConfig, ResolveModel, TrainModel


def _default_selfplay() -> SelfPlayPort:
    from training.adapters.selfplay import RustSelfPlay
    return RustSelfPlay()


def _default_benchmark() -> BenchmarkPort:
    from training.adapters.benchmark import SessionBenchmark
    return SessionBenchmark()


def _default_ppo() -> PPOPort:
    from training.adapters.ppo import PPOAdapter
    return PPOAdapter()


def _default_model() -> ModelPort:
    from training.adapters.model import TorchModelAdapter
    return TorchModelAdapter()


def _default_config() -> ConfigPort:
    from training.adapters.config import YAMLConfigLoader
    return YAMLConfigLoader()


def _default_presenter() -> PresenterPort:
    from training.adapters.presenter import TerminalPresenter
    return TerminalPresenter()


def _default_iteration_runner(
    selfplay: SelfPlayPort,
    ppo: PPOPort,
    benchmark: BenchmarkPort,
    model: ModelPort,
    presenter: PresenterPort,
) -> IterationRunnerPort:
    from training.adapters.iteration_runners import ConfigurableIterationRunner
    return ConfigurableIterationRunner(selfplay, ppo, benchmark, model, presenter)


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
        model: ModelPort | None = None,
        config_loader: ConfigPort | None = None,
        presenter: PresenterPort | None = None,
    ):
        self._selfplay = selfplay
        self._benchmark = benchmark
        self._ppo = ppo
        self._iteration_runner = iteration_runner
        self._model = model
        self._config_loader = config_loader
        self._presenter = presenter

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

    @property
    def model(self) -> ModelPort:
        if self._model is None:
            self._model = _default_model()
        return self._model

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

    def resolve_model(self) -> ResolveModel:
        return ResolveModel(self.model)

    def resolve_config(self) -> ResolveConfig:
        return ResolveConfig(self.config_loader)

    def train_model(self) -> TrainModel:
        return TrainModel(
            iteration_runner=self.iteration_runner,
            benchmark=self.benchmark,
            model=self.model,
            presenter=self.presenter,
        )
