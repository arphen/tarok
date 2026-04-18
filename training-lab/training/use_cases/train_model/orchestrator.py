"""Use case: full training loop orchestration — thin coordinator.

Delegates each phase to a single-responsibility use case:
  PrepareTraining     — validate, bootstrap infrastructure, initial benchmark, restore league
  AdvanceIteration    — per-iteration hyperparams + self-play + PPO + league update
  PromoteBestCheckpoint — post-loop: select best checkpoint by metric, copy to best.pt
"""

from __future__ import annotations

import time

from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.entities.training_run import TrainingRun
from training.ports.benchmark_port import BenchmarkPort
from training.ports.imitation_coef_policy_port import ImitationCoefPolicyPort
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.league_persistence_port import LeagueStatePersistencePort
from training.ports.learning_rate_policy_port import LearningRatePolicyPort
from training.ports.model_port import ModelPort
from training.ports.presenter_port import PresenterPort
from training.use_cases.advance_iteration import AdvanceIteration
from training.use_cases.maintain_league_pool import MaintainLeaguePool
from training.use_cases.prepare_training import PrepareTraining
from training.use_cases.promote_best_checkpoint import PromoteBestCheckpoint
from training.use_cases.sample_league_seats import SampleLeagueSeats
from training.use_cases.train_model.policies import (
    DefaultEntropyCoefPolicy,
    DefaultImitationCoefPolicy,
    DefaultLearningRatePolicy,
    EloDecayEntropyPolicy,
)
from training.use_cases.update_league_elo import UpdateLeagueElo


class TrainModel:
    def __init__(
        self,
        iteration_runner: IterationRunnerPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
        lr_policy: LearningRatePolicyPort | None = None,
        imitation_policy: ImitationCoefPolicyPort | None = None,
        entropy_policy: DefaultEntropyCoefPolicy | EloDecayEntropyPolicy | None = None,
        league_persistence: LeagueStatePersistencePort | None = None,
    ):
        if league_persistence is None:
            raise ValueError("league_persistence must be provided — wire JsonLeagueStatePersistence via the container")

        lr_policy = lr_policy if lr_policy is not None else DefaultLearningRatePolicy()
        imitation_policy = imitation_policy if imitation_policy is not None else DefaultImitationCoefPolicy()
        entropy_policy = entropy_policy if entropy_policy is not None else DefaultEntropyCoefPolicy()

        league_maintenance = MaintainLeaguePool(
            updater=UpdateLeagueElo(),
            presenter=presenter,
            persistence=league_persistence,
        )
        sample_seats = SampleLeagueSeats()

        self._iteration_runner = iteration_runner
        self._presenter = presenter
        self._prepare = PrepareTraining(benchmark, model, presenter, iteration_runner, league_persistence)
        self._advance_iteration = AdvanceIteration(
            iteration_runner, lr_policy, imitation_policy, entropy_policy,
            presenter, league_maintenance, sample_seats,
        )
        self._promote_best = PromoteBestCheckpoint(model)

    def execute(
        self,
        config: TrainingConfig,
        identity: ModelIdentity,
        weights: dict,
        device: str,
    ) -> TrainingRun:
        ctx = self._prepare.execute(config, identity, weights, device)
        self._presenter.on_training_loop_start(config)
        try:
            for i in range(1, config.iterations + 1):
                self._advance_iteration.execute(ctx, i)
        finally:
            self._iteration_runner.teardown()

        ctx.run.end_time = time.time()
        self._promote_best.execute(ctx.run, ctx.save_dir)
        self._presenter.on_training_complete(ctx.run)
        return ctx.run
