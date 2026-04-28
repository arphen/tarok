"""Use case: execute one iteration of the training loop."""

from __future__ import annotations

import time
from dataclasses import replace as dc_replace

from training.entities.training_context import TrainingContext
from training.entities.training_config import scheduled_coef
from training.ports.explore_rate_policy_port import ExploreRatePolicyPort
from training.ports.imitation_coef_policy_port import ImitationCoefPolicyPort
from training.ports.iteration_runner_port import IterationRunnerPort
from training.ports.learning_rate_policy_port import LearningRatePolicyPort
from training.ports.presenter_port import PresenterPort
from training.use_cases.maintain_league_pool import MaintainLeaguePool
from training.use_cases.sample_league_seats import SampleLeagueSeats


class AdvanceIteration:
    """Execute one full iteration of the training loop.

    Single responsibility: translate the current training context and iteration
    number into an updated context — compute hyperparameters via policy ports,
    sample league seats, delegate to the iteration runner, update the run
    aggregate, and advance league Elo.  All side effects flow through injected
    ports and use cases; no infrastructure imports.
    """

    def __init__(
        self,
        iteration_runner: IterationRunnerPort,
        lr_policy: LearningRatePolicyPort,
        imitation_policy: ImitationCoefPolicyPort,
        entropy_policy,  # DefaultEntropyCoefPolicy | EloDecayEntropyPolicy
        presenter: PresenterPort,
        league_maintenance: MaintainLeaguePool,
        sample_seats: SampleLeagueSeats,
        behavioral_clone_policy=None,
        explore_rate_policy: ExploreRatePolicyPort | None = None,
    ) -> None:
        self._iteration_runner = iteration_runner
        self._lr_policy = lr_policy
        self._imitation_policy = imitation_policy
        self._entropy_policy = entropy_policy
        self._behavioral_clone_policy = behavioral_clone_policy
        self._explore_rate_policy = explore_rate_policy
        self._presenter = presenter
        self._league_maintenance = league_maintenance
        self._sample_seats = sample_seats

    def execute(self, ctx: TrainingContext, iteration: int) -> None:
        config = ctx.run.config
        identity = ctx.run.identity

        elapsed = time.time() - ctx.run.start_time
        self._presenter.on_iteration_start(iteration, config.iterations, elapsed)

        iter_lr = self._lr_policy.compute(
            config=config, iteration=iteration, learner_elo=ctx.pool.learner_elo,
        )
        iter_imitation_coef = self._imitation_policy.compute(
            config=config, iteration=iteration, learner_elo=ctx.pool.learner_elo,
        )
        if self._behavioral_clone_policy is None:
            iter_behavioral_clone_coef = scheduled_coef(
                iteration=max(0, iteration - 1),
                total_iterations=config.iterations,
                coef_max=config.behavioral_clone_coef,
                coef_min=config.behavioral_clone_coef_min,
                schedule=config.behavioral_clone_schedule,
            )
        else:
            iter_behavioral_clone_coef = self._behavioral_clone_policy.compute(
                config=config, iteration=iteration, learner_elo=ctx.pool.learner_elo,
            )
        iter_entropy_coef = self._entropy_policy.compute(
            config=config, iteration=iteration, learner_elo=ctx.pool.learner_elo,
        )
        if self._explore_rate_policy is None:
            iter_explore_rate = scheduled_coef(
                iteration=max(0, iteration - 1),
                total_iterations=config.iterations,
                coef_max=config.explore_rate,
                coef_min=config.effective_explore_rate_min,
                schedule=config.explore_rate_schedule,
            )
        else:
            iter_explore_rate = self._explore_rate_policy.compute(
                config=config, iteration=iteration, learner_elo=ctx.pool.learner_elo,
            )

        seats_override = self._sample_seats.execute(
            ctx.pool,
            num_seats=len([s for s in config.seats.split(",") if s.strip()]),
        )
        prev_placement = ctx.run.placements[-1]

        result = self._iteration_runner.run_iteration(
            iteration, config, identity, ctx.ts_path, ctx.save_dir,
            prev_placement=prev_placement,
            iter_lr=iter_lr,
            iter_imitation_coef=iter_imitation_coef,
            iter_behavioral_clone_coef=iter_behavioral_clone_coef,
            iter_entropy_coef=iter_entropy_coef,
            iter_explore_rate=iter_explore_rate,
            seats_override=seats_override,
            run_benchmark=config.should_benchmark_iteration(iteration),
            pool=ctx.pool,
        )
        ctx.run.results.append(result)
        self._presenter.on_iteration_done(prev_placement, result.placement, result.total_time)

        result = dc_replace(result, learner_elo=ctx.pool.learner_elo)
        ctx.run.results[-1] = result

        ctx.last_snapshot_elo = self._league_maintenance.execute(
            pool=ctx.pool,
            result=result,
            iteration=iteration,
            ts_path=ctx.ts_path,
            league_pool_dir=ctx.league_pool_dir,
            last_snapshot_elo=ctx.last_snapshot_elo,
            concurrency=config.concurrency,
            session_size=config.outplace_session_size,
            lapajne_mc_worlds=config.lapajne_mc_worlds,
            lapajne_mc_sims=config.lapajne_mc_sims,
        )
