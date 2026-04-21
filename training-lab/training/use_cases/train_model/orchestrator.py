"""Use case: full training loop orchestration."""

from __future__ import annotations

import shutil
import time
from dataclasses import replace as dc_replace
from pathlib import Path

from training.entities.league import LeaguePool
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
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.calibrate_initial_league_elo import CalibrateInitialLeagueElo
from training.use_cases.maintain_league_pool import MaintainLeaguePool
from training.use_cases.sample_league_seats import SampleLeagueSeats
from training.use_cases.train_model.policies import (
    DefaultBehavioralCloneCoefPolicy,
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
        selfplay: SelfPlayPort | None = None,
        lr_policy: LearningRatePolicyPort | None = None,
        imitation_policy: ImitationCoefPolicyPort | None = None,
        entropy_policy: DefaultEntropyCoefPolicy | EloDecayEntropyPolicy | None = None,
        league_persistence: LeagueStatePersistencePort | None = None,
        behavioral_clone_policy: DefaultBehavioralCloneCoefPolicy | None = None,
    ):
        self._iteration_runner = iteration_runner
        self._benchmark = benchmark
        self._model = model
        self._presenter = presenter
        self._selfplay = selfplay
        self._lr_policy = lr_policy if lr_policy is not None else DefaultLearningRatePolicy()
        self._imitation_policy = (
            imitation_policy if imitation_policy is not None else DefaultImitationCoefPolicy()
        )
        self._entropy_policy = entropy_policy if entropy_policy is not None else DefaultEntropyCoefPolicy()
        self._bc_policy = behavioral_clone_policy if behavioral_clone_policy is not None else DefaultBehavioralCloneCoefPolicy()
        if league_persistence is None:
            raise ValueError("league_persistence must be provided — wire JsonLeagueStatePersistence via the container")
        self._league_persistence: LeagueStatePersistencePort = league_persistence

    def execute(
        self,
        config: TrainingConfig,
        identity: ModelIdentity,
        weights: dict,
        device: str,
    ) -> TrainingRun:
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts_path = str(save_dir / "_current.pt")

        self._model.export_for_inference(
            weights, identity.hidden_size, identity.oracle_critic, identity.model_arch, ts_path,
        )
        self._presenter.on_model_loaded(identity, str(save_dir))
        self._presenter.on_device_selected(device)

        # Setup iteration runner (PPO/setup ownership lives at adapter layer)
        self._iteration_runner.setup(weights, config, device)

        # Initial benchmark
        self._presenter.on_training_plan(config)
        t0 = time.time()
        initial = self._benchmark.measure_placement(
            ts_path, config.bench_games, config.effective_bench_seats,
            config.concurrency, session_size=50,
            lapajne_mc_worlds=config.lapajne_mc_worlds,
            lapajne_mc_sims=config.lapajne_mc_sims,
        )
        self._presenter.on_initial_benchmark(
            initial, config.bench_games, config.effective_bench_seats, time.time() - t0,
        )

        self._presenter.on_training_loop_start(config)

        run = TrainingRun(
            config=config,
            identity=identity,
            initial_placement=initial,
            start_time=time.time(),
        )

        if config.league is None or not config.league.enabled:
            raise ValueError("TrainModel requires league.enabled=true")

        pool = LeaguePool(config=config.league)
        league_maintenance = MaintainLeaguePool(
            updater=UpdateLeagueElo(),
            presenter=self._presenter,
            persistence=self._league_persistence,
            selfplay=self._selfplay,
        )
        league_pool_dir = save_dir / "league_pool"
        league_pool_dir.mkdir(parents=True, exist_ok=True)
        state_path = league_maintenance.state_path(league_pool_dir)
        profile_path = league_pool_dir / "profile.txt"

        previous_profile = (
            profile_path.read_text(encoding="utf-8").strip()
            if profile_path.exists()
            else ""
        )
        if state_path.exists() and previous_profile and previous_profile != config.profile_name:
            approved = self._presenter.confirm_league_state_reset(
                previous_profile=previous_profile,
                current_profile=config.profile_name,
                league_pool_dir=str(league_pool_dir),
            )
            if not approved:
                raise RuntimeError(
                    "League state reset declined. Aborting to avoid overriding league state "
                    f"from '{previous_profile}' with '{config.profile_name}'."
                )
            shutil.rmtree(league_pool_dir, ignore_errors=True)
            league_pool_dir.mkdir(parents=True, exist_ok=True)

        had_state = state_path.exists()
        self._league_persistence.restore(pool, state_path)
        profile_path.write_text(config.profile_name, encoding="utf-8")

        if config.league.initial_calibration_enabled and not had_state:
            if self._selfplay is None:
                raise ValueError("Initial league calibration requires selfplay port")
            self._presenter.on_initial_league_calibration_start(
                n_opponents=len(pool.entries),
                n_games_per_pair=config.league.initial_calibration_games_per_pair,
                anchor_name=config.league.initial_calibration_anchor,
                anchor_elo=config.league.initial_calibration_anchor_elo,
            )
            t_cal = time.time()
            calibrated = CalibrateInitialLeagueElo().execute(
                pool=pool,
                selfplay=self._selfplay,
                model_path=ts_path,
                n_games_per_pair=config.league.initial_calibration_games_per_pair,
                concurrency=config.concurrency,
                session_size=config.outplace_session_size,
                anchor_name=config.league.initial_calibration_anchor,
                anchor_elo=config.league.initial_calibration_anchor_elo,
                lapajne_mc_worlds=config.lapajne_mc_worlds,
                lapajne_mc_sims=config.lapajne_mc_sims,
                on_mixed_result=self._presenter.on_initial_league_calibration_mixed_result,
            )
            self._presenter.on_initial_league_calibration_done(time.time() - t_cal)
            if calibrated:
                self._league_persistence.save(pool, state_path)

        # Ensure there is always a baseline ghost before the first learning
        # iteration. On fresh league runs this captures the calibrated model;
        # future snapshots are then admitted only if stronger than this base.
        if pool.config.max_active_snapshots > 0:
            has_checkpoint = any(e.opponent.type == "nn_checkpoint" for e in pool.entries)
            if not has_checkpoint and Path(ts_path).exists():
                league_pool_dir.mkdir(parents=True, exist_ok=True)
                snap_path = str(league_pool_dir / "iter_000.pt")
                shutil.copy2(ts_path, snap_path)
                pool.add_snapshot("ghost@0", snap_path, initial_elo=pool.learner_elo)
                self._league_persistence.save(pool, state_path)
                self._presenter.on_league_snapshot_added(0, snap_path)

        last_snapshot_elo: float | None = league_maintenance.initial_snapshot_elo(pool)

        sample_seats = SampleLeagueSeats()

        try:
            for i in range(1, config.iterations + 1):
                elapsed = time.time() - run.start_time
                self._presenter.on_iteration_start(i, config.iterations, elapsed)

                iter_lr = self._lr_policy.compute(
                    config=config,
                    iteration=i,
                    learner_elo=pool.learner_elo,
                )
                iter_imitation_coef = self._imitation_policy.compute(
                    config=config, iteration=i, learner_elo=pool.learner_elo,
                )
                iter_entropy_coef = self._entropy_policy.compute(
                    config=config, iteration=i, learner_elo=pool.learner_elo,
                )
                iter_behavioral_clone_coef = self._bc_policy.compute(
                    config=config, iteration=i, learner_elo=pool.learner_elo,
                )

                seats_override = sample_seats.execute(pool)

                prev = run.placements[-1]
                should_bench = config.should_benchmark_iteration(i)
                result = self._iteration_runner.run_iteration(
                    i, config, identity, ts_path, save_dir,
                    prev_placement=prev,
                    iter_lr=iter_lr,
                    iter_imitation_coef=iter_imitation_coef,
                    iter_entropy_coef=iter_entropy_coef,
                    iter_behavioral_clone_coef=iter_behavioral_clone_coef,
                    seats_override=seats_override,
                    run_benchmark=should_bench,
                )
                run.results.append(result)
                self._presenter.on_iteration_done(prev, result.placement, result.total_time)

                result = dc_replace(result, learner_elo=pool.learner_elo)
                run.results[-1] = result

                last_snapshot_elo = league_maintenance.execute(
                    pool=pool,
                    result=result,
                    iteration=i,
                    ts_path=ts_path,
                    league_pool_dir=league_pool_dir,
                    last_snapshot_elo=last_snapshot_elo,
                    concurrency=config.concurrency,
                    session_size=config.outplace_session_size,
                    lapajne_mc_worlds=config.lapajne_mc_worlds,
                    lapajne_mc_sims=config.lapajne_mc_sims,
                )
        finally:
            self._iteration_runner.teardown()

        run.end_time = time.time()

        # Copy best checkpoint
        if run.results:
            if config.best_model_metric == "loss":
                best_iter = run.best_loss_iteration
            elif config.best_model_metric == "elo":
                best_iter = run.best_elo_iteration
            else:
                best_iter = run.best_iteration
            best_src = str(save_dir / f"iter_{best_iter:03d}.pt")
            best_dst = str(save_dir / "best.pt")
            self._model.copy_best(best_src, best_dst)

        self._presenter.on_training_complete(run)
        return run
