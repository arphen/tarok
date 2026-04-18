"""Focused tests for training-loop use cases extracted from TrainModel."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest

from training.entities.iteration_result import IterationResult
from training.entities.league import LeagueConfig, LeagueOpponent, LeaguePool, LeaguePoolEntry
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.entities.training_context import TrainingContext
from training.entities.training_run import TrainingRun
from training.use_cases.advance_iteration import AdvanceIteration
from training.use_cases.prepare_training import PrepareTraining
from training.use_cases.promote_best_checkpoint import PromoteBestCheckpoint


@pytest.fixture
def base_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        model_arch="v4",
        save_dir=str(tmp_path / "test_run"),
        iterations=3,
        bench_games=10,
        benchmark_checkpoints=(2,),
        league=LeagueConfig(enabled=True),
    )


@pytest.fixture
def identity() -> ModelIdentity:
    return ModelIdentity(
        name="TestModel",
        hidden_size=64,
        oracle_critic=False,
        model_arch="v4",
        is_new=True,
    )


def test_prepare_training_bootstraps_context_and_restores_snapshot_elo(
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    benchmark = MagicMock()
    benchmark.measure_placement.return_value = 2.75
    model = MagicMock()
    presenter = MagicMock()
    iteration_runner = MagicMock()
    league_persistence = MagicMock()

    def _restore(pool: LeaguePool, state_path: Path) -> None:
        assert state_path == Path(base_config.save_dir) / "league_pool" / "state.json"
        pool.learner_elo = 1444.0
        pool.entries.append(
            LeaguePoolEntry(
                opponent=LeagueOpponent(
                    name="snapshot_iter_005",
                    type="nn_checkpoint",
                    path=str(Path(base_config.save_dir) / "league_pool" / "iter_005.pt"),
                ),
                elo=1410.0,
            )
        )

    league_persistence.restore.side_effect = _restore

    ctx = PrepareTraining(
        benchmark=benchmark,
        model=model,
        presenter=presenter,
        iteration_runner=iteration_runner,
        league_persistence=league_persistence,
    ).execute(base_config, identity, {"dummy": "weights"}, "cpu")

    assert ctx.ts_path == str(Path(base_config.save_dir) / "_current.pt")
    assert ctx.save_dir == Path(base_config.save_dir)
    assert ctx.league_pool_dir == Path(base_config.save_dir) / "league_pool"
    assert ctx.run.initial_placement == pytest.approx(2.75)
    assert ctx.pool.learner_elo == pytest.approx(1444.0)
    assert ctx.last_snapshot_elo == pytest.approx(1410.0)

    model.export_for_inference.assert_called_once_with(
        {"dummy": "weights"},
        identity.hidden_size,
        identity.oracle_critic,
        identity.model_arch,
        str(Path(base_config.save_dir) / "_current.pt"),
    )
    iteration_runner.setup.assert_called_once_with({"dummy": "weights"}, base_config, "cpu")
    presenter.on_initial_benchmark.assert_called_once_with(
        2.75,
        base_config.bench_games,
        base_config.effective_bench_seats,
        ANY,
    )


def test_prepare_training_requires_enabled_league_before_setup(
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    benchmark = MagicMock()
    model = MagicMock()
    presenter = MagicMock()
    iteration_runner = MagicMock()
    league_persistence = MagicMock()
    config = replace(base_config, league=None)

    with pytest.raises(ValueError, match="requires league.enabled=true"):
        PrepareTraining(
            benchmark=benchmark,
            model=model,
            presenter=presenter,
            iteration_runner=iteration_runner,
            league_persistence=league_persistence,
        ).execute(config, identity, {}, "cpu")

    model.export_for_inference.assert_not_called()
    iteration_runner.setup.assert_not_called()
    benchmark.measure_placement.assert_not_called()


def test_advance_iteration_applies_policies_and_updates_context(
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    runner = MagicMock()
    lr_policy = MagicMock()
    imitation_policy = MagicMock()
    entropy_policy = MagicMock()
    presenter = MagicMock()
    league_maintenance = MagicMock()
    sample_seats = MagicMock()

    lr_policy.compute.return_value = 1.23e-4
    imitation_policy.compute.return_value = 0.77
    entropy_policy.compute.return_value = 0.015
    sample_seats.execute.return_value = "nn,bot_v5,nn,bot_v6"
    league_maintenance.execute.return_value = 1337.0
    runner.run_iteration.return_value = IterationResult(
        iteration=2,
        placement=2.4,
        loss=0.5,
        policy_loss=0.3,
        value_loss=0.2,
        entropy=0.9,
        n_experiences=256,
        selfplay_time=1.0,
        ppo_time=2.0,
        bench_time=0.5,
        seat_config_used="nn,bot_v5,nn,bot_v6",
        seat_outcomes={1: (2, 0, 0)},
    )

    ctx = TrainingContext(
        run=TrainingRun(config=base_config, identity=identity, initial_placement=3.1, start_time=0.0),
        pool=LeaguePool(config=base_config.league),
        last_snapshot_elo=1200.0,
        ts_path=str(Path(base_config.save_dir) / "_current.pt"),
        save_dir=Path(base_config.save_dir),
        league_pool_dir=Path(base_config.save_dir) / "league_pool",
    )
    ctx.pool.learner_elo = 900.0

    AdvanceIteration(
        iteration_runner=runner,
        lr_policy=lr_policy,
        imitation_policy=imitation_policy,
        entropy_policy=entropy_policy,
        presenter=presenter,
        league_maintenance=league_maintenance,
        sample_seats=sample_seats,
    ).execute(ctx, 2)

    run_call = runner.run_iteration.call_args
    assert run_call.args[:5] == (
        2,
        base_config,
        identity,
        str(Path(base_config.save_dir) / "_current.pt"),
        Path(base_config.save_dir),
    )
    assert run_call.kwargs["prev_placement"] == pytest.approx(3.1)
    assert run_call.kwargs["iter_lr"] == pytest.approx(1.23e-4)
    assert run_call.kwargs["iter_imitation_coef"] == pytest.approx(0.77)
    assert run_call.kwargs["iter_entropy_coef"] == pytest.approx(0.015)
    assert run_call.kwargs["seats_override"] == "nn,bot_v5,nn,bot_v6"
    assert run_call.kwargs["run_benchmark"] is True

    assert len(ctx.run.results) == 1
    assert ctx.run.results[0].learner_elo == pytest.approx(900.0)
    assert ctx.last_snapshot_elo == pytest.approx(1337.0)
    presenter.on_iteration_done.assert_called_once_with(3.1, 2.4, 3.5)

    maintenance_call = league_maintenance.execute.call_args
    assert maintenance_call.kwargs["pool"] is ctx.pool
    assert maintenance_call.kwargs["result"].learner_elo == pytest.approx(900.0)
    assert maintenance_call.kwargs["last_snapshot_elo"] == pytest.approx(1200.0)


def test_advance_iteration_uses_latest_result_as_previous_placement(
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    runner = MagicMock()
    runner.run_iteration.return_value = IterationResult(
        iteration=3,
        placement=2.2,
        loss=0.4,
        policy_loss=0.2,
        value_loss=0.2,
        entropy=0.8,
        n_experiences=128,
        selfplay_time=1.0,
        ppo_time=1.0,
        bench_time=0.0,
    )

    ctx = TrainingContext(
        run=TrainingRun(
            config=base_config,
            identity=identity,
            initial_placement=3.1,
            results=[
                IterationResult(
                    iteration=1,
                    placement=2.6,
                    loss=0.7,
                    policy_loss=0.4,
                    value_loss=0.3,
                    entropy=0.9,
                    n_experiences=128,
                    selfplay_time=1.0,
                    ppo_time=1.0,
                    bench_time=0.0,
                )
            ],
            start_time=0.0,
        ),
        pool=LeaguePool(config=base_config.league),
        last_snapshot_elo=None,
        ts_path=str(Path(base_config.save_dir) / "_current.pt"),
        save_dir=Path(base_config.save_dir),
        league_pool_dir=Path(base_config.save_dir) / "league_pool",
    )

    AdvanceIteration(
        iteration_runner=runner,
        lr_policy=MagicMock(compute=MagicMock(return_value=0.001)),
        imitation_policy=MagicMock(compute=MagicMock(return_value=0.0)),
        entropy_policy=MagicMock(compute=MagicMock(return_value=0.01)),
        presenter=MagicMock(),
        league_maintenance=MagicMock(execute=MagicMock(return_value=None)),
        sample_seats=MagicMock(execute=MagicMock(return_value="nn,nn,bot_v5,bot_v6")),
    ).execute(ctx, 3)

    assert runner.run_iteration.call_args.kwargs["prev_placement"] == pytest.approx(2.6)
    assert runner.run_iteration.call_args.kwargs["run_benchmark"] is False


@pytest.mark.parametrize(
    ("metric", "results", "expected_iter"),
    [
        (
            "loss",
            [
                IterationResult(1, 2.8, 0.7, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0),
                IterationResult(2, 2.4, 0.3, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0),
            ],
            2,
        ),
        (
            "elo",
            [
                IterationResult(1, 2.8, 0.7, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, learner_elo=900.0),
                IterationResult(2, 2.4, 0.5, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, learner_elo=950.0),
            ],
            2,
        ),
        (
            "placement",
            [
                IterationResult(1, 2.9, 0.7, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0),
                IterationResult(2, 2.2, 0.6, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0),
            ],
            2,
        ),
    ],
)
def test_promote_best_checkpoint_selects_configured_metric(
    base_config: TrainingConfig,
    identity: ModelIdentity,
    metric: str,
    results: list[IterationResult],
    expected_iter: int,
) -> None:
    model = MagicMock()
    run = TrainingRun(
        config=replace(base_config, best_model_metric=metric),
        identity=identity,
        initial_placement=3.1,
        results=results,
    )
    save_dir = Path(base_config.save_dir)

    PromoteBestCheckpoint(model).execute(run, save_dir)

    model.copy_best.assert_called_once_with(
        str(save_dir / f"iter_{expected_iter:03d}.pt"),
        str(save_dir / "best.pt"),
    )


def test_promote_best_checkpoint_skips_empty_runs(
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    model = MagicMock()
    run = TrainingRun(config=base_config, identity=identity)

    PromoteBestCheckpoint(model).execute(run, Path(base_config.save_dir))

    model.copy_best.assert_not_called()