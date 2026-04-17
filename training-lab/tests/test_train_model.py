"""Tests for the TrainModel use case orchestrator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.entities.iteration_result import IterationResult
from training.entities.league import LeagueConfig
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.use_cases.train_model import TrainModel


@pytest.fixture
def mock_iteration_runner() -> MagicMock:
    runner = MagicMock()
    runner.run_iteration.return_value = IterationResult(
        iteration=1,
        placement=2.5,
        loss=0.42,
        policy_loss=0.31,
        value_loss=0.27,
        entropy=0.9,
        n_experiences=128,
        selfplay_time=1.0,
        ppo_time=2.0,
        bench_time=0.5,
        seat_config_used="nn,bot_v5,bot_v5,bot_v5",
        seat_outcomes={1: (10, 0, 0)},
    )
    return runner


@pytest.fixture
def mock_benchmark() -> MagicMock:
    bench = MagicMock()
    bench.measure_placement.return_value = 3.1
    return bench


@pytest.fixture
def mock_model_port() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_presenter() -> MagicMock:
    return MagicMock()


@pytest.fixture
def base_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        model_arch="v4",
        save_dir=str(tmp_path / "test_run"),
        iterations=2,
        bench_games=10,
        benchmark_checkpoints=(),
        memory_telemetry=False,
        league=None,
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


@patch("training.use_cases.train_model._collect_memory_stats")
def test_train_model_basic_execution_flow(
    mock_mem_stats: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    """Verifies setup -> benchmark -> iterations -> teardown flow without league."""
    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
    )

    run_result = use_case.execute(
        config=base_config,
        identity=identity,
        weights={"dummy": "weights"},
        device="cpu",
    )

    # Setup / initial benchmark
    mock_model_port.export_for_inference.assert_called_once()
    mock_iteration_runner.setup.assert_called_once()
    mock_benchmark.measure_placement.assert_called_once()

    # 2 iterations requested
    assert mock_iteration_runner.run_iteration.call_count == 2
    assert len(run_result.results) == 2

    # No memory telemetry calls (disabled in fixture)
    mock_mem_stats.assert_not_called()

    # Always tears down and finalizes run
    mock_iteration_runner.teardown.assert_called_once()
    mock_model_port.copy_best.assert_called_once()
    mock_presenter.on_training_complete.assert_called_once_with(run_result)


@patch("training.use_cases.train_model.UpdateLeagueElo")
@patch("training.use_cases.train_model.SampleLeagueSeats")
@patch("training.use_cases.train_model.shutil.copy2")
@patch("training.use_cases.train_model._collect_memory_stats")
def test_train_model_with_league_and_snapshots(
    mock_mem_stats: MagicMock,
    mock_copy2: MagicMock,
    MockSampleSeats: MagicMock,
    MockUpdateElo: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    """Verifies league sampling, Elo updates, and snapshot cadence behavior."""
    cfg = replace(base_config, league=LeagueConfig(enabled=True, snapshot_interval=2))

    mock_sampler = MockSampleSeats.return_value
    mock_sampler.execute.return_value = "nn,bot_m6,bot_v5,nn"

    # Two distinct results for two iterations.
    mock_iteration_runner.run_iteration.side_effect = [
        IterationResult(
            iteration=1,
            placement=2.8,
            loss=0.8,
            policy_loss=0.5,
            value_loss=0.3,
            entropy=1.0,
            n_experiences=256,
            selfplay_time=1.0,
            ppo_time=1.0,
            bench_time=0.0,
            seat_config_used="nn,bot_m6,bot_v5,nn",
            seat_outcomes={1: (1, 0, 0), 2: (0, 1, 0)},
        ),
        IterationResult(
            iteration=2,
            placement=2.6,
            loss=0.6,
            policy_loss=0.4,
            value_loss=0.2,
            entropy=0.9,
            n_experiences=256,
            selfplay_time=1.0,
            ppo_time=1.0,
            bench_time=0.0,
            seat_config_used="nn,bot_m6,bot_v5,nn",
            seat_outcomes={1: (1, 0, 0), 2: (0, 1, 0)},
        ),
    ]

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
    )

    use_case.execute(config=cfg, identity=identity, weights={}, device="cpu")

    # League sampling and ELO update happen every iteration.
    assert mock_sampler.execute.call_count == 2
    mock_updater = MockUpdateElo.return_value
    assert mock_updater.execute.call_count == 2

    # Snapshot interval=2 => exactly one snapshot on iteration 2.
    save_dir = Path(cfg.save_dir)
    expected_snap = save_dir / "league_pool" / "iter_002.pt"

    mock_copy2.assert_called_once_with(str(save_dir / "_current.pt"), str(expected_snap))
    mock_presenter.on_league_snapshot_added.assert_called_once_with(2, str(expected_snap))

    # Teardown still called in successful path.
    mock_iteration_runner.teardown.assert_called_once()
    mock_mem_stats.assert_not_called()


@patch("training.use_cases.train_model._collect_memory_stats")
def test_train_model_ensures_teardown_on_error(
    mock_mem_stats: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    """Verifies teardown is always called if an iteration raises."""
    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
    )

    mock_iteration_runner.run_iteration.side_effect = RuntimeError("CUDA Out of Memory")

    with pytest.raises(RuntimeError, match="CUDA Out of Memory"):
        use_case.execute(config=base_config, identity=identity, weights={}, device="cpu")

    mock_iteration_runner.teardown.assert_called_once()

    # No successful completion path after crash.
    mock_model_port.copy_best.assert_not_called()
    mock_presenter.on_training_complete.assert_not_called()
    mock_mem_stats.assert_not_called()
