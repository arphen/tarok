"""Composition tests for RunIteration.

RunIteration is a thin orchestrator that wires five use cases together:
  CollectExperiences -> UpdatePolicy -> ExportModel -> MeasurePlacement -> SaveCheckpoint

These tests verify the wiring — not the behaviour of each use case — by stubbing
out all ports with MagicMocks and asserting the call sequence and argument flow.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.use_cases.run_iteration import RunIteration


@pytest.fixture
def config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        model_arch="v4",
        save_dir=str(tmp_path / "run"),
        iterations=2,
        bench_games=8,
        games=4,
        concurrency=1,
        explore_rate=0.0,
        batch_size=16,
        ppo_epochs=1,
    )


@pytest.fixture
def identity() -> ModelIdentity:
    return ModelIdentity(
        name="Test",
        hidden_size=16,
        oracle_critic=False,
        model_arch="v4",
        is_new=True,
    )


def _selfplay_mock() -> MagicMock:
    sp = MagicMock()
    sp.run.return_value = {"players": list(range(4)), "states": [], "actions": []}
    sp.compute_run_stats.return_value = (
        2,
        (1.0, -0.5, -0.5, 0.0),
        {1: (2, 1, 1), 2: (1, 2, 1), 3: (1, 1, 2)},
    )
    return sp


def _ppo_mock() -> MagicMock:
    ppo = MagicMock()
    ppo.update.return_value = (
        {"total_loss": 0.31, "policy_loss": 0.10, "value_loss": 0.15, "entropy": 0.06},
        {"w": 42.0},
    )
    ppo.load_human_data.return_value = None
    ppo.load_expert_data.return_value = None
    ppo.merge_experiences.side_effect = lambda a, b: {**a, **b}
    return ppo


def test_run_iteration_returns_result_and_new_weights(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    selfplay = _selfplay_mock()
    ppo = _ppo_mock()
    benchmark = MagicMock()
    benchmark.measure_placement.return_value = 1.7
    model = MagicMock()
    presenter = MagicMock()

    uc = RunIteration(selfplay, ppo, benchmark, model, presenter)
    result, new_weights = uc.execute(
        iteration=3,
        config=config,
        identity=identity,
        ts_path=str(tmp_path / "_current.pt"),
        save_dir=tmp_path,
        prev_placement=2.5,
        iter_lr=1e-4,
        iter_imitation_coef=0.1,
        iter_behavioral_clone_coef=0.0,
        iter_entropy_coef=0.02,
        seats_override="nn,bot_v5,bot_v5,bot_v5",
        run_benchmark=True,
    )

    assert new_weights == {"w": 42.0}
    assert result.iteration == 3
    assert result.loss == pytest.approx(0.31)
    assert result.placement == pytest.approx(1.7)


def test_run_iteration_propagates_hyperparameter_overrides_to_ppo(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    selfplay = _selfplay_mock()
    ppo = _ppo_mock()
    benchmark = MagicMock()
    benchmark.measure_placement.return_value = 2.0
    model = MagicMock()

    RunIteration(selfplay, ppo, benchmark, model, MagicMock()).execute(
        iteration=1,
        config=config,
        identity=identity,
        ts_path=str(tmp_path / "_current.pt"),
        save_dir=tmp_path,
        prev_placement=2.5,
        iter_lr=5e-5,
        iter_imitation_coef=0.42,
        iter_behavioral_clone_coef=0.25,
        iter_entropy_coef=0.015,
    )

    ppo.set_lr.assert_called_once_with(5e-5)
    ppo.set_imitation_coef.assert_called_once_with(0.42)
    ppo.set_behavioral_clone_coef.assert_called_once_with(0.25)
    ppo.set_entropy_coef.assert_called_once_with(0.015)


def test_run_iteration_skips_benchmark_when_flag_is_false(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    selfplay = _selfplay_mock()
    ppo = _ppo_mock()
    benchmark = MagicMock()
    model = MagicMock()

    RunIteration(selfplay, ppo, benchmark, model, MagicMock()).execute(
        iteration=2,
        config=config,
        identity=identity,
        ts_path=str(tmp_path / "_current.pt"),
        save_dir=tmp_path,
        prev_placement=2.3,
        run_benchmark=False,
    )

    benchmark.measure_placement.assert_not_called()


def test_run_iteration_exports_model_with_identity_and_weights(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    selfplay = _selfplay_mock()
    ppo = _ppo_mock()
    benchmark = MagicMock()
    benchmark.measure_placement.return_value = 2.0
    model = MagicMock()

    ts_path = str(tmp_path / "_current.pt")
    RunIteration(selfplay, ppo, benchmark, model, MagicMock()).execute(
        iteration=5,
        config=config,
        identity=identity,
        ts_path=ts_path,
        save_dir=tmp_path,
        prev_placement=2.5,
    )

    args, _ = model.export_for_inference.call_args
    # (weights, hidden_size, oracle_critic, model_arch, path)
    assert args[0] == {"w": 42.0}
    assert args[1] == identity.hidden_size
    assert args[2] == identity.oracle_critic
    assert args[3] == identity.model_arch
    assert args[4] == ts_path


def test_run_iteration_saves_checkpoint_with_iteration_and_placement(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    selfplay = _selfplay_mock()
    ppo = _ppo_mock()
    benchmark = MagicMock()
    benchmark.measure_placement.return_value = 2.9
    model = MagicMock()

    RunIteration(selfplay, ppo, benchmark, model, MagicMock()).execute(
        iteration=9,
        config=config,
        identity=identity,
        ts_path=str(tmp_path / "_current.pt"),
        save_dir=tmp_path,
        prev_placement=2.8,
    )

    save_args = model.save_checkpoint.call_args[0]
    # Trailing args are (iteration, loss, placement, path)
    assert save_args[-4] == 9
    assert save_args[-3] == pytest.approx(0.31)
    assert save_args[-2] == pytest.approx(2.9)
    assert save_args[-1] == str(tmp_path / "iter_009.pt")


def test_run_iteration_calls_phases_in_order(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    call_log: list[str] = []
    selfplay = _selfplay_mock()
    selfplay.run.side_effect = lambda *a, **kw: (
        call_log.append("selfplay"),
        {"players": [0], "states": [], "actions": []},
    )[1]
    selfplay.compute_run_stats.return_value = (1, (0, 0, 0, 0), {})

    ppo = _ppo_mock()
    ppo.update.side_effect = lambda *a, **kw: (
        call_log.append("ppo"),
        ({"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}, {"w": 1.0}),
    )[1]

    benchmark = MagicMock()
    benchmark.measure_placement.side_effect = lambda *a, **kw: (call_log.append("benchmark"), 2.0)[1]

    model = MagicMock()
    model.export_for_inference.side_effect = lambda *a, **kw: call_log.append("export")
    model.save_checkpoint.side_effect = lambda *a, **kw: call_log.append("save")

    RunIteration(selfplay, ppo, benchmark, model, MagicMock()).execute(
        iteration=1,
        config=config,
        identity=identity,
        ts_path=str(tmp_path / "_current.pt"),
        save_dir=tmp_path,
        prev_placement=2.5,
    )

    assert call_log == ["selfplay", "ppo", "export", "benchmark", "save"]
