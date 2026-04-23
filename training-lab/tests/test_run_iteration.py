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


# ---------------------------------------------------------------------------
# Duplicate-RL branching (docs/double_rl.md §3.5)
# ---------------------------------------------------------------------------


def test_run_iteration_uses_legacy_collect_when_duplicate_disabled(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    """Default config has ``duplicate.enabled=False`` → legacy path only."""
    selfplay = _selfplay_mock()
    ppo = _ppo_mock()
    benchmark = MagicMock()
    benchmark.measure_placement.return_value = 2.0
    model = MagicMock()
    pairing = MagicMock()
    reward = MagicMock()

    RunIteration(
        selfplay, ppo, benchmark, model, MagicMock(),
        duplicate_pairing=pairing, duplicate_reward=reward,
    ).execute(
        iteration=1,
        config=config,
        identity=identity,
        ts_path=str(tmp_path / "_current.pt"),
        save_dir=tmp_path,
        prev_placement=2.5,
    )

    # Legacy selfplay.run called; duplicate ports untouched.
    assert selfplay.run.called
    assert not selfplay.run_seeded_pods.called
    assert not pairing.build_pods.called
    assert not reward.compute_rewards.called


def test_run_iteration_routes_to_duplicate_when_enabled(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    """When ``duplicate.enabled=True`` and ports are injected, the orchestrator
    must call ``run_seeded_pods`` / ``build_pods`` / ``compute_rewards`` and
    skip the legacy ``selfplay.run`` path."""
    import numpy as np

    import dataclasses

    from training.entities.duplicate_config import DuplicateConfig
    from training.entities.duplicate_pod import DuplicatePod
    from training.entities.duplicate_run_result import DuplicateRunResult

    config = dataclasses.replace(
        config, duplicate=DuplicateConfig(enabled=True, pods_per_iteration=2),
    )

    pod = DuplicatePod(
        deck_seed=0,
        opponents=("bot_v5", "bot_v5", "bot_v5"),
        active_seatings=((0, 1, 2, 3),) * 8,
        shadow_seatings=((0, 1, 2, 3),) * 8,
        learner_positions=(0,) * 8,
    )
    pods = [pod, pod]
    n_games = 16  # 2 pods * 8 games/group
    active_raw = {
        "players": np.zeros(n_games, dtype=np.int8),
        "game_ids": np.arange(n_games, dtype=np.int64),
        "states": np.zeros((n_games, 2), dtype=np.float32),
        "scores": np.zeros((n_games, 4), dtype=np.float32),
    }
    run_result = DuplicateRunResult(
        active=active_raw,
        shadow_scores=np.zeros((n_games, 4), dtype=np.float32),
        pod_ids=np.repeat(np.arange(2), 8).astype(np.int64),
        learner_positions=np.zeros((2, 8), dtype=np.int8),
        active_game_ids=np.arange(n_games, dtype=np.int64),
    )

    selfplay = _selfplay_mock()
    selfplay.run_seeded_pods.return_value = run_result
    selfplay.compute_run_stats.return_value = (n_games, (0.0, 0.0, 0.0, 0.0), {})

    pairing = MagicMock()
    pairing.build_pods.return_value = pods
    reward = MagicMock()
    reward.compute_rewards.return_value = np.zeros(n_games, dtype=np.float32)

    ppo = _ppo_mock()
    benchmark = MagicMock()
    benchmark.measure_placement.return_value = 2.0
    model = MagicMock()

    from training.adapters.duplicate.shadow_sources import (
        PreviousIterationShadowSource,
    )

    RunIteration(
        selfplay, ppo, benchmark, model, MagicMock(),
        duplicate_pairing=pairing, duplicate_reward=reward,
        duplicate_shadow_source=PreviousIterationShadowSource(),
    ).execute(
        iteration=1,
        config=config,
        identity=identity,
        ts_path=str(tmp_path / "_current.pt"),
        save_dir=tmp_path,
        prev_placement=2.5,
    )

    assert pairing.build_pods.called
    assert selfplay.run_seeded_pods.called
    assert reward.compute_rewards.called
    assert not selfplay.run.called  # legacy path bypassed


def test_run_iteration_falls_back_to_legacy_when_duplicate_ports_missing(
    config: TrainingConfig, identity: ModelIdentity, tmp_path: Path,
) -> None:
    """Config says enabled=True but no ports injected → safe fallback to legacy."""
    import dataclasses

    from training.entities.duplicate_config import DuplicateConfig

    config = dataclasses.replace(
        config, duplicate=DuplicateConfig(enabled=True, pods_per_iteration=2),
    )

    selfplay = _selfplay_mock()
    ppo = _ppo_mock()
    benchmark = MagicMock()
    benchmark.measure_placement.return_value = 2.0
    model = MagicMock()

    # No duplicate_pairing / duplicate_reward → must fall back.
    RunIteration(selfplay, ppo, benchmark, model, MagicMock()).execute(
        iteration=1,
        config=config,
        identity=identity,
        ts_path=str(tmp_path / "_current.pt"),
        save_dir=tmp_path,
        prev_placement=2.5,
    )

    assert selfplay.run.called
    assert not selfplay.run_seeded_pods.called
