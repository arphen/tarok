"""Tests for ConfigurableIterationRunner mode dispatch and teardown lifecycle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

# Import torch before tarok_engine so the native Torch libraries are resolved
# via the torch wheel's RPATH before the tarok_engine extension tries to load.
import torch  # noqa: F401
import pytest

from training.adapters.iteration_runners.configurable import ConfigurableIterationRunner
from training.adapters.iteration_runners.in_process import InProcessIterationRunner
from training.adapters.iteration_runners.spawn import SpawnIterationRunner
from training.entities.duplicate_config import DuplicateConfig
from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig


def _runner(
    *,
    ppo: MagicMock | None = None,
) -> ConfigurableIterationRunner:
    ppo = ppo or MagicMock()
    return ConfigurableIterationRunner(
        selfplay=MagicMock(),
        ppo=ppo,
        benchmark=MagicMock(),
        model=MagicMock(),
        presenter=MagicMock(),
    )


def _base_config(mode: str, tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        model_arch="v4",
        save_dir=str(tmp_path / "run"),
        iterations=1,
        bench_games=8,
        iteration_runner_mode=mode,
        iteration_runner_restart_every=5,
    )


def _identity() -> ModelIdentity:
    return ModelIdentity(
        name="x", hidden_size=8, oracle_critic=False, model_arch="v4", is_new=True,
    )


def test_setup_in_process_mode_selects_in_process_delegate(tmp_path: Path) -> None:
    ppo = MagicMock()
    runner = _runner(ppo=ppo)

    runner.setup({"w": 1.0}, _base_config("in-process", tmp_path), "cpu")

    assert isinstance(runner._delegate, InProcessIterationRunner)
    ppo.setup.assert_called_once()


@pytest.mark.parametrize("mode", ["spawn", "process", "subprocess", "SPAWN", " Spawn "])
def test_setup_spawn_mode_selects_spawn_delegate(tmp_path: Path, mode: str) -> None:
    ppo = MagicMock()
    runner = _runner(ppo=ppo)

    runner.setup({"w": 1.0}, _base_config(mode, tmp_path), "cpu")

    assert isinstance(runner._delegate, SpawnIterationRunner)
    # Spawn does not eagerly boot PPO — weights are stashed until first iteration.
    ppo.setup.assert_not_called()


def test_setup_unknown_mode_falls_back_to_in_process(tmp_path: Path) -> None:
    ppo = MagicMock()
    runner = _runner(ppo=ppo)

    runner.setup({"w": 1.0}, _base_config("", tmp_path), "cpu")

    assert isinstance(runner._delegate, InProcessIterationRunner)


def test_run_iteration_raises_before_setup(tmp_path: Path) -> None:
    runner = _runner()

    with pytest.raises(RuntimeError, match="setup\\(\\) must be called first"):
        runner.run_iteration(
            1,
            _base_config("in-process", tmp_path),
            _identity(),
            str(tmp_path / "ts.pt"),
            tmp_path,
            prev_placement=2.5,
            iter_lr=None,
            iter_imitation_coef=None,
            iter_behavioral_clone_coef=None,
            iter_entropy_coef=None,
            seats_override=None,
            run_benchmark=False,
        )


def test_run_iteration_forwards_all_kwargs_to_delegate(tmp_path: Path) -> None:
    runner = _runner()
    runner.setup({"w": 1.0}, _base_config("in-process", tmp_path), "cpu")

    fake_delegate = MagicMock()
    fake_delegate.run_iteration.return_value = IterationResult(
        iteration=7, placement=2.1, loss=0.1, policy_loss=0.05, value_loss=0.04,
        entropy=0.01, n_experiences=16, selfplay_time=0.1, ppo_time=0.1, bench_time=0.0,
    )
    runner._delegate = fake_delegate

    result = runner.run_iteration(
        7,
        _base_config("in-process", tmp_path),
        _identity(),
        str(tmp_path / "ts.pt"),
        tmp_path,
        prev_placement=2.3,
        iter_lr=1e-4,
        iter_imitation_coef=0.5,
        iter_behavioral_clone_coef=0.25,
        iter_entropy_coef=0.02,
        seats_override="nn,bot_v5,bot_v5,bot_v5",
        run_benchmark=True,
    )

    assert result.iteration == 7
    call_kwargs = fake_delegate.run_iteration.call_args.kwargs
    assert call_kwargs["prev_placement"] == pytest.approx(2.3)
    assert call_kwargs["iter_lr"] == pytest.approx(1e-4)
    assert call_kwargs["iter_imitation_coef"] == pytest.approx(0.5)
    assert call_kwargs["iter_behavioral_clone_coef"] == pytest.approx(0.25)
    assert call_kwargs["iter_entropy_coef"] == pytest.approx(0.02)
    assert call_kwargs["seats_override"] == "nn,bot_v5,bot_v5,bot_v5"
    assert call_kwargs["run_benchmark"] is True


def test_teardown_releases_delegate_and_is_idempotent(tmp_path: Path) -> None:
    runner = _runner()
    runner.setup({"w": 1.0}, _base_config("in-process", tmp_path), "cpu")

    fake_delegate = MagicMock()
    runner._delegate = fake_delegate

    runner.teardown()
    assert runner._delegate is None
    fake_delegate.teardown.assert_called_once()

    # Second teardown is a no-op — no delegate, no crash.
    runner.teardown()
    assert runner._delegate is None


# ----------------------------------------------------------------------
# Duplicate-RL spawn support (docs/double_rl.md Phase 3)
# ----------------------------------------------------------------------


def _duplicate_config(tmp_path: Path, *, enabled: bool) -> TrainingConfig:
    return TrainingConfig(
        model_arch="v4",
        save_dir=str(tmp_path / "run"),
        iterations=1,
        bench_games=8,
        iteration_runner_mode="spawn",
        iteration_runner_restart_every=5,
        duplicate=DuplicateConfig(enabled=enabled, pods_per_iteration=2),
    )


def test_spawn_mode_supports_duplicate_without_raising(tmp_path: Path) -> None:
    """Regression: duplicate + spawn used to raise NotImplementedError."""
    runner = _runner()

    runner.setup({"w": 1.0}, _duplicate_config(tmp_path, enabled=True), "cpu")

    assert isinstance(runner._delegate, SpawnIterationRunner)


def test_adapter_factory_for_spawn_constructs_duplicate_ports_when_enabled(
    tmp_path: Path,
) -> None:
    from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
    from training.adapters.duplicate.seeded_self_play_adapter import SeededSelfPlayAdapter
    from training.adapters.duplicate.shadow_score_reward import ShadowScoreRewardAdapter
    from training.adapters.duplicate.shadow_sources import (
        PreviousIterationShadowSource,
    )

    runner = _runner()
    config = _duplicate_config(tmp_path, enabled=True)

    (
        selfplay,
        _ppo,
        _bench,
        _model,
        dup_pairing,
        dup_reward,
        dup_shadow,
        dup_stats,
    ) = runner._adapter_factory_for_spawn(config)

    assert isinstance(selfplay, SeededSelfPlayAdapter)
    assert isinstance(dup_pairing, RotationPairingAdapter)
    assert isinstance(dup_reward, ShadowScoreRewardAdapter)
    assert isinstance(dup_shadow, PreviousIterationShadowSource)
    from training.adapters.duplicate.numpy_iteration_stats import NumpyDuplicateIterationStats
    assert isinstance(dup_stats, NumpyDuplicateIterationStats)


def test_adapter_factory_for_spawn_leaves_duplicate_ports_none_when_disabled(
    tmp_path: Path,
) -> None:
    from training.adapters.duplicate.seeded_self_play_adapter import SeededSelfPlayAdapter

    runner = _runner()
    config = _duplicate_config(tmp_path, enabled=False)

    (
        selfplay,
        _ppo,
        _bench,
        _model,
        dup_pairing,
        dup_reward,
        dup_shadow,
        dup_stats,
    ) = runner._adapter_factory_for_spawn(config)

    assert not isinstance(selfplay, SeededSelfPlayAdapter)
    assert dup_pairing is None
    assert dup_reward is None
    assert dup_shadow is None
    assert dup_stats is None
