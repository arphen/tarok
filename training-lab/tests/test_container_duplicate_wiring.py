"""Wiring tests for the DI container's duplicate-RL path.

These verify that enabling ``config.duplicate.enabled=True`` causes the
container to construct the iteration runner with the duplicate pairing and
reward adapters injected (and wraps self-play in SeededSelfPlayAdapter),
while the default path remains unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
from training.adapters.duplicate.seeded_self_play_adapter import SeededSelfPlayAdapter
from training.adapters.duplicate.shadow_score_reward import ShadowScoreRewardAdapter
from training.container import _default_iteration_runner
from training.entities.duplicate_config import DuplicateConfig
from training.entities.training_config import TrainingConfig


def _make_config(duplicate_enabled: bool) -> TrainingConfig:
    return TrainingConfig(
        model_arch="v4",
        save_dir="/tmp/run",
        iterations=1,
        bench_games=1,
        games=1,
        concurrency=1,
        explore_rate=0.0,
        batch_size=16,
        ppo_epochs=1,
        duplicate=DuplicateConfig(
            enabled=duplicate_enabled,
            pods_per_iteration=2,
            negative_reward_multiplier=3.0,
            berac_bid_penalty=-10.0,
        ),
    )


def test_iteration_runner_factory_wires_duplicate_adapters_when_enabled():
    selfplay = MagicMock()
    cfg = _make_config(duplicate_enabled=True)

    runner = _default_iteration_runner(
        selfplay, MagicMock(), MagicMock(), MagicMock(), MagicMock(), config=cfg,
    )

    # Self-play was wrapped so run_seeded_pods is available.
    assert isinstance(runner._selfplay, SeededSelfPlayAdapter)  # type: ignore[attr-defined]
    assert isinstance(runner._duplicate_pairing, RotationPairingAdapter)  # type: ignore[attr-defined]
    assert isinstance(runner._duplicate_reward, ShadowScoreRewardAdapter)  # type: ignore[attr-defined]
    assert runner._duplicate_reward._negative_reward_multiplier == 3.0  # type: ignore[attr-defined]
    assert runner._duplicate_reward._berac_bid_penalty == -10.0  # type: ignore[attr-defined]


def test_iteration_runner_factory_skips_duplicate_when_disabled():
    selfplay = MagicMock()
    cfg = _make_config(duplicate_enabled=False)

    runner = _default_iteration_runner(
        selfplay, MagicMock(), MagicMock(), MagicMock(), MagicMock(), config=cfg,
    )

    assert runner._selfplay is selfplay  # type: ignore[attr-defined]
    assert runner._duplicate_pairing is None  # type: ignore[attr-defined]
    assert runner._duplicate_reward is None  # type: ignore[attr-defined]


def test_iteration_runner_factory_no_config_leaves_selfplay_intact():
    selfplay = MagicMock()
    runner = _default_iteration_runner(
        selfplay, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
    )
    assert runner._selfplay is selfplay  # type: ignore[attr-defined]
    assert runner._duplicate_pairing is None  # type: ignore[attr-defined]
