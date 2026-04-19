"""Boundary-condition tests for Elo-driven schedules used in league play."""

from __future__ import annotations

import math

import pytest

from training.entities.training_config import TrainingConfig
from training.use_cases.train_model.policies import (
    EloDecayEntropyPolicy,
    EloGaussianILPolicy,
    elo_based_lr,
)


# ---------------------------------------------------------------------------
# elo_based_lr — the shared decay helper
# ---------------------------------------------------------------------------


class TestEloBasedLR:
    def test_returns_base_lr_when_disabled(self) -> None:
        assert elo_based_lr(current_elo=1500.0, base_lr=0.0, min_lr=1e-5) == pytest.approx(0.0)

    def test_clamps_min_above_base(self) -> None:
        # When min_lr >= base_lr the caller misconfigured; we return min_lr to
        # avoid producing values below the configured floor.
        assert elo_based_lr(1500.0, base_lr=1e-5, min_lr=1e-4) == pytest.approx(1e-4)

    def test_floor_elo_maps_to_base_lr(self) -> None:
        # At or below floor_elo, no decay has happened yet.
        assert elo_based_lr(800.0, base_lr=1e-3, min_lr=1e-5) == pytest.approx(1e-3)
        assert elo_based_lr(500.0, base_lr=1e-3, min_lr=1e-5) == pytest.approx(1e-3)

    def test_ceiling_elo_maps_to_min_lr(self) -> None:
        assert elo_based_lr(2000.0, base_lr=1e-3, min_lr=1e-5) == pytest.approx(1e-5)
        assert elo_based_lr(3000.0, base_lr=1e-3, min_lr=1e-5) == pytest.approx(1e-5)

    def test_midpoint_is_geometric_mean(self) -> None:
        # Power-law interpolation between base and min at progress=0.5.
        mid = elo_based_lr(1400.0, base_lr=1e-3, min_lr=1e-5)  # 1400 is halfway between 800/2000
        assert mid == pytest.approx(math.sqrt(1e-3 * 1e-5), rel=1e-6)


# ---------------------------------------------------------------------------
# EloDecayEntropyPolicy
# ---------------------------------------------------------------------------


def _config(**overrides) -> TrainingConfig:
    defaults = dict(
        model_arch="v4",
        save_dir="/tmp",
        iterations=1,
        bench_games=5,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TestEloDecayEntropyPolicy:
    def test_first_call_seeds_smoothed_elo_to_observation(self) -> None:
        policy = EloDecayEntropyPolicy(alpha=0.05)
        cfg = _config(entropy_coef=0.02, entropy_coef_min=1e-4)

        value = policy.compute(cfg, iteration=1, learner_elo=800.0)
        # At floor Elo the policy should return the base coefficient unchanged.
        assert value == pytest.approx(0.02)
        assert policy._smoothed_elo == pytest.approx(800.0)

    def test_smoothing_is_an_ema_with_configured_alpha(self) -> None:
        policy = EloDecayEntropyPolicy(alpha=0.1)
        cfg = _config(entropy_coef=0.02, entropy_coef_min=1e-4)

        policy.compute(cfg, iteration=1, learner_elo=1000.0)
        policy.compute(cfg, iteration=2, learner_elo=1400.0)

        expected = 0.1 * 1400.0 + 0.9 * 1000.0
        assert policy._smoothed_elo == pytest.approx(expected)

    def test_min_is_floored_when_config_zero(self) -> None:
        # When min is zero the policy replaces it with 1e-6 to keep the power
        # law defined. A very high Elo therefore cannot collapse to literal 0.
        policy = EloDecayEntropyPolicy()
        cfg = _config(entropy_coef=0.01, entropy_coef_min=0.0)

        value = policy.compute(cfg, iteration=1, learner_elo=2000.0)
        assert value == pytest.approx(1e-6)

    def test_decays_monotonically_as_elo_increases(self) -> None:
        cfg = _config(entropy_coef=0.02, entropy_coef_min=1e-4)
        policy = EloDecayEntropyPolicy(alpha=1.0)  # alpha=1 → no smoothing lag

        low = policy.compute(cfg, iteration=1, learner_elo=900.0)
        policy._smoothed_elo = None
        high = policy.compute(cfg, iteration=2, learner_elo=1800.0)

        assert low > high


# ---------------------------------------------------------------------------
# EloGaussianILPolicy
# ---------------------------------------------------------------------------


class TestEloGaussianILPolicy:
    def test_peak_at_centre_elo_returns_full_imitation_coef(self) -> None:
        cfg = _config(
            imitation_coef=0.3,
            imitation_center_elo=1500.0,
            imitation_width_elo=250.0,
        )
        policy = EloGaussianILPolicy(alpha=1.0, floor=0.0)

        value = policy.compute(cfg, iteration=1, learner_elo=1500.0)
        assert value == pytest.approx(0.3)

    def test_one_width_from_centre_decays_by_gaussian(self) -> None:
        cfg = _config(
            imitation_coef=0.3,
            imitation_center_elo=1500.0,
            imitation_width_elo=250.0,
        )
        policy = EloGaussianILPolicy(alpha=1.0, floor=0.0)

        value = policy.compute(cfg, iteration=1, learner_elo=1750.0)  # centre + 1σ
        expected = 0.3 * math.exp(-0.5)
        assert value == pytest.approx(expected, rel=1e-6)

    def test_below_floor_clamps_to_zero(self) -> None:
        # At Elo several widths away the Gaussian tail is below `floor`, so the
        # policy should return exactly 0.0 (not noise around the threshold).
        cfg = _config(
            imitation_coef=0.3,
            imitation_center_elo=1500.0,
            imitation_width_elo=100.0,
        )
        policy = EloGaussianILPolicy(alpha=1.0, floor=0.01)

        value = policy.compute(cfg, iteration=1, learner_elo=2500.0)
        assert value == 0.0

    def test_symmetric_around_centre(self) -> None:
        cfg = _config(
            imitation_coef=0.3,
            imitation_center_elo=1500.0,
            imitation_width_elo=300.0,
        )
        policy = EloGaussianILPolicy(alpha=1.0, floor=0.0)

        below = policy.compute(cfg, iteration=1, learner_elo=1350.0)  # centre - 150
        policy._smoothed_elo = None
        above = policy.compute(cfg, iteration=1, learner_elo=1650.0)  # centre + 150

        assert below == pytest.approx(above, rel=1e-6)

    def test_smoothing_delays_response_to_elo_spikes(self) -> None:
        cfg = _config(
            imitation_coef=0.3,
            imitation_center_elo=1500.0,
            imitation_width_elo=200.0,
        )
        policy = EloGaussianILPolicy(alpha=0.1, floor=0.0)

        # Seed smoothed_elo near floor; a single high-elo spike should not jump
        # the policy straight to the bell-curve peak.
        policy.compute(cfg, iteration=1, learner_elo=900.0)
        smoothed_before = policy._smoothed_elo

        spike = policy.compute(cfg, iteration=2, learner_elo=1500.0)
        smoothed_after = policy._smoothed_elo

        # After one alpha=0.1 step the smoothed Elo only travels 10% of the way
        # toward the spike value, so the returned coef is well below the peak.
        assert smoothed_after > smoothed_before
        assert smoothed_after < 1500.0
        assert spike < 0.3
