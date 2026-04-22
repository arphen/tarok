"""Default policy implementations used by TrainModel."""

from __future__ import annotations

import math

from training.entities.training_config import TrainingConfig, scheduled_coef, scheduled_lr
from training.ports.explore_rate_policy_port import ExploreRatePolicyPort
from training.ports.imitation_coef_policy_port import ImitationCoefPolicyPort
from training.ports.learning_rate_policy_port import LearningRatePolicyPort


class DefaultLearningRatePolicy(LearningRatePolicyPort):
    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float) -> float:
        base_lr = scheduled_lr(
            iteration - 1,
            config.iterations,
            config.lr,
            config.effective_lr_min,
            config.lr_schedule,
        )
        return elo_based_lr(
            current_elo=learner_elo,
            base_lr=base_lr,
            min_lr=config.effective_lr_min,
        )


class DefaultImitationCoefPolicy(ImitationCoefPolicyPort):
    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        return scheduled_coef(
            iteration=max(0, iteration - 1),
            total_iterations=config.iterations,
            coef_max=config.imitation_coef,
            coef_min=config.imitation_coef_min,
            schedule=config.imitation_schedule,
        )


class DefaultEntropyCoefPolicy:
    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        return scheduled_coef(
            iteration=max(0, iteration - 1),
            total_iterations=config.iterations,
            coef_max=config.entropy_coef,
            coef_min=config.entropy_coef_min,
            schedule=config.entropy_schedule,
        )


class DefaultBehavioralCloneCoefPolicy:
    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        return scheduled_coef(
            iteration=max(0, iteration - 1),
            total_iterations=config.iterations,
            coef_max=config.behavioral_clone_coef,
            coef_min=config.behavioral_clone_coef_min,
            schedule=config.behavioral_clone_schedule,
        )


class DefaultExploreRatePolicy(ExploreRatePolicyPort):
    """Iteration-scheduled explore rate (constant / linear / cosine).

    Mirrors :class:`DefaultEntropyCoefPolicy` so the classic non-Elo schedules
    remain available when ``explore_rate_schedule`` is set to one of those
    values.  Elo-aware scheduling is handled by :class:`EloDecayExplorePolicy`.
    """

    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        return scheduled_coef(
            iteration=max(0, iteration - 1),
            total_iterations=config.iterations,
            coef_max=config.explore_rate,
            coef_min=config.effective_explore_rate_min,
            schedule=config.explore_rate_schedule,
        )


class EloDecayEntropyPolicy:
    """Entropy coefficient that decays with Elo — full exploration at low skill, minimal at high.

    Uses the same power-law interpolation as elo_based_lr so entropy and LR stay
    in sync: both are highest when the agent is weakest and taper as it improves.
    EMA smoothing prevents thrashing on noisy Elo estimates.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._smoothed_elo: float | None = None

    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        if self._smoothed_elo is None:
            self._smoothed_elo = learner_elo
        else:
            self._smoothed_elo = self.alpha * learner_elo + (1.0 - self.alpha) * self._smoothed_elo
        return elo_based_lr(
            current_elo=self._smoothed_elo,
            base_lr=config.entropy_coef,
            min_lr=config.entropy_coef_min if config.entropy_coef_min > 0 else 1e-6,
        )


class EloGaussianILPolicy(ImitationCoefPolicyPort):
    """Gaussian bell curve IL coefficient centred on a target Elo.

    Keeps the tutor silent at low Elo, ramps up near the centre, then fades
    back to zero at high Elo — automatically managing the distillation curriculum.
    EMA smoothing prevents erratic jumps during unlucky Elo streaks.

    center_elo, width, and peak_il are read from config on every call so that
    YAML changes are always honoured without reconstructing the policy.
    """

    def __init__(self, alpha: float = 0.05, floor: float = 0.001) -> None:
        self.alpha = alpha
        self.floor = floor
        self._smoothed_elo: float | None = None

    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        if self._smoothed_elo is None:
            self._smoothed_elo = learner_elo
        else:
            self._smoothed_elo = self.alpha * learner_elo + (1.0 - self.alpha) * self._smoothed_elo
        exponent = -((self._smoothed_elo - config.imitation_center_elo) ** 2) / (
            2.0 * config.imitation_width_elo ** 2
        )
        coef = config.imitation_coef * math.exp(exponent)
        return 0.0 if coef < self.floor else coef


class EloDecayExplorePolicy(ExploreRatePolicyPort):
    """Explore rate (epsilon) that decays with learner Elo.

    Uses the same power-law interpolation as :func:`elo_based_lr` so epsilon
    stays in sync with the Elo-decayed LR and entropy schedules: maximum
    exploration at the Elo floor (800), minimal at the ceiling (2000).  EMA
    smoothing with the same ``alpha`` prevents thrashing on noisy Elo
    estimates.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._smoothed_elo: float | None = None

    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        if self._smoothed_elo is None:
            self._smoothed_elo = learner_elo
        else:
            self._smoothed_elo = self.alpha * learner_elo + (1.0 - self.alpha) * self._smoothed_elo
        min_rate = config.effective_explore_rate_min
        return elo_based_lr(
            current_elo=self._smoothed_elo,
            base_lr=config.explore_rate,
            min_lr=min_rate if min_rate > 0 else 1e-6,
        )


def elo_based_lr(
    current_elo: float,
    base_lr: float,
    min_lr: float,
    floor_elo: float = 800.0,
    ceiling_elo: float = 2000.0,
) -> float:
    if base_lr <= 0 or min_lr <= 0:
        return base_lr
    if min_lr >= base_lr:
        return min_lr
    clamped_elo = max(floor_elo, min(current_elo, ceiling_elo))
    progress = (clamped_elo - floor_elo) / (ceiling_elo - floor_elo)
    decay_ratio = min_lr / base_lr
    return base_lr * (decay_ratio ** progress)
