"""Default learning-rate policy adapter."""

from __future__ import annotations

from training.entities.training_config import TrainingConfig, scheduled_lr
from training.ports.learning_rate_policy_port import LearningRatePolicyPort


class LeagueEloLearningRatePolicy(LearningRatePolicyPort):
    """Compute LR from base schedule, then apply smooth Elo-based decay."""

    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float) -> float:
        base_lr = scheduled_lr(
            iteration - 1,
            config.iterations,
            config.lr,
            config.effective_lr_min,
            config.lr_schedule,
        )
        return _elo_based_lr(
            current_elo=learner_elo,
            base_lr=base_lr,
            min_lr=config.effective_lr_min,
        )


def _elo_based_lr(
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
