"""Default imitation-coefficient policy adapter."""

from __future__ import annotations

from training.entities.training_config import TrainingConfig, scheduled_coef
from training.ports.imitation_coef_policy_port import ImitationCoefPolicyPort


class ScheduledImitationCoefPolicy(ImitationCoefPolicyPort):
    """Compute imitation coefficient from config values and schedule policy."""

    def compute(self, config: TrainingConfig, iteration: int, learner_elo: float = 0.0) -> float:
        return scheduled_coef(
            iteration=max(0, iteration - 1),
            total_iterations=config.iterations,
            coef_max=config.imitation_coef,
            coef_min=config.imitation_coef_min,
            schedule=config.imitation_schedule,
        )
