"""Use case: configure PPO hyperparameters and run one policy update."""

from __future__ import annotations

import time
from typing import Any

from training.entities.policy_update_result import PolicyUpdateResult
from training.entities.training_config import TrainingConfig
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort


class UpdatePolicy:
    """Apply one PPO gradient update using the collected experience data.

    Single responsibility: set any iteration-specific hyperparameter overrides,
    call the PPO update, and return the new weights together with metrics.
    """

    def __init__(self, ppo: PPOPort, presenter: PresenterPort) -> None:
        self._ppo = ppo
        self._presenter = presenter

    def execute(
        self,
        raw: Any,
        nn_seats: list[int],
        config: TrainingConfig,
        iter_lr: float | None = None,
        iter_imitation_coef: float | None = None,
        iter_entropy_coef: float | None = None,
    ) -> PolicyUpdateResult:
        if iter_lr is not None:
            self._ppo.set_lr(iter_lr)
        if iter_imitation_coef is not None:
            self._ppo.set_imitation_coef(iter_imitation_coef)
        if iter_entropy_coef is not None:
            self._ppo.set_entropy_coef(iter_entropy_coef)

        self._presenter.on_ppo_start(
            config,
            iter_lr=iter_lr,
            iter_imitation_coef=iter_imitation_coef,
            iter_entropy_coef=iter_entropy_coef,
        )

        t0 = time.time()
        metrics, new_weights = self._ppo.update(raw, nn_seats)
        ppo_time = time.time() - t0

        self._presenter.on_ppo_done(metrics, ppo_time)
        return PolicyUpdateResult(new_weights=new_weights, metrics=metrics, ppo_time=ppo_time)
