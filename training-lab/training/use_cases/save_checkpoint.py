"""Use case: persist a checkpoint and assemble the IterationResult."""

from __future__ import annotations

from pathlib import Path

from training.entities.experience_bundle import ExperienceBundle
from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.policy_update_result import PolicyUpdateResult
from training.ports.model_port import ModelPort


class SaveCheckpoint:
    """Write the iteration checkpoint and return the structured result record.

    Single responsibility: persist weights to disk and assemble ``IterationResult``
    from the outputs of all preceding use cases.  No compute, no I/O beyond the
    checkpoint file.
    """

    def __init__(self, model: ModelPort) -> None:
        self._model = model

    def execute(
        self,
        iteration: int,
        bundle: ExperienceBundle,
        update: PolicyUpdateResult,
        identity: ModelIdentity,
        save_dir: Path,
        placement: float,
        bench_time: float,
    ) -> IterationResult:
        ckpt_path = str(save_dir / f"iter_{iteration:03d}.pt")
        self._model.save_checkpoint(
            update.new_weights,
            identity.hidden_size,
            identity.oracle_critic,
            identity.model_arch,
            iteration,
            update.metrics["total_loss"],
            placement,
            ckpt_path,
        )
        return IterationResult(
            iteration=iteration,
            placement=placement,
            loss=update.metrics["total_loss"],
            policy_loss=update.metrics["policy_loss"],
            value_loss=update.metrics["value_loss"],
            entropy=update.metrics["entropy"],
            n_experiences=bundle.n_total,
            selfplay_time=bundle.sp_time,
            ppo_time=update.ppo_time,
            bench_time=bench_time,
            seat_config_used=bundle.effective_seats,
            mean_scores=bundle.mean_scores,
            seat_outcomes=bundle.seat_outcomes,
        )
