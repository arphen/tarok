"""Use case: select and promote the best training checkpoint after the loop."""

from __future__ import annotations

from pathlib import Path

from training.entities.training_run import TrainingRun
from training.ports.model_port import ModelPort


class PromoteBestCheckpoint:
    """Copy the iteration checkpoint with the best metric to 'best.pt'.

    Single responsibility: apply the configured selection metric, locate the
    correct source checkpoint, and delegate the file copy to ModelPort.
    """

    def __init__(self, model: ModelPort) -> None:
        self._model = model

    def execute(self, run: TrainingRun, save_dir: Path) -> None:
        if not run.results:
            return

        if run.config.best_model_metric == "loss":
            best_iter = run.best_loss_iteration
        elif run.config.best_model_metric == "elo":
            best_iter = run.best_elo_iteration
        else:
            best_iter = run.best_iteration

        best_src = str(save_dir / f"iter_{best_iter:03d}.pt")
        best_dst = str(save_dir / "best.pt")
        self._model.copy_best(best_src, best_dst)
