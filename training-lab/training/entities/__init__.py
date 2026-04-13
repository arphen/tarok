"""Entities — pure data, no dependencies on frameworks or I/O."""

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.names import name_from_checkpoint, random_slovenian_name
from training.entities.training_config import TrainingConfig, scheduled_lr
from training.entities.training_run import TrainingRun

__all__ = [
    "IterationResult",
    "ModelIdentity",
    "TrainingConfig",
    "TrainingRun",
    "name_from_checkpoint",
    "random_slovenian_name",
    "scheduled_lr",
]
