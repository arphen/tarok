"""CheckpointStorePort — abstract interface for model persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from training_lab.entities.checkpoint import Checkpoint


class CheckpointStorePort(ABC):
    """Save/load model checkpoints."""

    @abstractmethod
    def save(
        self,
        state_dict: dict,
        checkpoint: Checkpoint,
    ) -> Path:
        """Save model weights + metadata. Returns the path."""

    @abstractmethod
    def load(self, identifier: str) -> tuple[dict, Checkpoint] | None:
        """Load model state_dict + metadata by identifier (hash or filename)."""

    @abstractmethod
    def list(self) -> list[Checkpoint]:
        """List all saved checkpoints."""

    @abstractmethod
    def delete(self, identifier: str) -> bool:
        """Delete a checkpoint by identifier."""
