"""ProgressPort — abstract interface for reporting training progress."""

from __future__ import annotations

from abc import ABC, abstractmethod

from training_lab.entities.metrics import TrainingProgress


class ProgressPort(ABC):
    """Callback so the web layer can poll training status."""

    @abstractmethod
    def report(self, progress: TrainingProgress) -> None:
        """Report current training progress."""
