"""HoFPort — abstract interface for Hall of Fame management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from training_lab.entities.checkpoint import Checkpoint


class HoFPort(ABC):
    """Hall of Fame management — save, pin, unpin, evict models."""

    @abstractmethod
    def save(
        self,
        state_dict: dict,
        checkpoint: Checkpoint,
        pinned: bool = False,
    ) -> Checkpoint:
        """Save a model to the HoF. Auto-evicts if over limit."""

    @abstractmethod
    def list(self) -> list[Checkpoint]:
        """Return all HoF entries (pinned first, then auto by score desc)."""

    @abstractmethod
    def list_auto(self) -> list[Checkpoint]:
        """Return only auto (non-pinned) entries."""

    @abstractmethod
    def load(self, model_hash: str) -> tuple[dict, Checkpoint] | None:
        """Load a HoF model by hash. Returns (state_dict, metadata) or None."""

    @abstractmethod
    def pin(self, model_hash: str) -> bool:
        """Pin a model (exempt from auto-eviction)."""

    @abstractmethod
    def unpin(self, model_hash: str) -> bool:
        """Unpin a model (subject to auto-eviction again)."""

    @abstractmethod
    def remove(self, model_hash: str) -> bool:
        """Remove a model by hash."""
