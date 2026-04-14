"""Port: model persistence interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelPort(ABC):
    @abstractmethod
    def load_weights(self, checkpoint_path: str) -> tuple[dict, int, bool, str]:
        """Load checkpoint → (state_dict, hidden_size, oracle_critic, model_arch)."""

    @abstractmethod
    def create_new(self, hidden_size: int, oracle: bool, model_arch: str) -> dict:
        """Create fresh random weights → state_dict."""

    @abstractmethod
    def export_for_inference(self, weights: dict, hidden_size: int, oracle: bool, model_arch: str, path: str) -> None:
        """Export model to format the self-play engine can load."""

    @abstractmethod
    def save_checkpoint(
        self, weights: dict, hidden_size: int, oracle: bool, model_arch: str,
        iteration: int, loss: float, placement: float, path: str,
    ) -> None:
        """Save training checkpoint."""

    @abstractmethod
    def copy_best(self, src: str, dst: str) -> None:
        """Copy the best checkpoint."""
