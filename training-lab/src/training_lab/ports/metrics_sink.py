"""MetricsSinkPort — abstract interface for recording training metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from training_lab.entities.metrics import SessionMetrics


class MetricsSinkPort(ABC):
    """Push training metrics to an external consumer (API, TensorBoard, etc.)."""

    @abstractmethod
    def record(self, metrics: SessionMetrics) -> None:
        """Record a session's metrics."""

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered metrics."""
