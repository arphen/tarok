"""Port: benchmark engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BenchmarkPort(ABC):
    @abstractmethod
    def measure_placement(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        concurrency: int,
        session_size: int,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
    ) -> float:
        """Play greedy games and return average placement (1.0–4.0)."""
