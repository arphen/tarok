"""GameSimulatorPort — abstract interface for running batches of games."""

from __future__ import annotations

from typing import Protocol

from training_lab.entities.experience import Experience


class GameResult:
    """Result of a single completed game."""
    __slots__ = ("experiences", "scores", "winner")

    def __init__(
        self,
        experiences: list[Experience],
        scores: list[float],
        winner: int,
    ):
        self.experiences = experiences
        self.scores = scores
        self.winner = winner


class GameSimulatorPort(Protocol):
    """Runs N games and returns experiences.

    The single adapter is RustBatchGameRunner (via tarok_engine).
    """

    def play_batch(
        self,
        network: object,
        n_games: int,
        explore_rate: float = 0.1,
    ) -> list[GameResult]:
        """Play n_games and return results with experiences."""
        ...
