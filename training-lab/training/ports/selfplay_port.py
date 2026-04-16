"""Port: self-play engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SelfPlayPort(ABC):
    @abstractmethod
    def run(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        explore_rate: float,
        concurrency: int,
        include_replay_data: bool = False,
        include_oracle_states: bool = False,
    ) -> dict[str, Any]:
        """Run self-play games, return raw experience dict."""
