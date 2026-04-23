"""Port: compute per-iteration duplicate statistics.

See :class:`training.entities.duplicate_iteration_stats.DuplicateIterationStats`
for the output shape. Implementations live in ``training.adapters.duplicate``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from training.entities.duplicate_iteration_stats import DuplicateIterationStats
from training.entities.duplicate_pod import DuplicatePod


class DuplicateIterationStatsPort(ABC):
    @abstractmethod
    def compute(
        self,
        *,
        active_raw: dict[str, Any],
        shadow_scores: Any,  # ndarray (n_pods, n_games_per_group, 4)
        pods: Sequence[DuplicatePod],
        pod_ids: Any,  # ndarray per-step, aligned with active_raw["players"]
        learner_positions: Any,  # ndarray (n_pods, n_games_per_group)
        active_game_ids: Any,  # ndarray (n_pods, n_games_per_group)
    ) -> DuplicateIterationStats:
        """Aggregate per-game comparisons into a ``DuplicateIterationStats``."""
