"""TrainingRun — mutable aggregate root."""

from __future__ import annotations

from dataclasses import dataclass, field

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig


@dataclass
class TrainingRun:
    config: TrainingConfig
    identity: ModelIdentity
    initial_placement: float = 0.0
    results: list[IterationResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def placements(self) -> list[float]:
        return [self.initial_placement] + [r.placement for r in self.results]

    @property
    def best_placement(self) -> float:
        return min(self.placements)

    @property
    def best_iteration(self) -> int:
        p = self.placements
        return p.index(min(p))

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time

    @property
    def improved(self) -> bool:
        return self.best_iteration > 0
