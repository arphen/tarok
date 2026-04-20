"""Value object: per-iteration hyperparameter overrides computed by policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IterationHyperparams:
    """Bundle of per-iteration hyperparameters resolved by policy objects.

    Replaces the five separate ``float | None`` values that were threaded
    through every layer of the iteration pipeline.
    """

    lr: float
    imitation_coef: float
    behavioral_clone_coef: float
    entropy_coef: float
    explore_rate: float
