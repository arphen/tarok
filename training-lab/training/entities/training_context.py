"""Entity: mutable context shared across training-loop iterations.

Passed by reference between PrepareTraining, AdvanceIteration, and
PromoteBestCheckpoint so the orchestrator (TrainModel) stays declarative —
it never reads or mutates individual fields, only passes the object along.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from training.entities.league import LeaguePool
from training.entities.training_run import TrainingRun


@dataclass
class TrainingContext:
    """Mutable state container for one training run.

    Lifetime: created by PrepareTraining, mutated in-place by AdvanceIteration
    for each loop iteration, and read by PromoteBestCheckpoint after the loop.
    """

    run: TrainingRun
    pool: LeaguePool
    last_snapshot_elo: Optional[float]
    ts_path: str
    save_dir: Path
    league_pool_dir: Path
