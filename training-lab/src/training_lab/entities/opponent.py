"""Opponent specification for training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OpponentType(Enum):
    SELF_PLAY = "self_play"
    FSP = "fsp"        # Fictitious Self-Play (historical snapshots)
    HEURISTIC = "heuristic"  # StockSkis or similar rule-based
    HOF = "hof"        # Hall of Fame model


@dataclass
class OpponentSpec:
    """Describes an opponent for training mix configuration."""
    type: OpponentType
    checkpoint_hash: str | None = None  # for FSP/HOF opponents
    strength: float = 1.0               # for heuristic opponents
    weight: float = 1.0                 # sampling weight in opponent mix
