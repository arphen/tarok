"""Checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Checkpoint:
    """Metadata for a saved model checkpoint."""
    model_hash: str
    train_step: int = 0
    win_rate: float = 0.0
    persona: dict = field(default_factory=dict)
    phase_label: str = ""
    hidden_size: int = 256
    oracle_critic: bool = False
    timestamp: float = 0.0
    pinned: bool = False
    eval_history: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
