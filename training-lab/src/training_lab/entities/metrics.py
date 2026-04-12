"""Training metrics and progress tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class SessionMetrics:
    """Metrics for a single training session (batch of games)."""
    session_id: int = 0
    games_played: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    win_rate: float = 0.0
    avg_placement: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    games_per_sec: float = 0.0
    experiences_count: int = 0
    explore_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingProgress:
    """Overall training progress state."""
    phase: str = "idle"  # "idle", "warmup", "imitation", "ppo", "eval"
    current_session: int = 0
    total_sessions: int = 0
    policy_version: int = 0
    best_win_rate: float = 0.0
    elapsed_seconds: float = 0.0
    buffer_size: int = 0
    is_running: bool = False
