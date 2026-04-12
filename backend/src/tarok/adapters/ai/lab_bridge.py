"""Bridge between training-lab package and the backend API layer.

Implements MetricsSinkPort and ProgressPort to accumulate training-lab
metrics into the dashboard-compatible TrainingMetrics format used by
the existing frontend.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from training_lab.entities.config import TrainingConfig
from training_lab.entities.metrics import SessionMetrics, TrainingProgress
from training_lab.entities.network import TarokNet
from training_lab.ports.metrics_sink import MetricsSinkPort
from training_lab.ports.progress import ProgressPort


@dataclass
class LabDashboardMetrics:
    """Dashboard-compatible metrics accumulated from training-lab sessions.

    Mirrors the shape of TrainingMetrics.to_dict() so the existing
    frontend charts work without changes.
    """
    run_id: str = ""
    episode: int = 0
    total_episodes: int = 0
    session: int = 0
    total_sessions: int = 0
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    avg_placement: float = 0.0
    entropy: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    games_per_second: float = 0.0
    bid_rate: float = 0.0
    klop_rate: float = 0.0
    solo_rate: float = 0.0

    reward_history: list[float] = field(default_factory=list)
    avg_placement_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    session_avg_score_history: list[float] = field(default_factory=list)

    # These remain empty for now — training-lab doesn't track them yet
    bid_rate_history: list[float] = field(default_factory=list)
    klop_rate_history: list[float] = field(default_factory=list)
    solo_rate_history: list[float] = field(default_factory=list)
    table_score_history: list[float] = field(default_factory=list)
    stockskis_place_history: list[float] = field(default_factory=list)
    lookahead_score_history: list[float] = field(default_factory=list)
    lookahead_bid_rate_history: list[float] = field(default_factory=list)
    contract_stats: dict[str, dict] = field(default_factory=dict)
    contract_win_rate_history: dict[str, list[float]] = field(default_factory=dict)
    tarok_count_bids: dict[str, dict[str, int]] = field(default_factory=dict)
    snapshots: list[dict] = field(default_factory=list)
    placement_selfplay_history: list[float] = field(default_factory=list)
    placement_hof_history: list[float] = field(default_factory=list)
    placement_v5_history: list[float] = field(default_factory=list)

    # training-lab specific
    buffer_size: int = 0
    policy_version: int = 0
    phase: str = "idle"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "episode": self.episode,
            "total_episodes": self.total_episodes,
            "session": self.session,
            "total_sessions": self.total_sessions,
            "avg_reward": round(self.avg_reward, 4),
            "avg_loss": round(self.avg_loss, 4),
            "avg_placement": round(self.avg_placement, 4),
            "entropy": round(self.entropy, 4),
            "value_loss": round(self.value_loss, 4),
            "policy_loss": round(self.policy_loss, 4),
            "games_per_second": round(self.games_per_second, 2),
            "bid_rate": round(self.bid_rate, 4),
            "klop_rate": round(self.klop_rate, 4),
            "solo_rate": round(self.solo_rate, 4),
            "history_offset": max(0, len(self.reward_history) - 500),
            "reward_history": self.reward_history[-500:],
            "avg_placement_history": self.avg_placement_history[-500:],
            "loss_history": self.loss_history[-500:],
            "bid_rate_history": self.bid_rate_history[-500:],
            "klop_rate_history": self.klop_rate_history[-500:],
            "solo_rate_history": self.solo_rate_history[-500:],
            "contract_stats": self.contract_stats,
            "contract_win_rate_history": {
                k: v[-500:] for k, v in self.contract_win_rate_history.items()
            },
            "session_avg_score_history": self.session_avg_score_history[-500:],
            "table_score_history": self.table_score_history[-500:],
            "stockskis_place_history": self.stockskis_place_history[-500:],
            "lookahead_score_history": self.lookahead_score_history[-500:],
            "lookahead_bid_rate_history": self.lookahead_bid_rate_history[-500:],
            "snapshots": self.snapshots,
            "tarok_count_bids": self.tarok_count_bids,
            "vs_v5": {"games": 0, "wins": 0, "win_rate": 0, "avg_score": 0,
                       "avg_place": 0, "bid_rate": 0, "decl_games": 0,
                       "decl_win_rate": 0, "decl_avg_score": 0, "def_games": 0,
                       "def_win_rate": 0, "def_avg_score": 0},
            "vs_v5_contract_stats": {},
            "vs_v5_win_rate_history": [],
            "vs_v5_avg_score_history": [],
            "vs_v5_avg_place_history": [],
            "vs_v5_bid_rate_history": [],
            "vs_v5_eval_signal_history": [],
            "placement_selfplay_history": self.placement_selfplay_history[-500:],
            "placement_hof_history": self.placement_hof_history[-500:],
            "placement_v5_history": self.placement_v5_history[-500:],
            # Training-lab extras
            "buffer_size": self.buffer_size,
            "policy_version": self.policy_version,
            "lab_phase": self.phase,
        }


class DashboardMetricsSink(MetricsSinkPort):
    """Accumulates SessionMetrics into LabDashboardMetrics for the frontend."""

    def __init__(self, total_sessions: int):
        self._lock = threading.Lock()
        self.metrics = LabDashboardMetrics()
        self.metrics.total_sessions = total_sessions
        self.metrics.run_id = hashlib.sha256(
            f"{time.time()}-lab-{id(self)}".encode()
        ).hexdigest()[:8]

    def record(self, m: SessionMetrics) -> None:
        with self._lock:
            self.metrics.session = m.session_id + 1
            self.metrics.episode += m.games_played
            self.metrics.avg_reward = m.avg_reward
            self.metrics.avg_loss = m.total_loss
            self.metrics.avg_placement = m.avg_placement
            self.metrics.policy_loss = m.policy_loss
            self.metrics.value_loss = m.value_loss
            self.metrics.entropy = m.entropy
            self.metrics.games_per_second = m.games_per_sec

            self.metrics.reward_history.append(round(m.avg_reward, 4))
            self.metrics.loss_history.append(round(m.total_loss, 4))
            self.metrics.avg_placement_history.append(round(m.avg_placement, 4))
            self.metrics.session_avg_score_history.append(
                round(m.total_reward / max(m.games_played, 1) * 100, 1)
            )

    def flush(self) -> None:
        pass

    def snapshot(self) -> dict:
        """Thread-safe copy of metrics dict for the API."""
        with self._lock:
            return self.metrics.to_dict()


class DashboardProgress(ProgressPort):
    """Feeds TrainingProgress into the metrics sink for dashboard display."""

    def __init__(self, sink: DashboardMetricsSink):
        self._sink = sink

    def report(self, progress: TrainingProgress) -> None:
        with self._sink._lock:
            self._sink.metrics.phase = progress.phase
            self._sink.metrics.buffer_size = progress.buffer_size
            self._sink.metrics.policy_version = progress.policy_version
            self._sink.metrics.total_episodes = (
                progress.total_sessions * 20  # approximate
            )


def start_lab_training(
    checkpoint_path: str | None,
    config_overrides: dict[str, Any],
) -> tuple[Any, DashboardMetricsSink]:
    """Create and start training-lab RunPPOTraining in a background thread.

    Returns (runner, metrics_sink) so the API can poll metrics and stop.
    """
    from training_lab.entities.config import TrainingConfig
    from training_lab.adapters.compute.factory import create as create_compute
    from training_lab.adapters.engine.rust_batch_runner import RustBatchGameRunner
    from training_lab.adapters.storage.file_checkpoint_store import FileCheckpointStore
    from training_lab.use_cases.ppo_training import RunPPOTraining

    # Load checkpoint if provided
    resume_sd = None
    hidden_size = config_overrides.get("hidden_size", 256)
    oracle = config_overrides.get("oracle_critic", False)

    if checkpoint_path:
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in cp:
            resume_sd = cp["model_state_dict"]
        else:
            resume_sd = cp
        hidden_size = resume_sd["shared.0.weight"].shape[0]
        oracle = any(k.startswith("critic_backbone") for k in resume_sd)

    config = TrainingConfig(
        num_sessions=config_overrides.get("num_sessions", 1000),
        games_per_session=config_overrides.get("games_per_session", 20),
        learning_rate=config_overrides.get("learning_rate", 3e-4),
        hidden_size=hidden_size,
        oracle_critic=oracle,
        buffer_capacity=config_overrides.get("buffer_capacity", 50_000),
        min_experiences=config_overrides.get("min_experiences", 5_000),
        producer_concurrency=config_overrides.get("concurrency", 128),
        ppo_epochs=config_overrides.get("ppo_epochs", 6),
        mini_batch_size=config_overrides.get("batch_size", 256),
        explore_rate=config_overrides.get("explore_rate", 0.1),
        checkpoint_interval=config_overrides.get("checkpoint_interval", 50),
        device=config_overrides.get("device", "auto"),
        num_producers=config_overrides.get("num_producers", 0),
    )

    compute = create_compute(config.device)
    simulator = RustBatchGameRunner(
        compute=compute,
        concurrency=config.producer_concurrency,
        oracle=config.oracle_critic,
    )
    save_dir = config_overrides.get("save_dir", "checkpoints")
    store = FileCheckpointStore(save_dir)

    sink = DashboardMetricsSink(total_sessions=config.num_sessions)
    progress = DashboardProgress(sink)

    runner = RunPPOTraining(
        simulator=simulator,
        compute=compute,
        store=store,
        config=config,
        metrics_sink=sink,
        progress=progress,
        resume_state_dict=resume_sd,
    )

    return runner, sink
