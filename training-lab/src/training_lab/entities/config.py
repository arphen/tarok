"""Training configuration — all hyperparameters as a frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """All hyperparameters for a PPO training run."""

    # --- PPO ---
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 6
    mini_batch_size: int = 8192

    # --- Producer-consumer ---
    buffer_capacity: int = 50_000
    min_experiences: int = 5_000
    max_staleness: int = 3
    producer_concurrency: int = 128
    network_refresh_interval: int = 1  # refresh frozen copy every N consumer updates

    # --- Session ---
    num_sessions: int = 1000
    games_per_session: int = 20
    explore_rate: float = 0.1
    explore_decay: float = 0.995
    explore_floor: float = 0.02

    # --- Network ---
    hidden_size: int = 256
    oracle_critic: bool = False

    # --- Checkpointing ---
    checkpoint_interval: int = 50
    eval_interval: int = 100
    eval_games: int = 100

    # --- Opponents ---
    self_play_ratio: float = 0.70
    stockskis_ratio: float = 0.10
    fsp_ratio: float = 0.20

    # --- Multi-process ---
    num_producers: int = 0  # 0 = auto-detect (cpu_count // 2 - 1)

    # --- Device ---
    device: str = "auto"
