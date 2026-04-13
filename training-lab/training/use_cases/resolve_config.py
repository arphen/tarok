"""Use case: resolve training config from CLI + YAML."""

from __future__ import annotations

from typing import Any

from training.entities.training_config import TrainingConfig
from training.ports.config_port import ConfigPort


class ResolveConfig:
    def __init__(self, config_loader: ConfigPort):
        self._loader = config_loader

    def resolve(self, cli: dict[str, Any], config_path: str | None) -> TrainingConfig:
        base: dict[str, Any] = {}
        if config_path:
            base = self._loader.load(config_path)

        merged = {**base, **{k: v for k, v in cli.items() if v is not None}}

        return TrainingConfig(
            seats=merged.get("seats", "nn,bot_v5,bot_v5,bot_v5"),
            bench_seats=merged.get("bench_seats"),
            iterations=merged.get("iterations", 10),
            games=merged.get("games", 10_000),
            bench_games=merged.get("bench_games", 10_000),
            ppo_epochs=merged.get("ppo_epochs", 6),
            batch_size=merged.get("batch_size", 8192),
            lr=merged.get("lr", 3e-4),
            lr_schedule=merged.get("lr_schedule", "constant"),
            lr_min=merged.get("lr_min"),
            explore_rate=merged.get("explore_rate", 0.10),
            device=merged.get("device", "auto"),
            save_dir=merged.get("save_dir", "checkpoints/training_run"),
            concurrency=merged.get("concurrency", 128),
        )
