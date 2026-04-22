"""Use case: resolve training config from CLI + YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.entities.league import LeagueConfig, LeagueOpponent
from training.entities.training_config import TrainingConfig
from training.entities.duplicate_config import DuplicateConfig
from training.ports.config_port import ConfigPort


def _parse_league(raw: dict[str, Any], default_outplace_unit_weight: float) -> LeagueConfig | None:
    """Parse the optional ``league:`` block from a YAML dict."""
    if not raw:
        return None
    opponents = tuple(
        LeagueOpponent(
            name=o["name"],
            type=o["type"],
            path=o.get("path"),
            initial_elo=float(o.get("initial_elo", 1500.0)),
        )
        for o in raw.get("opponents", [])
    )
    return LeagueConfig(
        enabled=raw.get("enabled", True),
        opponents=opponents,
        min_nn_per_game=raw.get("min_nn_per_game", 1),
        sampling=raw.get("sampling", "pfsp"),
        pfsp_alpha=raw.get("pfsp_alpha", 1.5),
        snapshot_interval=raw.get("snapshot_interval", 5),
        snapshot_elo_delta=float(raw.get("snapshot_elo_delta", 50.0)),
        max_active_snapshots=max(0, int(raw.get("max_active_snapshots", 3))),
        elo_use_greedy_eval_only=bool(raw.get("elo_use_greedy_eval_only", True)),
        elo_eval_games=max(1, int(raw.get("elo_eval_games", 2_000))),
        elo_eval_interval=max(1, int(raw.get("elo_eval_interval", 1))),
        initial_calibration_enabled=bool(raw.get("initial_calibration_enabled", False)),
        initial_calibration_games_per_pair=max(
            1, int(raw.get("initial_calibration_games_per_pair", 2_000))
        ),
        initial_calibration_anchor=raw.get("initial_calibration_anchor"),
        initial_calibration_anchor_elo=float(raw.get("initial_calibration_anchor_elo", 1500.0)),
        snapshot_calibration_enabled=bool(raw.get("snapshot_calibration_enabled", False)),
        snapshot_calibration_games_per_opponent=max(
            1, int(raw.get("snapshot_calibration_games_per_opponent", 1_000))
        ),
        elo_outplace_unit_weight=float(raw.get("elo_outplace_unit_weight", default_outplace_unit_weight)),
    )


def _parse_duplicate(raw: dict[str, Any]) -> DuplicateConfig:
    """Parse the optional ``duplicate:`` block from a YAML dict.

    An absent or empty block yields a default :class:`DuplicateConfig` which is
    equivalent to the feature being disabled.
    """
    if not raw:
        return DuplicateConfig()
    return DuplicateConfig(
        enabled=bool(raw.get("enabled", False)),
        actor_only=bool(raw.get("actor_only", False)),
        pairing=str(raw.get("pairing", "rotation_8game")),
        pods_per_iteration=int(raw.get("pods_per_iteration", 400)),
        shadow_source=str(raw.get("shadow_source", "previous_iteration")),
        apply_shaped_bonuses=bool(raw.get("apply_shaped_bonuses", False)),
        reward_model=str(raw.get("reward_model", "shadow_score_diff")),
        rng_seed=int(raw.get("rng_seed", 0)),
    )


class ResolveConfig:
    def __init__(self, config_loader: ConfigPort):
        self._loader = config_loader

    def resolve(self, cli: dict[str, Any], config_path: str | None) -> TrainingConfig:
        base: dict[str, Any] = {}
        if config_path:
            base = self._loader.load(config_path)

        merged = {**base, **{k: v for k, v in cli.items() if v is not None}}

        default_outplace_unit_weight = float(max(1, int(merged.get("outplace_session_size", 50))))
        league = _parse_league(merged.get("league") or {}, default_outplace_unit_weight)
        duplicate = _parse_duplicate(merged.get("duplicate") or {})

        raw_bench_checkpoints = merged.get("benchmark_checkpoints", [0, 4, 7])
        bench_checkpoints = tuple(sorted({int(x) for x in raw_bench_checkpoints if int(x) >= 0}))
        metric = str(merged.get("best_model_metric", "loss")).strip().lower()
        if metric not in {"loss", "placement", "elo"}:
            metric = "loss"

        return TrainingConfig(
            profile_name=(Path(config_path).stem if config_path else "custom"),
            seats=merged.get("seats", "nn,nn,nn,nn"),
            bench_seats=merged.get("bench_seats"),
            iterations=merged.get("iterations", 10),
            games=merged.get("games", 10_000),
            outplace_session_size=max(1, int(merged.get("outplace_session_size", 50))),
            bench_games=merged.get("bench_games", 3_000),
            benchmark_checkpoints=bench_checkpoints,
            best_model_metric=metric,
            ppo_epochs=merged.get("ppo_epochs", 6),
            batch_size=merged.get("batch_size", 8192),
            lr=merged.get("lr", 3e-4),
            lr_schedule=str(merged.get("lr_schedule", "constant")),
            lr_min=merged.get("lr_min"),
            explore_rate=merged.get("explore_rate", 0.10),
            explore_rate_min=merged.get("explore_rate_min"),
            explore_rate_schedule=str(merged.get("explore_rate_schedule", "constant")),
            device=merged.get("device", "auto"),
            save_dir=merged.get("save_dir", "data/checkpoints/training_run"),
            concurrency=merged.get("concurrency", 128),
            lapajne_mc_worlds=(
                int(merged["lapajne_mc_worlds"])
                if merged.get("lapajne_mc_worlds") is not None
                else None
            ),
            lapajne_mc_sims=(
                int(merged["lapajne_mc_sims"])
                if merged.get("lapajne_mc_sims") is not None
                else None
            ),
            imitation_coef=merged.get("imitation_coef", 0.3),
            imitation_schedule=str(merged.get("imitation_schedule", "constant")),
            imitation_coef_min=merged.get("imitation_coef_min", 0.0),
            imitation_center_elo=float(merged.get("imitation_center_elo", 1500.0)),
            imitation_width_elo=float(merged.get("imitation_width_elo", 250.0)),
            behavioral_clone_coef=float(merged.get("behavioral_clone_coef", 0.0)),
            behavioral_clone_schedule=str(merged.get("behavioral_clone_schedule", "constant")),
            behavioral_clone_coef_min=float(merged.get("behavioral_clone_coef_min", 0.0)),
            behavioral_clone_teacher=merged.get("behavioral_clone_teacher"),
            behavioral_clone_games_per_iteration=max(
                0, int(merged.get("behavioral_clone_games_per_iteration", 0))
            ),
            policy_coef=float(merged.get("policy_coef", 1.0)),
            entropy_coef=merged.get("entropy_coef", 0.01),
            entropy_schedule=merged.get("entropy_schedule", "constant"),
            entropy_coef_min=float(merged.get("entropy_coef_min", 0.005)),
            iteration_runner_mode=str(merged.get("iteration_runner_mode", "in-process")),
            iteration_runner_restart_every=max(1, int(merged.get("iteration_runner_restart_every", 10))),
            model_arch=merged.get("model_arch", "v4"),
            oracle_critic=bool(merged.get("oracle_critic", True)),
            human_data_dir=merged.get("human_data_dir"),
            league=league,
            duplicate=duplicate,
        )
