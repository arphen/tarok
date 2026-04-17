#!/usr/bin/env python3
"""Train a Tarok agent iteratively and track improvement.

Each iteration:
  1. Self-play N games (seat layout from config) via Rust engine
  2. PPO update on the collected experiences (GPU if available)
  3. Benchmark: N games greedy, measure avg placement

Configs live in training-lab/configs/ — pick a profile or make your own.

Usage:
    python train_and_evaluate.py --config configs/vs-3-bots.yaml -c model.pt
    python train_and_evaluate.py -c model.pt --seats nn,bot_v6,bot_v6,bot_v6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# ── Resolve paths before chdir ──────────────────────────────────────
# Config files may be given relative to cwd (Makefile) or to script dir (manual).
# Checkpoint paths are always relative to project root (cwd after chdir).
_script_dir = Path(__file__).resolve().parent
_orig_cwd = Path.cwd()
_root = _script_dir.parent

sys.path.insert(0, str(_root / "model" / "src"))
sys.path.insert(0, str(_root / "backend" / "src"))
sys.path.insert(0, str(_script_dir))
os.chdir(str(_root))

from training.container import Container
from training.entities import ModelIdentity, TrainingConfig


def _detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_path(raw: str | None) -> str | None:
    """Resolve a path that might be relative to the original cwd or script dir."""
    if raw is None:
        return None
    # Try relative to original cwd first (handles Makefile invocation)
    candidate = _orig_cwd / raw
    if candidate.exists():
        return str(candidate)
    # Try relative to script dir (handles `cd training-lab && python ...`)
    candidate = _script_dir / raw
    if candidate.exists():
        return str(candidate)
    # Try relative to project root (cwd after chdir)
    candidate = _root / raw
    if candidate.exists():
        return str(candidate)
    # Return as-is and let downstream fail with a clear error
    return raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative PPO training with progress bar + benchmark placement tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configs live in training-lab/configs/.  Available profiles:
  vs-3-bots   NN vs 3 rule-based bots (default, easiest)
  vs-2-bots   2 NNs vs 2 bots
  vs-1-bot    3 NNs vs 1 bot
  self-play   4 NNs, no bots
  vs-3-v6     NN vs 3 stronger v6 bots

Examples:
  python train_and_evaluate.py --config configs/vs-3-bots.yaml -c model.pt
  python train_and_evaluate.py --config configs/self-play.yaml -c model.pt --iterations 20
  python train_and_evaluate.py -c model.pt --seats nn,bot_v6,bot_v6,bot_v6
  python train_and_evaluate.py --new --config configs/vs-3-bots.yaml
        """,
    )
    parser.add_argument("--config", type=str, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", "-c", type=str, default=None)
    group.add_argument("--new", action="store_true", default=False)
    parser.add_argument("--seats", type=str, default=None)
    parser.add_argument("--bench-seats", type=str, default=None)
    parser.add_argument("--iterations", "-n", type=int, default=None)
    parser.add_argument("--games", "-g", type=int, default=None)
    parser.add_argument("--bench-games", type=int, default=None)
    parser.add_argument(
        "--benchmark-checkpoints",
        type=str,
        default=None,
        help="Comma-separated checkpoints to benchmark (0=initial), e.g. 0,4,7",
    )
    parser.add_argument(
        "--best-model-metric",
        type=str,
        choices=["loss", "placement"],
        default=None,
        help="Metric used to choose best.pt",
    )
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--explore-rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="Hidden layer size for new models (default: 256)")
    parser.add_argument("--model-arch", type=str, default=None,
                        choices=["v4"],
                        help="Model architecture: v4")
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["constant", "cosine", "linear"],
                        help="LR schedule across iterations: constant (default), cosine, or linear decay")
    parser.add_argument("--lr-min", type=float, default=None,
                        help="Minimum LR for cosine/linear schedules (default: lr / 10)")
    parser.add_argument("--human-data", type=str, default=None,
                        help="Directory of human-play JSONL files to mix into every PPO update")
    parser.add_argument(
        "--imitation-schedule",
        type=str,
        default=None,
        choices=["constant", "linear", "cosine"],
        help="Schedule for oracle distillation coefficient",
    )
    parser.add_argument(
        "--imitation-coef-min",
        type=float,
        default=None,
        help="Final imitation coefficient for linear/cosine schedules",
    )
    parser.add_argument(
        "--memory-telemetry",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Enable per-iteration memory telemetry output",
    )
    parser.add_argument(
        "--memory-telemetry-every",
        type=int,
        default=None,
        help="Print memory telemetry every N iterations",
    )
    parser.add_argument(
        "--iteration-runner-mode",
        type=str,
        choices=["in-process", "spawn"],
        default=None,
        help="Iteration execution strategy",
    )
    parser.add_argument(
        "--iteration-runner-restart-every",
        type=int,
        default=None,
        help="For spawn runner: restart worker every N iterations",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    container = Container()
    requested_model_arch = getattr(args, "model_arch", None)

    # ── Resolve config ──────────────────────────────────────────────
    cli_overrides = {
        "seats": args.seats,
        "bench_seats": getattr(args, "bench_seats", None),
        "iterations": args.iterations,
        "games": args.games,
        "bench_games": getattr(args, "bench_games", None),
        "benchmark_checkpoints": (
            [int(s.strip()) for s in args.benchmark_checkpoints.split(",")]
            if getattr(args, "benchmark_checkpoints", None)
            else None
        ),
        "best_model_metric": getattr(args, "best_model_metric", None),
        "ppo_epochs": getattr(args, "ppo_epochs", None),
        "batch_size": getattr(args, "batch_size", None),
        "lr": args.lr,
        "lr_schedule": args.lr_schedule if args.lr_schedule != "constant" else None,
        "lr_min": getattr(args, "lr_min", None),
        "explore_rate": getattr(args, "explore_rate", None),
        "device": args.device,
        "concurrency": args.concurrency,
        "model_arch": getattr(args, "model_arch", None),
        "human_data_dir": _resolve_path(getattr(args, "human_data", None)),
        "imitation_schedule": getattr(args, "imitation_schedule", None),
        "imitation_coef_min": getattr(args, "imitation_coef_min", None),
        "memory_telemetry": (
            True if args.memory_telemetry == "true"
            else False if args.memory_telemetry == "false"
            else None
        ),
        "memory_telemetry_every": getattr(args, "memory_telemetry_every", None),
        "iteration_runner_mode": getattr(args, "iteration_runner_mode", None),
        "iteration_runner_restart_every": getattr(args, "iteration_runner_restart_every", None),
    }
    config_path = _resolve_path(args.config)
    config = container.resolve_config().resolve(cli_overrides, config_path)

    # ── Resolve model ───────────────────────────────────────────────
    resolve = container.resolve_model()
    if args.new:
        hidden_size = getattr(args, "hidden_size", None) or 256
        identity, weights = resolve.from_scratch(
            hidden_size=hidden_size,
            model_arch=config.model_arch,
        )
    else:
        checkpoint_path = _resolve_path(args.checkpoint)
        print(f"Loading checkpoint: {args.checkpoint}")
        identity, weights = resolve.from_checkpoint(checkpoint_path)
        if requested_model_arch is None:
            if config.model_arch != identity.model_arch:
                print(
                    f"Using checkpoint architecture {identity.model_arch} "
                    f"(config requested {config.model_arch})."
                )
            config = TrainingConfig(
                seats=config.seats,
                bench_seats=config.bench_seats,
                iterations=config.iterations,
                games=config.games,
                bench_games=config.bench_games,
                benchmark_checkpoints=config.benchmark_checkpoints,
                best_model_metric=config.best_model_metric,
                ppo_epochs=config.ppo_epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                lr_schedule=config.lr_schedule,
                lr_min=config.lr_min,
                explore_rate=config.explore_rate,
                device=config.device,
                save_dir=config.save_dir,
                concurrency=config.concurrency,
                imitation_coef=config.imitation_coef,
                imitation_schedule=config.imitation_schedule,
                imitation_coef_min=config.imitation_coef_min,
                memory_telemetry=config.memory_telemetry,
                memory_telemetry_every=config.memory_telemetry_every,
                iteration_runner_mode=config.iteration_runner_mode,
                iteration_runner_restart_every=config.iteration_runner_restart_every,
                model_arch=identity.model_arch,
                human_data_dir=config.human_data_dir,
                league=config.league,
            )
        elif config.model_arch != identity.model_arch:
            print(
                f"Promoting architecture {identity.model_arch} -> {config.model_arch} "
                f"from explicit CLI override"
            )
            identity = ModelIdentity(
                name=identity.name,
                hidden_size=identity.hidden_size,
                oracle_critic=identity.oracle_critic,
                model_arch=config.model_arch,
                is_new=identity.is_new,
            )

    # ── Resolve save directory ──────────────────────────────────────
    save_dir = args.save_dir if args.save_dir is not None else f"checkpoints/{identity.name}"

    config = TrainingConfig(
        seats=config.seats,
        bench_seats=config.bench_seats,
        iterations=config.iterations,
        games=config.games,
        bench_games=config.bench_games,
        benchmark_checkpoints=config.benchmark_checkpoints,
        best_model_metric=config.best_model_metric,
        ppo_epochs=config.ppo_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        lr_min=config.lr_min,
        explore_rate=config.explore_rate,
        device=config.device,
        save_dir=save_dir,
        concurrency=config.concurrency,
        imitation_coef=config.imitation_coef,
        imitation_schedule=config.imitation_schedule,
        imitation_coef_min=config.imitation_coef_min,
        memory_telemetry=config.memory_telemetry,
        memory_telemetry_every=config.memory_telemetry_every,
        iteration_runner_mode=config.iteration_runner_mode,
        iteration_runner_restart_every=config.iteration_runner_restart_every,
        model_arch=config.model_arch,
        human_data_dir=config.human_data_dir,
        league=config.league,
    )

    device = _detect_device(config.device)

    # ── Run ─────────────────────────────────────────────────────────
    container.train_model().execute(config, identity, weights, device)


if __name__ == "__main__":
    main()
