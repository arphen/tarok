#!/usr/bin/env python3
"""Train an existing Tarok model using the training-lab package.

Usage:
    # PPO training from an existing checkpoint:
    python train.py ppo --checkpoint ../backend/checkpoints/tarok_agent_latest.pt

    # PPO from scratch:
    python train.py ppo

    # Imitation pre-training (from StockŠkis expert games):
    python train.py imitation --checkpoint ../backend/checkpoints/tarok_agent_latest.pt

    # Imitation then PPO (full pipeline):
    python train.py imitation --checkpoint ../backend/checkpoints/tarok_agent_latest.pt
    python train.py ppo --checkpoint checkpoints/tarok_agent_imitation.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def load_checkpoint(path: str) -> tuple[dict, int, bool]:
    """Load a checkpoint and return (state_dict, hidden_size, oracle)."""
    cp = torch.load(path, map_location="cpu", weights_only=False)

    # Handle both old backend format and new training-lab format
    if "model_state_dict" in cp:
        sd = cp["model_state_dict"]
    else:
        sd = cp

    # Infer hidden size from weights
    hidden = sd["shared.0.weight"].shape[0]
    oracle = any(k.startswith("critic_backbone") for k in sd)
    return sd, hidden, oracle


def run_ppo(args: argparse.Namespace) -> None:
    from training_lab.entities.config import TrainingConfig
    from training_lab.adapters.compute.factory import create as create_compute
    from training_lab.adapters.engine.rust_batch_runner import RustBatchGameRunner
    from training_lab.adapters.storage.file_checkpoint_store import FileCheckpointStore
    from training_lab.use_cases.ppo_training import RunPPOTraining

    # Load existing checkpoint if provided
    resume_sd = None
    hidden_size = args.hidden_size
    oracle = args.oracle

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        resume_sd, hidden_size, oracle = load_checkpoint(args.checkpoint)
        print(f"  hidden_size={hidden_size}, oracle={oracle}")

    config = TrainingConfig(
        num_sessions=args.sessions,
        games_per_session=args.games_per_session,
        learning_rate=args.lr,
        hidden_size=hidden_size,
        oracle_critic=oracle,
        buffer_capacity=args.buffer_capacity,
        min_experiences=args.min_experiences,
        producer_concurrency=args.concurrency,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.batch_size,
        explore_rate=args.explore_rate,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        num_producers=args.producers,
    )

    compute = create_compute(config.device)
    print(f"Compute device: {compute.device}")

    simulator = RustBatchGameRunner(
        compute=compute,
        concurrency=config.producer_concurrency,
        oracle=config.oracle_critic,
    )

    store = FileCheckpointStore(args.save_dir)

    trainer = RunPPOTraining(
        simulator=simulator,
        compute=compute,
        store=store,
        config=config,
        resume_state_dict=resume_sd,
    )

    from training_lab.infra.mp_producer import auto_num_producers
    n_prod = config.num_producers if config.num_producers > 0 else auto_num_producers()

    print(f"\nStarting PPO training:")
    print(f"  sessions={config.num_sessions}, games/session={config.games_per_session}")
    print(f"  buffer={config.buffer_capacity}, min_exps={config.min_experiences}")
    print(f"  lr={config.learning_rate}, concurrency={config.producer_concurrency}")
    print(f"  producers={n_prod} {'(auto)' if config.num_producers <= 0 else ''}")
    print(f"  checkpoints → {args.save_dir}/")
    print()

    t0 = time.time()
    result = trainer.run()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  sessions={result['total_sessions']}, games={result['total_games']}")
    print(f"  experiences={result['total_experiences']}")
    print(f"  policy_version={result['policy_version']}")


def run_imitation(args: argparse.Namespace) -> None:
    from training_lab.adapters.compute.factory import create as create_compute
    from training_lab.use_cases.imitation import RunImitationPretraining

    # Load existing checkpoint if provided
    resume_sd = None
    hidden_size = args.hidden_size
    oracle = args.oracle

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        resume_sd, hidden_size, oracle = load_checkpoint(args.checkpoint)
        print(f"  hidden_size={hidden_size}, oracle={oracle}")

    compute = create_compute(args.device)
    print(f"Compute device: {compute.device}")

    # Build network (optionally with checkpoint weights)
    from training_lab.entities.network import TarokNet
    network = TarokNet(hidden_size=hidden_size, oracle_critic=oracle)
    if resume_sd is not None:
        network.load_state_dict(resume_sd)

    trainer = RunImitationPretraining(
        compute=compute,
        network=network,
    )

    def on_progress(info: dict) -> None:
        print(
            f"  chunk {info.get('chunk', '?')}: "
            f"exps={info.get('total_experiences', 0)}, "
            f"policy_loss={info.get('avg_policy_loss', 0):.4f}, "
            f"value_loss={info.get('avg_value_loss', 0):.4f}",
            flush=True,
        )

    print(f"\nStarting imitation pre-training:")
    print(f"  games={args.num_games}, epochs={args.epochs}, lr={args.lr}")
    print()

    t0 = time.time()
    result = trainer.run(
        num_games=args.num_games,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        include_oracle=oracle,
        progress_callback=on_progress,
    )
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  experiences={result.get('total_experiences', 0)}")

    # Save the imitation checkpoint
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "tarok_agent_imitation.pt"
    torch.save({
        "model_state_dict": trainer.network.state_dict(),
        "hidden_size": hidden_size,
        "oracle_critic": oracle,
        "phase": "imitation",
        "num_games": args.num_games,
    }, save_path)
    print(f"  saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Tarok training-lab CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- PPO subcommand ---
    ppo = sub.add_parser("ppo", help="PPO self-play training")
    ppo.add_argument("--checkpoint", "-c", type=str, default=None,
                     help="Path to existing .pt checkpoint to resume from")
    ppo.add_argument("--sessions", "-s", type=int, default=1000)
    ppo.add_argument("--games-per-session", type=int, default=20)
    ppo.add_argument("--lr", type=float, default=3e-4)
    ppo.add_argument("--hidden-size", type=int, default=256)
    ppo.add_argument("--oracle", action="store_true")
    ppo.add_argument("--buffer-capacity", type=int, default=50_000)
    ppo.add_argument("--min-experiences", type=int, default=5_000)
    ppo.add_argument("--concurrency", type=int, default=128)
    ppo.add_argument("--ppo-epochs", type=int, default=6)
    ppo.add_argument("--batch-size", type=int, default=8192)
    ppo.add_argument("--explore-rate", type=float, default=0.1)
    ppo.add_argument("--checkpoint-interval", type=int, default=50)
    ppo.add_argument("--save-dir", type=str, default="checkpoints")
    ppo.add_argument("--producers", type=int, default=0,
                     help="Number of producer processes (0=auto-detect)")
    ppo.add_argument("--device", type=str, default="auto")

    # --- Imitation subcommand ---
    imit = sub.add_parser("imitation", help="Imitation pre-training from StockŠkis experts")
    imit.add_argument("--checkpoint", "-c", type=str, default=None,
                      help="Path to existing .pt checkpoint to fine-tune")
    imit.add_argument("--num-games", "-n", type=int, default=1_000_000)
    imit.add_argument("--epochs", type=int, default=3)
    imit.add_argument("--lr", type=float, default=1e-3)
    imit.add_argument("--batch-size", type=int, default=2048)
    imit.add_argument("--hidden-size", type=int, default=256)
    imit.add_argument("--oracle", action="store_true")
    imit.add_argument("--save-dir", type=str, default="checkpoints")
    imit.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.command == "ppo":
        run_ppo(args)
    elif args.command == "imitation":
        run_imitation(args)


if __name__ == "__main__":
    main()
