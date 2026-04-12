#!/usr/bin/env python3
"""Train a Tarok agent iteratively and track improvement.

Each iteration:
  1. Self-play N games (seat layout from config)
  2. PPO update on the collected experiences (MPS GPU)
  3. Benchmark: N games greedy, measure avg placement

Configs live in training-lab/configs/ — pick a profile or make your own.

Usage:
    python train_and_evaluate.py --config configs/vs-3-bots.yaml -c model.pt
    python train_and_evaluate.py -c model.pt --seats nn,bot_v6,bot_v6,bot_v6
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

import numpy as np
import torch
import torch.nn as nn

# ── Ensure training-lab and backend are importable ──────────────────
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "training-lab" / "src"))
sys.path.insert(0, str(_root / "backend" / "src"))
os.chdir(str(_root))

import tarok_engine as te
from training_lab import DecisionType
from training_lab.entities.config import TrainingConfig
from training_lab.entities.experience import Experience
from training_lab.entities.network import TarokNet
from training_lab.adapters.compute.factory import create as create_compute
from training_lab.adapters.storage.file_checkpoint_store import FileCheckpointStore
from training_lab.use_cases.ppo_training import RunPPOTraining

# ── Constants ───────────────────────────────────────────────────────
DT_MAP = {
    0: DecisionType.BID,
    1: DecisionType.KING_CALL,
    2: DecisionType.TALON_PICK,
    3: DecisionType.CARD_PLAY,
}


# ── Helpers ─────────────────────────────────────────────────────────

def _format_time(seconds: float) -> str:
    """Human-friendly elapsed / ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _progress_bar(fraction: float, width: int = 40) -> str:
    filled = int(width * fraction)
    return "[" + "=" * filled + ">" * min(1, width - filled) + "." * max(0, width - filled - 1) + "]"


def _export_torchscript(model: TarokNet, path: str) -> None:
    """Trace model to TorchScript for Rust self-play."""
    class Wrapper(torch.nn.Module):
        def __init__(self, base: TarokNet):
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor):
            s = self.base.shared(x)
            s = self.base.res_blocks(s)
            cf = self.base._extract_card_features(x)
            a = self.base.card_attention(cf)
            f = self.base.fuse(torch.cat([s, a], dim=-1))
            return (
                self.base.bid_head(f),
                self.base.king_head(f),
                self.base.talon_head(f),
                self.base.card_head(f),
                self.base.critic(f).squeeze(-1),
            )

    w = Wrapper(model)
    w.eval()
    with torch.no_grad():
        traced = torch.jit.trace(w, torch.randn(1, 450), check_trace=False)
    traced.save(path)


def _self_play(
    model_path: str,
    n_games: int,
    seat_config: str,
    explore_rate: float,
    concurrency: int = 128,
) -> dict:
    """Run Rust self-play and return raw result dict."""
    return te.run_self_play(
        n_games=n_games,
        concurrency=concurrency,
        model_path=model_path,
        explore_rate=explore_rate,
        seat_config=seat_config,
    )


def _raw_to_experiences(raw: dict) -> dict[int, list[Experience]]:
    """Convert raw Rust output to per-game Experience lists (NN player only)."""
    exps_by_game: dict[int, list[Experience]] = {}
    for i in range(len(raw["actions"])):
        # Only collect experiences from player 0 (our NN agent)
        if int(raw["players"][i]) != 0:
            continue
        gid = int(raw["game_ids"][i])
        scores = raw["scores"]
        # Reward = player-0 score normalised
        reward = float(scores[gid % scores.shape[0], 0]) / 100.0
        exps_by_game.setdefault(gid, []).append(
            Experience(
                state=torch.tensor(raw["states"][i], dtype=torch.float32),
                action=torch.tensor(int(raw["actions"][i]), dtype=torch.long),
                log_prob=torch.tensor(float(raw["log_probs"][i]), dtype=torch.float32),
                value=torch.tensor(float(raw["values"][i]), dtype=torch.float32),
                reward=reward,
                decision_type=DT_MAP[int(raw["decision_types"][i])],
                legal_mask=torch.tensor(raw["legal_masks"][i], dtype=torch.float32),
                game_id=gid,
            )
        )
    return exps_by_game


def _benchmark_placement(model_path: str, n_games: int, seat_config: str, concurrency: int = 128, session_size: int = 50) -> float:
    """Play n_games greedy (NN vs opponents), return avg placement.

    Placement is computed at the SESSION level (cumulative score across
    session_size games), not per-game.  In Tarok the losing team scores 0,
    so per-game placement is meaningless — it just reflects team assignment.
    """
    raw = te.run_self_play(
        n_games=n_games,
        concurrency=concurrency,
        model_path=model_path,
        explore_rate=0.0,  # greedy
        seat_config=seat_config,
    )
    scores = np.array(raw["scores"])  # (n_games, 4)
    n_total = scores.shape[0]
    n_sessions = max(1, n_total // session_size)

    session_placements = []
    for s in range(n_sessions):
        start = s * session_size
        end = min(start + session_size, n_total)
        if start >= n_total:
            break
        cumulative = scores[start:end].sum(axis=0)  # shape (4,)
        # Rank: count how many players scored strictly higher than player 0
        placement = int(np.sum(cumulative > cumulative[0])) + 1
        session_placements.append(placement)

    return float(np.mean(session_placements)) if session_placements else 2.5


# ── Main ────────────────────────────────────────────────────────────

def _load_config(config_path: str, cli_args: argparse.Namespace) -> argparse.Namespace:
    """Load a YAML config and let CLI args override any values."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Map YAML keys to argparse attribute names
    key_map = {
        "seats": "seats",
        "bench_seats": "bench_seats",
        "iterations": "iterations",
        "games": "games",
        "bench_games": "bench_games",
        "ppo_epochs": "ppo_epochs",
        "batch_size": "batch_size",
        "lr": "lr",
        "explore_rate": "explore_rate",
        "device": "device",
        "concurrency": "concurrency",
    }

    for yaml_key, attr in key_map.items():
        if yaml_key in cfg and getattr(cli_args, attr, None) is None:
            setattr(cli_args, attr, cfg[yaml_key])

    return cli_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative PPO training with progress tracking and benchmark evaluation",
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
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML training config (e.g. configs/vs-3-bots.yaml)")
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to pre-trained .pt checkpoint")
    parser.add_argument("--seats", type=str, default=None,
                        help="Seat layout: e.g. nn,bot_v5,bot_v5,bot_v5 (default from config)")
    parser.add_argument("--bench-seats", type=str, default=None,
                        help="Seat layout for benchmarks (default: same as --seats)")
    parser.add_argument("--iterations", "-n", type=int, default=None,
                        help="Number of train→evaluate iterations (default: 10)")
    parser.add_argument("--games", "-g", type=int, default=None,
                        help="Self-play games per training iteration (default: 10000)")
    parser.add_argument("--bench-games", type=int, default=None,
                        help="Benchmark games per evaluation (default: 10000)")
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="PPO epochs per iteration (default: 6)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="PPO mini-batch size (default: 8192)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--explore-rate", type=float, default=None,
                        help="Exploration rate during training self-play (default: 0.10)")
    parser.add_argument("--device", type=str, default=None,
                        help="Training device: auto/cpu/mps/cuda (default: auto)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save iteration checkpoints")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Rust self-play concurrency (default: 128)")
    args = parser.parse_args()

    # Load YAML config if provided, then apply CLI overrides
    if args.config:
        args = _load_config(args.config, args)

    # Apply hardcoded defaults for anything still None
    defaults = dict(
        seats="nn,bot_v5,bot_v5,bot_v5",
        bench_seats=None,  # will fall back to seats
        iterations=10,
        games=10_000,
        bench_games=10_000,
        ppo_epochs=6,
        batch_size=8192,
        lr=3e-4,
        explore_rate=0.10,
        device="auto",
        save_dir="checkpoints/training_run",
        concurrency=128,
    )
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.bench_seats is None:
        args.bench_seats = args.seats

    # ── Load checkpoint ─────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = cp.get("model_state_dict", cp)
    hidden_size = sd["shared.0.weight"].shape[0]
    oracle = any(k.startswith("critic_backbone") for k in sd)
    print(f"  hidden_size={hidden_size}, oracle={oracle}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts_path = str(save_dir / "_current.pt")

    # ── Build model and export TorchScript ──────────────────────────
    model = TarokNet(hidden_size=hidden_size, oracle_critic=oracle)
    model.load_state_dict(sd)
    model.eval()
    _export_torchscript(model, ts_path)

    compute = create_compute(args.device)
    print(f"Training device: {compute.device}")
    print()

    # ── Initial benchmark ───────────────────────────────────────────
    print(f"{'=' * 70}")
    print(f" INITIAL BENCHMARK  ({args.bench_games} games, {args.bench_seats}, greedy)")
    print(f"{'=' * 70}")
    t_bench = time.time()
    initial_placement = _benchmark_placement(ts_path, args.bench_games, args.bench_seats, args.concurrency)
    bench_time = time.time() - t_bench
    print(f"  Avg placement: {initial_placement:.3f}  (1.0 = always 1st, 4.0 = always last)")
    print(f"  Benchmark took {_format_time(bench_time)}")
    print()

    # ── Training loop ───────────────────────────────────────────────
    placements = [initial_placement]
    losses = []
    iter_times = []

    print(f"{'=' * 70}")
    print(f" TRAINING PLAN")
    print(f"   {args.iterations} iterations × {args.games} games/iter")
    print(f"   train seats  = {args.seats}")
    print(f"   bench seats  = {args.bench_seats}")
    print(f"   PPO: {args.ppo_epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"   Benchmark: {args.bench_games} games per iteration (greedy)")
    print(f"{'=' * 70}")
    print()

    overall_t0 = time.time()

    for iteration in range(1, args.iterations + 1):
        iter_t0 = time.time()

        # ── Progress header ─────────────────────────────────────────
        frac = (iteration - 1) / args.iterations
        elapsed_total = time.time() - overall_t0
        if iteration > 1:
            avg_iter_time = elapsed_total / (iteration - 1)
            eta = avg_iter_time * (args.iterations - iteration + 1)
            eta_str = f"ETA {_format_time(eta)}"
        else:
            eta_str = "ETA calculating..."

        bar = _progress_bar(frac)
        print(f"─── Iteration {iteration}/{args.iterations}  {bar} {frac*100:.0f}%  {eta_str} ───")

        # ── Step 1: Self-play (training data) ───────────────────────
        t0 = time.time()
        print(f"  [1/3] Self-play: {args.games} games ({args.seats}, explore={args.explore_rate})...", end="", flush=True)
        raw = _self_play(
            ts_path, args.games,
            seat_config=args.seats,
            explore_rate=args.explore_rate,
            concurrency=args.concurrency,
        )
        n_exps = sum(1 for p in raw["players"] if int(p) == 0)
        sp_time = time.time() - t0
        print(f" {n_exps} exps in {_format_time(sp_time)}")

        # ── Step 2: PPO update ──────────────────────────────────────
        t0 = time.time()
        print(f"  [2/3] PPO update ({args.ppo_epochs} epochs, batch={args.batch_size})...", end="", flush=True)

        exps_by_game = _raw_to_experiences(raw)

        config = TrainingConfig(
            num_sessions=1,
            games_per_session=args.games,
            min_experiences=100,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.batch_size,
            hidden_size=hidden_size,
            oracle_critic=oracle,
            learning_rate=args.lr,
            device=str(compute.device),
        )

        class _DummySim:
            def play_batch(self, *a, **kw):
                return []

        store = FileCheckpointStore(str(save_dir))
        trainer = RunPPOTraining(
            simulator=_DummySim(),
            compute=compute,
            store=store,
            config=config,
            resume_state_dict=sd,
        )

        # Compute GAE and run PPO
        all_data = []
        for gid, game_exps in exps_by_game.items():
            all_data.extend(trainer._compute_gae(game_exps))

        metrics = trainer._ppo_update(all_data)
        ppo_time = time.time() - t0

        loss = metrics["total_loss"]
        p_loss = metrics["policy_loss"]
        v_loss = metrics["value_loss"]
        entropy = metrics["entropy"]
        losses.append(loss)
        print(f" loss={loss:.4f} (p={p_loss:.4f} v={v_loss:.4f} ent={entropy:.4f}) in {_format_time(ppo_time)}")

        # Update weights for next iteration
        sd = trainer.network.state_dict()
        model.load_state_dict(sd)
        model.eval()
        _export_torchscript(model, ts_path)

        # ── Step 3: Benchmark ───────────────────────────────────────
        t0 = time.time()
        print(f"  [3/3] Benchmark: {args.bench_games} games (greedy, {args.bench_seats})...", end="", flush=True)
        placement = _benchmark_placement(ts_path, args.bench_games, args.bench_seats, args.concurrency)
        bench_time = time.time() - t0
        placements.append(placement)
        print(f" placement={placement:.3f} in {_format_time(bench_time)}")

        # ── Save iteration checkpoint ───────────────────────────────
        ckpt_path = save_dir / f"iter_{iteration:03d}.pt"
        torch.save({
            "model_state_dict": sd,
            "hidden_size": hidden_size,
            "oracle_critic": oracle,
            "iteration": iteration,
            "loss": loss,
            "placement": placement,
        }, ckpt_path)

        iter_time = time.time() - iter_t0
        iter_times.append(iter_time)

        # ── Iteration summary ───────────────────────────────────────
        delta = placements[-1] - placements[-2]
        direction = "▲ better!" if delta < 0 else "▼ worse" if delta > 0 else "─ same"
        print(f"  → placement {placements[-2]:.3f} → {placements[-1]:.3f}  ({delta:+.3f} {direction})")
        print(f"  → iteration took {_format_time(iter_time)}")
        print()

    # ── Final summary ───────────────────────────────────────────────
    total_time = time.time() - overall_t0
    bar = _progress_bar(1.0)
    print(f"─── Done  {bar} 100%  Total: {_format_time(total_time)} ───")
    print()
    print(f"{'=' * 70}")
    print(f" RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print()

    # Placement table
    print(f"  {'Iter':>6s}  {'Placement':>10s}  {'Change':>8s}  {'Loss':>10s}")
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 8}  {'─' * 10}")
    print(f"  {'init':>6s}  {placements[0]:>10.3f}  {'':>8s}  {'':>10s}")
    for i in range(1, len(placements)):
        delta = placements[i] - placements[i - 1]
        arrow = "▲" if delta < 0 else "▼" if delta > 0 else "─"
        loss_str = f"{losses[i - 1]:.4f}" if i - 1 < len(losses) else ""
        print(f"  {i:>6d}  {placements[i]:>10.3f}  {delta:>+7.3f}{arrow}  {loss_str:>10s}")

    print()
    overall_delta = placements[-1] - placements[0]
    direction = "IMPROVED" if overall_delta < 0 else "REGRESSED" if overall_delta > 0 else "UNCHANGED"
    print(f"  Overall: {placements[0]:.3f} → {placements[-1]:.3f}  ({overall_delta:+.3f})  {direction}")
    print(f"  Best: {min(placements):.3f} at iteration {placements.index(min(placements))}")
    print()

    # Save best model
    best_iter = placements.index(min(placements))
    if best_iter > 0:
        best_src = save_dir / f"iter_{best_iter:03d}.pt"
        best_dst = save_dir / "best.pt"
        shutil.copy2(best_src, best_dst)
        print(f"  Best model saved to {best_dst}")
    else:
        print(f"  Initial model was best — no improvement achieved.")
        print(f"  Consider: more iterations, higher games/iter, lower LR, or different explore_rate.")

    print()
    print(f"  Total training time: {_format_time(total_time)}")
    print(f"  Avg iteration time:  {_format_time(sum(iter_times) / len(iter_times))}")
    print(f"  Checkpoints saved in {save_dir}/")


if __name__ == "__main__":
    main()
