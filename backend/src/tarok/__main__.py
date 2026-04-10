"""CLI entry point — run the server or training."""

import asyncio
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        num_sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        games_per_session = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        asyncio.run(run_training(num_sessions, games_per_session))
    elif cmd == "evolve":
        asyncio.run(run_evolution())
    elif cmd == "train-evolved":
        results_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/evo_results/evo_best_hparams.json"
        num_sessions = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        games_per_session = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        asyncio.run(run_train_evolved(results_path, num_sessions, games_per_session))
    elif cmd == "breed":
        asyncio.run(run_breeding_cli())
    elif cmd == "train-bred":
        checkpoint = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/breeding_results/bred_model_final.pt"
        num_sessions = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        games_per_session = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        asyncio.run(run_train_bred(checkpoint, num_sessions, games_per_session))
    elif cmd == "pipeline":
        asyncio.run(run_pipeline_cli())
    elif cmd == "generate-expert-data":
        asyncio.run(run_generate_expert_data())
    elif cmd == "imitation-pretrain":
        asyncio.run(run_imitation_pretrain_cli())
    else:
        run_server()


def run_server():
    import uvicorn
    uvicorn.run(
        "tarok.adapters.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


async def run_training(num_sessions: int, games_per_session: int):
    import os
    import torch
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.trainer import PPOTrainer

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    oracle = "--oracle" in sys.argv
    fsp = "--no-fsp" not in sys.argv  # FSP on by default

    total_games = num_sessions * games_per_session
    print(f"🧠 Starting session-based training: {num_sessions} sessions × {games_per_session} games = {total_games} games")
    print(f"   Device: {device}")
    print(f"   Oracle Critic (PTIE): {'ON' if oracle else 'OFF'}")
    print(f"   Fictitious Self-Play: {'ON' if fsp else 'OFF'}")
    print(f"   Agent learns bidding, king calling, talon selection, announcements, and card play via PPO")
    print()

    agents = [
        RLAgent(name=f"Agent-{i}", device=device, oracle_critic=oracle)
        for i in range(4)
    ]
    trainer = PPOTrainer(
        agents,
        device=device,
        games_per_session=games_per_session,
        fsp_ratio=0.3 if fsp else 0.0,
    )

    async def print_metrics(metrics):
        bar_len = 30
        filled = int(bar_len * metrics.episode / metrics.total_episodes)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r[{bar}] S{metrics.session}/{metrics.total_sessions} G{metrics.episode}/{metrics.total_episodes} | "
            f"Win: {metrics.win_rate:.1%} | "
            f"Reward: {metrics.avg_reward:+.2f} | "
            f"Bid: {metrics.bid_rate:.0%} | "
            f"Klop: {metrics.klop_rate:.0%} | "
            f"Solo: {metrics.solo_rate:.0%} | "
            f"{metrics.games_per_second:.1f} g/s",
            end="",
            flush=True,
        )

    trainer.add_metrics_callback(print_metrics)
    result = await trainer.train(num_sessions)

    print()
    print()
    print("✅ Training complete!")
    print(f"   Final win rate: {result.win_rate:.1%}")
    print(f"   Final avg reward: {result.avg_reward:+.2f}")
    print(f"   Throughput: {result.games_per_second:.1f} games/sec")
    print()

    # Per-contract breakdown
    d = result.to_dict()
    print("   Contract breakdown (as declarer → as defender):")
    print(f"   {'Contract':<14} {'Decl':>5} {'D.Win%':>7} {'D.Avg':>7} │ {'Def':>5} {'Def.Win%':>8} {'Def.Avg':>8}")
    print(f"   {'─'*14} {'─'*5} {'─'*7} {'─'*7} │ {'─'*5} {'─'*8} {'─'*8}")
    for cname, cs in d["contract_stats"].items():
        if cs["played"] > 0:
            dp = cs["decl_played"]
            dw = f"{cs['decl_win_rate']:.0%}" if dp > 0 else "—"
            da = f"{cs['decl_avg_score']:+.1f}" if dp > 0 else "—"
            fp = cs["def_played"]
            fw = f"{cs['def_win_rate']:.0%}" if fp > 0 else "—"
            fa = f"{cs['def_avg_score']:+.1f}" if fp > 0 else "—"
            print(f"   {cname:<14} {dp:>5} {dw:>7} {da:>7} │ {fp:>5} {fw:>8} {fa:>8}")
    print()
    # Per-session avg score trend
    scores_hist = result.session_avg_score_history
    if scores_hist:
        first5 = sum(scores_hist[:5]) / min(5, len(scores_hist))
        last5 = sum(scores_hist[-5:]) / min(5, len(scores_hist))
        print(f"   Session avg score: first 5 sessions={first5:+.1f} → last 5 sessions={last5:+.1f}")
        print()
    print(f"   Snapshots saved: {len(result.snapshots)}")
    print(f"   Checkpoint: checkpoints/tarok_agent_latest.pt")


if __name__ == "__main__":
    main()


async def run_evolution():
    import torch
    from tarok.adapters.ai.evo_optimizer import EvoConfig, run_evolution as _run_evo

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    oracle = "--oracle" in sys.argv
    pop = 12
    gens = 10
    eval_sessions = 20
    games_per_session = 10

    # Parse optional CLI overrides: --pop N --gens N --eval-sessions N --eval-games N
    for i, arg in enumerate(sys.argv):
        if arg == "--pop" and i + 1 < len(sys.argv):
            pop = int(sys.argv[i + 1])
        elif arg == "--gens" and i + 1 < len(sys.argv):
            gens = int(sys.argv[i + 1])
        elif arg == "--eval-sessions" and i + 1 < len(sys.argv):
            eval_sessions = int(sys.argv[i + 1])
        elif arg == "--eval-games" and i + 1 < len(sys.argv):
            games_per_session = int(sys.argv[i + 1])

    config = EvoConfig(
        population_size=pop,
        num_generations=gens,
        eval_sessions=eval_sessions,
        games_per_session=games_per_session,
        oracle=oracle,
        device=device,
    )

    result = await _run_evo(config)
    print()
    print(f"🏆 Evolution complete! Best fitness: {result['best_fitness']:.4f}")
    print(f"   Run `uv run python -m tarok train-evolved` to train with the best hyperparameters.")


async def run_train_evolved(results_path: str, num_sessions: int, games_per_session: int):
    import torch
    from tarok.adapters.ai.evo_optimizer import train_with_best

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    oracle = "--oracle" in sys.argv

    print(f"🧬 Training with evolved hyperparameters from {results_path}")
    print(f"   {num_sessions} sessions × {games_per_session} games")
    print()

    result = await train_with_best(
        results_path, num_sessions, games_per_session,
        oracle=oracle, device=device,
    )

    print()
    print("✅ Evolved training complete!")
    print(f"   Final win rate: {result.win_rate:.1%}")
    print(f"   Final avg reward: {result.avg_reward:+.2f}")


async def run_breeding_cli():
    import torch
    from tarok.adapters.ai.breeding import BreedingConfig, run_breeding

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    oracle = "--oracle" in sys.argv
    warmup = 50
    pop = 12
    gens = 5
    cycles = 3
    eval_games = 100
    refine = 30

    for i, arg in enumerate(sys.argv):
        if arg == "--warmup" and i + 1 < len(sys.argv):
            warmup = int(sys.argv[i + 1])
        elif arg == "--pop" and i + 1 < len(sys.argv):
            pop = int(sys.argv[i + 1])
        elif arg == "--gens" and i + 1 < len(sys.argv):
            gens = int(sys.argv[i + 1])
        elif arg == "--cycles" and i + 1 < len(sys.argv):
            cycles = int(sys.argv[i + 1])
        elif arg == "--eval-games" and i + 1 < len(sys.argv):
            eval_games = int(sys.argv[i + 1])
        elif arg == "--refine" and i + 1 < len(sys.argv):
            refine = int(sys.argv[i + 1])

    config = BreedingConfig(
        warmup_sessions=warmup,
        population_size=pop,
        num_generations=gens,
        num_cycles=cycles,
        eval_games=eval_games,
        refine_sessions=refine,
        oracle=oracle,
        device=device,
    )

    await run_breeding(config)


async def run_train_bred(checkpoint: str, num_sessions: int, games_per_session: int):
    import torch
    from tarok.adapters.ai.breeding import train_with_bred_model

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    oracle = "--oracle" in sys.argv

    result = await train_with_bred_model(
        checkpoint, num_sessions, games_per_session,
        oracle=oracle, device=device,
    )

    print()
    print("✅ Bred model training complete!")
    print(f"   Final win rate: {result.win_rate:.1%}")
    print(f"   Final avg reward: {result.avg_reward:+.2f}")


# ──────────────────────────────────────────────
# Expert data generation (standalone)
# ──────────────────────────────────────────────

async def run_generate_expert_data():
    """Generate expert games from StockŠkis bots (Rust speed)."""
    num_games = 1_000_000
    include_oracle = "--oracle" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == "--games" and i + 1 < len(sys.argv):
            num_games = int(sys.argv[i + 1])

    import time
    import tarok_engine as te

    print(f"🎮 Generating {num_games:,} StockŠkis expert games...")
    print(f"   Oracle states: {'ON' if include_oracle else 'OFF'}")
    t0 = time.time()
    data = te.generate_expert_data(num_games, include_oracle=include_oracle)
    elapsed = time.time() - t0

    n = data["num_experiences"]
    print(f"   Generated {n:,} expert experiences in {elapsed:.1f}s")
    print(f"   Speed: {num_games/elapsed:,.0f} games/sec")
    print(f"   Avg {n/num_games:.1f} decisions/game")


# ──────────────────────────────────────────────
# Imitation pre-training (standalone)
# ──────────────────────────────────────────────

async def run_imitation_pretrain_cli():
    """Pre-train from StockŠkis expert games (policy + value)."""
    import torch
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.imitation import imitation_pretrain

    num_games = 1_000_000
    oracle = "--oracle" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == "--games" and i + 1 < len(sys.argv):
            num_games = int(sys.argv[i + 1])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    agent = RLAgent(name="Agent-0", device=device, oracle_critic=oracle)

    print(f"🧠 Imitation pre-training from {num_games:,} StockŠkis expert games")
    print(f"   Device: {device}")
    print(f"   Oracle: {'ON' if oracle else 'OFF'}")
    print()

    def on_progress(info: dict):
        print(
            f"\r   Chunk {info['chunk']} | "
            f"Games: {info['games_done']:,}/{info['total_games']:,} | "
            f"Exps: {info['experiences']:,} | "
            f"Policy: {info['policy_loss']:.4f} | "
            f"Value: {info['value_loss']:.4f} | "
            f"Gen: {info['gen_speed']:,} g/s | "
            f"{info['elapsed']:.0f}s",
            end="", flush=True,
        )

    result = imitation_pretrain(
        network=agent.network,
        num_games=num_games,
        chunk_size=50_000,
        device=device,
        include_oracle=oracle,
        progress_callback=on_progress,
    )

    # Save checkpoint
    from pathlib import Path
    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "tarok_agent_imitation.pt"
    torch.save({
        "model_state_dict": agent.network.state_dict(),
        "imitation_stats": result,
    }, path)

    print()
    print()
    print("✅ Imitation pre-training complete!")
    print(f"   Total experiences: {result['total_experiences']:,}")
    print(f"   Policy loss: {result['avg_policy_loss']:.4f}")
    print(f"   Value loss: {result['avg_value_loss']:.4f}")
    print(f"   Time: {result['elapsed_secs']:.0f}s ({result['games_per_sec']:,} games/sec)")
    print(f"   Checkpoint: {path}")


# ──────────────────────────────────────────────
# Full 3-phase training pipeline
# ──────────────────────────────────────────────

async def run_pipeline_cli():
    """Run the full 3-phase training pipeline:

    Phase 1: Imitation pre-training from StockŠkis expert games
    Phase 2: Fine-tune with PPO against StockŠkis bots (monitor plateau)
    Phase 3: Fictitious self-play against frozen past selves
    """
    import torch
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.imitation import imitation_pretrain
    from tarok.adapters.ai.trainer import PPOTrainer

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    oracle = "--oracle" in sys.argv

    # Default pipeline parameters
    p1_games = 1_000_000     # Phase 1: expert games for imitation
    p2_sessions = 200        # Phase 2: sessions vs StockŠkis
    p2_games_per = 50        # Phase 2: games per session
    p2_plateau_window = 20   # Phase 2: window for plateau detection
    p2_plateau_thresh = 0.01 # Phase 2: win rate improvement threshold
    p3_sessions = 500        # Phase 3: self-play sessions
    p3_games_per = 50        # Phase 3: games per session

    # Parse CLI overrides
    for i, arg in enumerate(sys.argv):
        if arg == "--p1-games" and i + 1 < len(sys.argv):
            p1_games = int(sys.argv[i + 1])
        elif arg == "--p2-sessions" and i + 1 < len(sys.argv):
            p2_sessions = int(sys.argv[i + 1])
        elif arg == "--p2-games" and i + 1 < len(sys.argv):
            p2_games_per = int(sys.argv[i + 1])
        elif arg == "--p3-sessions" and i + 1 < len(sys.argv):
            p3_sessions = int(sys.argv[i + 1])
        elif arg == "--p3-games" and i + 1 < len(sys.argv):
            p3_games_per = int(sys.argv[i + 1])
        elif arg == "--plateau-window" and i + 1 < len(sys.argv):
            p2_plateau_window = int(sys.argv[i + 1])
        elif arg == "--plateau-thresh" and i + 1 < len(sys.argv):
            p2_plateau_thresh = float(sys.argv[i + 1])

    print("=" * 60)
    print("  TAROK 3-PHASE TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Oracle Critic: {'ON' if oracle else 'OFF'}")
    print()
    print(f"  Phase 1: Imitation from {p1_games:,} StockŠkis games")
    print(f"  Phase 2: PPO vs StockŠkis — up to {p2_sessions} sessions × {p2_games_per} games")
    print(f"           Plateau: {p2_plateau_window}-session window, {p2_plateau_thresh:.1%} threshold")
    print(f"  Phase 3: Self-play — {p3_sessions} sessions × {p3_games_per} games")
    print("=" * 60)
    print()

    # Create agents
    agents = [
        RLAgent(name=f"Agent-{i}", device=device, oracle_critic=oracle)
        for i in range(4)
    ]

    # ── Phase 1: Imitation pre-training ──────────────────────────
    print("━" * 60)
    print("  PHASE 1: Imitation Pre-Training (StockŠkis Expert Games)")
    print("━" * 60)

    def p1_progress(info: dict):
        pct = info["games_done"] / info["total_games"] * 100
        print(
            f"\r  [{pct:5.1f}%] "
            f"Games: {info['games_done']:,}/{info['total_games']:,} | "
            f"Policy: {info['policy_loss']:.4f} | "
            f"Value: {info['value_loss']:.4f} | "
            f"{info['gen_speed']:,} g/s",
            end="", flush=True,
        )

    p1_result = imitation_pretrain(
        network=agents[0].network,
        num_games=p1_games,
        chunk_size=50_000,
        device=device,
        include_oracle=oracle,
        progress_callback=p1_progress,
    )

    print()
    print(f"  ✓ Phase 1 complete: {p1_result['total_experiences']:,} experiences, "
          f"policy={p1_result['avg_policy_loss']:.4f}, "
          f"value={p1_result['avg_value_loss']:.4f}")
    print()

    # Sync network to all agents (imitation trained agent 0's network)
    for agent in agents:
        agent.network = agents[0].network

    # Save Phase 1 checkpoint
    from pathlib import Path
    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": agents[0].network.state_dict(),
        "phase": "imitation",
        "stats": p1_result,
    }, save_dir / "tarok_agent_phase1.pt")

    # ── Phase 2: PPO vs StockŠkis ───────────────────────────────
    print("━" * 60)
    print("  PHASE 2: PPO Fine-Tuning vs StockŠkis Bots")
    print("━" * 60)

    trainer = PPOTrainer(
        agents,
        device=device,
        games_per_session=p2_games_per,
        stockskis_ratio=1.0,   # 100% games vs StockŠkis in Phase 2
        stockskis_strength=1.0,
        fsp_ratio=0.0,          # No self-play yet
        use_rust_engine=False,   # Need Python engine for StockŠkis PlayerPort
    )

    # Track win rate for plateau detection
    win_rate_history: list[float] = []
    plateau_reached = False

    async def p2_metrics(metrics):
        bar_len = 20
        filled = int(bar_len * metrics.session / p2_sessions)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] S{metrics.session}/{p2_sessions} | "
            f"Win: {metrics.win_rate:.1%} | "
            f"Reward: {metrics.avg_reward:+.2f} | "
            f"Bid: {metrics.bid_rate:.0%} | "
            f"{metrics.games_per_second:.1f} g/s",
            end="", flush=True,
        )

    trainer.add_metrics_callback(p2_metrics)

    # Run Phase 2 in manual session loop for plateau detection
    trainer._running = True
    total_p2_games = 0

    for session_idx in range(p2_sessions):
        if not trainer._running:
            break

        # Run one session
        result = await trainer.train(1)
        total_p2_games += p2_games_per

        win_rate_history.append(result.win_rate)

        # Plateau detection: check if win rate has stopped improving
        if len(win_rate_history) >= p2_plateau_window * 2:
            early = win_rate_history[-p2_plateau_window * 2:-p2_plateau_window]
            recent = win_rate_history[-p2_plateau_window:]
            early_avg = sum(early) / len(early)
            recent_avg = sum(recent) / len(recent)
            improvement = recent_avg - early_avg

            if improvement < p2_plateau_thresh and session_idx >= p2_plateau_window * 3:
                print()
                print(f"  ⚡ Plateau detected at session {session_idx + 1}: "
                      f"win rate {early_avg:.1%} → {recent_avg:.1%} "
                      f"(Δ={improvement:+.2%} < {p2_plateau_thresh:.1%})")
                plateau_reached = True
                break

    print()
    p2_final_wr = win_rate_history[-1] if win_rate_history else 0.0
    print(f"  ✓ Phase 2 complete: {total_p2_games:,} games, "
          f"final win rate: {p2_final_wr:.1%}"
          f"{' (plateau)' if plateau_reached else ''}")
    print()

    # Save Phase 2 checkpoint
    torch.save({
        "model_state_dict": agents[0].network.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "phase": "stockskis",
        "win_rate": p2_final_wr,
        "sessions": len(win_rate_history),
    }, save_dir / "tarok_agent_phase2.pt")

    # ── Phase 3: Fictitious Self-Play ────────────────────────────
    print("━" * 60)
    print("  PHASE 3: Fictitious Self-Play (GTO Convergence)")
    print("━" * 60)

    # New trainer with FSP enabled, no StockŠkis
    trainer3 = PPOTrainer(
        agents,
        device=device,
        games_per_session=p3_games_per,
        stockskis_ratio=0.0,     # No more bots
        fsp_ratio=0.3,           # 30% vs historical selves
        bank_size=20,
        bank_save_interval=5,
        use_rust_engine=True,    # Rust engine for speed
    )

    # Seed the network bank with current weights
    trainer3.network_bank.push(agents[0].network.state_dict())

    async def p3_metrics(metrics):
        bar_len = 20
        filled = int(bar_len * metrics.session / p3_sessions)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] S{metrics.session}/{p3_sessions} | "
            f"Win: {metrics.win_rate:.1%} | "
            f"Reward: {metrics.avg_reward:+.2f} | "
            f"Entropy: {metrics.entropy:.3f} | "
            f"{metrics.games_per_second:.1f} g/s",
            end="", flush=True,
        )

    trainer3.add_metrics_callback(p3_metrics)
    p3_result = await trainer3.train(p3_sessions)

    print()
    print(f"  ✓ Phase 3 complete: {p3_sessions * p3_games_per:,} games, "
          f"final win rate: {p3_result.win_rate:.1%}")
    print()

    # Save final checkpoint
    torch.save({
        "model_state_dict": agents[0].network.state_dict(),
        "optimizer_state_dict": trainer3.optimizer.state_dict(),
        "phase": "pipeline_complete",
        "p1_stats": p1_result,
        "p2_win_rate": p2_final_wr,
        "p3_win_rate": p3_result.win_rate,
        "p3_avg_reward": p3_result.avg_reward,
    }, save_dir / "tarok_agent_pipeline.pt")

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Phase 1 (Imitation):  {p1_result['total_experiences']:,} expert experiences")
    print(f"  Phase 2 (StockŠkis):  Win rate {p2_final_wr:.1%}")
    print(f"  Phase 3 (Self-Play):  Win rate {p3_result.win_rate:.1%}, "
          f"Reward {p3_result.avg_reward:+.2f}")
    print()
    print("  Checkpoints:")
    print(f"    Phase 1: checkpoints/tarok_agent_phase1.pt")
    print(f"    Phase 2: checkpoints/tarok_agent_phase2.pt")
    print(f"    Final:   checkpoints/tarok_agent_pipeline.pt")
    print("=" * 60)

