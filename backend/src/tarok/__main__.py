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
