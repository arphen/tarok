"""Benchmark: StockŠkis v1 vs StockŠkis v2.

Run N games with mixed tables and report win rates, average scores,
and per-contract performance.

Usage:
    uv run python -m scripts.benchmark_stockskis --games 200
    # or from backend/:
    uv run python scripts/benchmark_stockskis.py --games 200
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure backend src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tarok.adapters.ai.stockskis_player import StockSkisPlayer
from tarok.adapters.ai.stockskis_v2 import StockSkisPlayerV2
from tarok.entities.game_state import Phase
from tarok.use_cases.game_loop import GameLoop


async def run_benchmark(num_games: int, seed: int = 42) -> None:
    """Run head-to-head benchmark between v1 and v2 bots."""

    # Setup: player 0 = v2 bot, players 1-3 = v1 bots
    # Then flip: player 0 = v1 bot, players 1-3 = v2 bots
    # This controls for positional advantage.

    configs = [
        ("v2 vs 3×v1", lambda s: [
            StockSkisPlayerV2(name="V2", seed=s),
            StockSkisPlayer(name="V1-a", seed=s + 1),
            StockSkisPlayer(name="V1-b", seed=s + 2),
            StockSkisPlayer(name="V1-c", seed=s + 3),
        ]),
        ("v1 vs 3×v2", lambda s: [
            StockSkisPlayer(name="V1", seed=s),
            StockSkisPlayerV2(name="V2-a", seed=s + 1),
            StockSkisPlayerV2(name="V2-b", seed=s + 2),
            StockSkisPlayerV2(name="V2-c", seed=s + 3),
        ]),
        ("2×v2 vs 2×v1", lambda s: [
            StockSkisPlayerV2(name="V2-a", seed=s),
            StockSkisPlayer(name="V1-a", seed=s + 1),
            StockSkisPlayerV2(name="V2-b", seed=s + 2),
            StockSkisPlayer(name="V1-b", seed=s + 3),
        ]),
    ]

    for config_name, make_players in configs:
        print(f"\n{'='*60}")
        print(f"  {config_name}  ({num_games} games)")
        print(f"{'='*60}")

        stats: dict[str, dict] = {}
        contract_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "score": 0})
        errors = 0

        t0 = time.perf_counter()

        for g in range(num_games):
            players = make_players(seed + g * 10)
            game = GameLoop(players)

            try:
                state, scores = await game.run(dealer=g % 4)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Game {g} error: {e}")
                continue

            contract_name = state.contract.value if state.contract else "none"

            for i, player in enumerate(players):
                name = player.name
                if name not in stats:
                    stats[name] = {
                        "games": 0, "wins": 0, "total_score": 0,
                        "declarer_games": 0, "declarer_wins": 0,
                    }
                s = stats[name]
                s["games"] += 1
                s["total_score"] += scores.get(i, 0)
                if scores.get(i, 0) > 0:
                    s["wins"] += 1
                if state.declarer == i:
                    s["declarer_games"] += 1
                    if scores.get(i, 0) > 0:
                        s["declarer_wins"] += 1

            # Track contract outcomes
            if state.declarer is not None:
                decl_name = players[state.declarer].name
                decl_score = scores.get(state.declarer, 0)
                key = f"{decl_name}:{contract_name}"
                if decl_score > 0:
                    contract_stats[key]["wins"] += 1
                else:
                    contract_stats[key]["losses"] += 1
                contract_stats[key]["score"] += decl_score

        elapsed = time.perf_counter() - t0
        games_per_sec = (num_games - errors) / elapsed if elapsed > 0 else 0

        print(f"\n  {num_games - errors} games completed in {elapsed:.1f}s ({games_per_sec:.0f} games/s)")
        if errors:
            print(f"  {errors} errors")

        # Aggregate by version
        v2_stats = {"games": 0, "wins": 0, "total_score": 0, "declarer_games": 0, "declarer_wins": 0}
        v1_stats = {"games": 0, "wins": 0, "total_score": 0, "declarer_games": 0, "declarer_wins": 0}

        for name, s in stats.items():
            target = v2_stats if "V2" in name else v1_stats
            for k in target:
                target[k] += s[k]

        print(f"\n  {'Player':<12} {'Games':>6} {'Wins':>6} {'WinRate':>8} {'AvgScore':>10} {'DeclGames':>10} {'DeclWR':>8}")
        print(f"  {'-'*66}")
        for label, s in [("V2 (total)", v2_stats), ("V1 (total)", v1_stats)]:
            wr = s["wins"] / s["games"] * 100 if s["games"] else 0
            avg = s["total_score"] / s["games"] if s["games"] else 0
            dwr = s["declarer_wins"] / s["declarer_games"] * 100 if s["declarer_games"] else 0
            print(f"  {label:<12} {s['games']:>6} {s['wins']:>6} {wr:>7.1f}% {avg:>+10.1f} {s['declarer_games']:>10} {dwr:>7.1f}%")

        # Per-player breakdown
        print(f"\n  Per-player breakdown:")
        for name, s in sorted(stats.items()):
            wr = s["wins"] / s["games"] * 100 if s["games"] else 0
            avg = s["total_score"] / s["games"] if s["games"] else 0
            print(f"    {name:<10} {s['games']:>5}g  {wr:>5.1f}% WR  {avg:>+8.1f} avg")

        # Contract breakdown (top 5 most played)
        if contract_stats:
            print(f"\n  Contract breakdown (as declarer):")
            sorted_contracts = sorted(contract_stats.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True)
            for key, cs in sorted_contracts[:10]:
                total = cs["wins"] + cs["losses"]
                wr = cs["wins"] / total * 100 if total else 0
                avg = cs["score"] / total if total else 0
                print(f"    {key:<20} {total:>4} games  {wr:>5.1f}% WR  {avg:>+8.1f} avg")


def main():
    parser = argparse.ArgumentParser(description="Benchmark StockŠkis v1 vs v2")
    parser.add_argument("--games", type=int, default=100, help="Number of games per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.games, args.seed))


if __name__ == "__main__":
    main()
