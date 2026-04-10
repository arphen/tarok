"""Benchmark: StockŠkis v3.2 vs other bot versions.

Runs head-to-head games and reports win rates, average scores,
and per-contract (game mode) performance.

Also runs Rust benchmark for V3 vs V5 proxy data (V3.2 is Python-only,
V5 is Rust-only and cannot face V3.2 directly).

Usage:
    cd backend && uv run python scripts/benchmark_v3_2.py --games 500
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tarok.adapters.ai.stockskis_player import StockSkisPlayer
from tarok.adapters.ai.stockskis_v2 import StockSkisPlayerV2
from tarok.adapters.ai.stockskis_v3 import StockSkisPlayerV3
from tarok.adapters.ai.stockskis_v3_2 import StockSkisPlayerV3_2
from tarok.adapters.ai.stockskis_v4 import StockSkisPlayerV4
from tarok.use_cases.game_loop import GameLoop


async def run_config(
    config_name: str,
    make_players,
    num_games: int,
    seed: int,
    target_version: str,
) -> dict:
    """Run games for one config, return aggregated stats."""

    version_stats: dict[str, dict] = {}
    contract_stats: dict[str, dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "score": 0, "games": 0}
    )
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

        contract_name = state.contract.value if state.contract else "klop"

        for i, player in enumerate(players):
            name = player.name.split("-")[0]  # Group: "V3.2", "V3", etc.
            if name not in version_stats:
                version_stats[name] = {
                    "games": 0, "wins": 0, "total_score": 0,
                    "declarer_games": 0, "declarer_wins": 0,
                }
            s = version_stats[name]
            s["games"] += 1
            s["total_score"] += scores.get(i, 0)
            if scores.get(i, 0) > 0:
                s["wins"] += 1
            if state.declarer == i:
                s["declarer_games"] += 1
                if scores.get(i, 0) > 0:
                    s["declarer_wins"] += 1

        # Track contract outcomes for BOTH sides
        if state.declarer is not None:
            decl_name = players[state.declarer].name.split("-")[0]
            decl_score = scores.get(state.declarer, 0)
            key = f"{decl_name}:{contract_name}"
            contract_stats[key]["games"] += 1
            contract_stats[key]["score"] += decl_score
            if decl_score > 0:
                contract_stats[key]["wins"] += 1
            else:
                contract_stats[key]["losses"] += 1

        # Also track overall contract distribution
        contract_stats[f"ALL:{contract_name}"]["games"] += 1

    elapsed = time.perf_counter() - t0
    gps = (num_games - errors) / elapsed if elapsed > 0 else 0

    return {
        "config_name": config_name,
        "num_games": num_games,
        "errors": errors,
        "elapsed": elapsed,
        "gps": gps,
        "version_stats": version_stats,
        "contract_stats": dict(contract_stats),
        "target_version": target_version,
    }


def print_result(result: dict) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*70}")
    print(f"  {result['config_name']}  ({result['num_games']} games, {result['gps']:.0f} g/s)")
    print(f"{'='*70}")

    print(f"\n  {'Version':<12} {'Games':>6} {'Wins':>6} {'WinRate':>8} {'AvgScore':>10} {'DeclGames':>10} {'DeclWR':>8}")
    print(f"  {'-'*68}")
    for name, s in sorted(result["version_stats"].items()):
        wr = s["wins"] / s["games"] * 100 if s["games"] else 0
        avg = s["total_score"] / s["games"] if s["games"] else 0
        dwr = (s["declarer_wins"] / s["declarer_games"] * 100
               if s["declarer_games"] else 0)
        print(f"  {name:<12} {s['games']:>6} {s['wins']:>6} {wr:>7.1f}% {avg:>+10.1f} {s['declarer_games']:>10} {dwr:>7.1f}%")


def print_contract_breakdown(result: dict, version: str) -> None:
    """Print per-contract stats for a specific version."""
    cs = result["contract_stats"]
    print(f"\n  Contract breakdown for {version} as declarer:")
    print(f"  {'Contract':<22} {'Games':>6} {'Wins':>6} {'WinRate':>8} {'AvgScore':>10}")
    print(f"  {'-'*56}")

    items = [(k, v) for k, v in cs.items()
             if k.startswith(f"{version}:") and v["games"] > 0]
    items.sort(key=lambda x: x[1]["games"], reverse=True)

    total_games = 0
    total_wins = 0
    for key, s in items:
        contract = key.split(":")[1]
        wr = s["wins"] / s["games"] * 100 if s["games"] else 0
        avg = s["score"] / s["games"] if s["games"] else 0
        print(f"  {contract:<22} {s['games']:>6} {s['wins']:>6} {wr:>7.1f}% {avg:>+10.1f}")
        total_games += s["games"]
        total_wins += s["wins"]

    if total_games:
        print(f"  {'TOTAL':<22} {total_games:>6} {total_wins:>6} {total_wins/total_games*100:>7.1f}%")


async def main(num_games: int, seed: int) -> None:
    configs = [
        ("V3.2 (1) vs 3×V2", "V3.2",
         lambda s: [
             StockSkisPlayerV3_2(name="V3.2", seed=s),
             StockSkisPlayerV2(name="V2-a", seed=s + 1),
             StockSkisPlayerV2(name="V2-b", seed=s + 2),
             StockSkisPlayerV2(name="V2-c", seed=s + 3),
         ]),
        ("V3.2 (1) vs 3×V3", "V3.2",
         lambda s: [
             StockSkisPlayerV3_2(name="V3.2", seed=s),
             StockSkisPlayerV3(name="V3-a", seed=s + 1),
             StockSkisPlayerV3(name="V3-b", seed=s + 2),
             StockSkisPlayerV3(name="V3-c", seed=s + 3),
         ]),
        ("V3.2 (1) vs 3×V4", "V3.2",
         lambda s: [
             StockSkisPlayerV3_2(name="V3.2", seed=s),
             StockSkisPlayerV4(name="V4-a", seed=s + 1),
             StockSkisPlayerV4(name="V4-b", seed=s + 2),
             StockSkisPlayerV4(name="V4-c", seed=s + 3),
         ]),
        ("V2 (1) vs 3×V3.2", "V2",
         lambda s: [
             StockSkisPlayerV2(name="V2", seed=s),
             StockSkisPlayerV3_2(name="V3.2-a", seed=s + 1),
             StockSkisPlayerV3_2(name="V3.2-b", seed=s + 2),
             StockSkisPlayerV3_2(name="V3.2-c", seed=s + 3),
         ]),
        ("V3 (1) vs 3×V3.2", "V3",
         lambda s: [
             StockSkisPlayerV3(name="V3", seed=s),
             StockSkisPlayerV3_2(name="V3.2-a", seed=s + 1),
             StockSkisPlayerV3_2(name="V3.2-b", seed=s + 2),
             StockSkisPlayerV3_2(name="V3.2-c", seed=s + 3),
         ]),
    ]

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║   StockŠkis V3.2 Benchmark: Game Mode Analysis                  ║")
    print(f"║   {num_games} games per configuration                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    results = []
    for config_name, target, make_players in configs:
        result = await run_config(config_name, make_players, num_games, seed, target)
        print_result(result)
        print_contract_breakdown(result, target)
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: V3.2 Head-to-Head Win Rates")
    print("=" * 70)
    print(f"  {'Matchup':<30} {'V3.2 WR':>10} {'Opp WR':>10} {'V3.2 Avg':>10}")
    print(f"  {'-'*62}")

    for r in results:
        vs = r["version_stats"]
        v3_2_stats = vs.get("V3.2", {})
        opp_name = [k for k in vs if k != "V3.2"][0] if len(vs) > 1 else "?"
        opp_stats = vs.get(opp_name, {})

        v3_2_wr = (v3_2_stats["wins"] / v3_2_stats["games"] * 100
                   if v3_2_stats.get("games") else 0)
        opp_wr = (opp_stats["wins"] / opp_stats["games"] * 100
                  if opp_stats.get("games") else 0)
        v3_2_avg = (v3_2_stats["total_score"] / v3_2_stats["games"]
                    if v3_2_stats.get("games") else 0)

        print(f"  {r['config_name']:<30} {v3_2_wr:>9.1f}% {opp_wr:>9.1f}% {v3_2_avg:>+10.1f}")

    # V3 vs V5 via Rust engine (proxy for V3.2 vs V5)
    print("\n" + "=" * 70)
    print("  RUST ENGINE: V3 vs V5 (proxy — V3.2 is Python-only, V5 is Rust-only)")
    print("=" * 70)

    try:
        import io
        import contextlib

        import tarok_engine
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tarok_engine.py_run_benchmark(num_games)
        output = buf.getvalue()

        # Extract V3 vs V5 and V5 vs V3 lines
        for line in output.split("\n"):
            stripped = line.strip()
            if any(x in stripped for x in [
                "V5 vs V3", "V3 vs V5",
                "V5 vs V4", "V4 vs V5",
                "GLOBAL SUMMARY",
            ]):
                print(f"  {stripped}")
            # Print header/separator lines near matches
            if "Bot" in stripped and "Games" in stripped and "Wins" in stripped:
                print(f"  {stripped}")
            if stripped.startswith("V") and "%" in stripped:
                print(f"  {stripped}")

        # Also get structured data
        eval_data = tarok_engine.py_eval_vs_bots(num_games)
        print(f"\n  Rust self-play baselines ({num_games} games each):")
        for key in ["vs_v3", "vs_v4", "vs_v5"]:
            if key in eval_data:
                d = eval_data[key]
                print(f"    {key}: WR={d['win_rate']*100:.1f}%, AvgScore={d['avg_score']:+.1f}, DeclWR={d['declarer_wr']*100:.1f}%")

    except ImportError:
        print("  ⚠ tarok_engine not available — skip Rust benchmark")
    except Exception as e:
        print(f"  ⚠ Rust benchmark error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark StockŠkis v3.2")
    parser.add_argument("--games", type=int, default=500, help="Games per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    asyncio.run(main(args.games, args.seed))
