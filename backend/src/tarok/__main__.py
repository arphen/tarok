"""CLI entry point — run the server."""

import asyncio
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "generate-expert-data":
        asyncio.run(run_generate_expert_data())
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


async def run_generate_expert_data():
    """Generate expert games from StockŠkis bots (Rust speed)."""
    num_games = 1_000_000

    for i, arg in enumerate(sys.argv):
        if arg == "--games" and i + 1 < len(sys.argv):
            num_games = int(sys.argv[i + 1])

    import time
    import tarok_engine as te

    include_oracle = "--oracle" in sys.argv
    print(f"Generating {num_games:,} StockŠkis expert games...")
    t0 = time.time()
    data = te.generate_expert_data(num_games, include_oracle=include_oracle)
    elapsed = time.time() - t0

    n = data["num_experiences"]
    print(f"Generated {n:,} expert experiences in {elapsed:.1f}s")
    print(f"Speed: {num_games/elapsed:,.0f} games/sec")


if __name__ == "__main__":
    main()

