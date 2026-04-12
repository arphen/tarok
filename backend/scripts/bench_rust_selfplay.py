#!/usr/bin/env python3
"""Benchmark the pure-Rust self-play loop with tch-rs inference."""

import sys
import time
import tempfile
from pathlib import Path

import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tarok.adapters.ai.network import TarokNet


class AllHeadsModel(torch.nn.Module):
    """Wrapper that runs all 5 heads in a single forward pass for TorchScript."""

    def __init__(self, net: TarokNet):
        super().__init__()
        self.net = net

    def forward(self, state: torch.Tensor) -> tuple:
        h = self.net.shared(state)
        h = self.net.res_blocks(h)
        card_feats = self.net._extract_card_features(state)
        attn_out = self.net.card_attention(card_feats)
        h = self.net.fuse(torch.cat([h, attn_out], dim=-1))

        bid = self.net.bid_head(h)
        king = self.net.king_head(h)
        talon = self.net.talon_head(h)
        card = self.net.card_head(h)
        value = self.net.critic(h).squeeze(-1)
        return bid, king, talon, card, value


def export_torchscript(checkpoint_path: str | None = None) -> str:
    """Export TarokNet to TorchScript, return path."""
    net = TarokNet(hidden_size=256)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        net.load_state_dict(sd, strict=False)
    net.eval()

    wrapper = AllHeadsModel(net)
    wrapper.eval()

    dummy = torch.randn(1, 450)
    traced = torch.jit.trace(wrapper, dummy, check_trace=False)

    path = tempfile.mktemp(suffix=".pt", prefix="tarok_traced_")
    traced.save(path)
    print(f"Exported TorchScript model to {path} ({Path(path).stat().st_size // 1024} KB)")
    return path


def benchmark(model_path: str, n_games: int = 100, concurrency: int = 64):
    """Run the Rust self-play benchmark."""
    import tarok_engine

    print(f"\n--- Rust self-play benchmark ---")
    print(f"Games: {n_games}, Concurrency: {concurrency}")

    t0 = time.perf_counter()
    result = tarok_engine.run_self_play(model_path, n_games, concurrency, 0.05)
    elapsed = time.perf_counter() - t0

    n_exp = result["n_experiences"]
    games_per_sec = n_games / elapsed
    exp_per_sec = n_exp / elapsed

    print(f"Completed {n_games} games in {elapsed:.2f}s")
    print(f"  {games_per_sec:.1f} games/s")
    print(f"  {n_exp} experiences ({exp_per_sec:.0f} exp/s)")
    print(f"  States shape: {result['states'].shape}")
    print(f"  Actions shape: {result['actions'].shape}")
    print(f"  Scores shape: {result['scores'].shape}")

    # Print some score statistics
    import numpy as np
    scores = result["scores"]
    print(f"\n  Mean scores per player: {scores.mean(axis=0)}")
    print(f"  Std scores per player:  {scores.std(axis=0)}")

    return games_per_sec


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Rust self-play")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint (.pt)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()

    model_path = export_torchscript(args.checkpoint)

    # Warmup
    print("\nWarmup run...")
    import tarok_engine
    tarok_engine.run_self_play(model_path, 10, 8, 0.05)

    # Benchmark
    benchmark(model_path, args.games, args.concurrency)

    # Compare with different concurrency levels
    print("\n--- Concurrency sweep ---")
    for c in [8, 32, 64, 128]:
        gps = benchmark(model_path, args.games, c)


if __name__ == "__main__":
    main()
