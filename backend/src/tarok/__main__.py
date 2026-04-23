"""CLI entry point — run the server."""

import asyncio
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "generate-expert-data":
        asyncio.run(run_generate_expert_data())
    elif cmd == "generate-dd-data":
        asyncio.run(run_generate_dd_data())
    elif cmd == "dd-pretrain":
        asyncio.run(run_dd_pretrain())
    elif cmd == "arena-duplicate":
        run_arena_duplicate(sys.argv[2:])
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
    print(f"Speed: {num_games / elapsed:,.0f} games/sec")


async def run_generate_dd_data():
    """Generate DD-labeled training data from random games."""
    num_games = 1_000

    for i, arg in enumerate(sys.argv):
        if arg == "--games" and i + 1 < len(sys.argv):
            num_games = int(sys.argv[i + 1])

    import time
    import tarok_engine as te

    include_oracle = "--oracle" in sys.argv
    print(f"Generating DD training data from {num_games:,} games...")
    t0 = time.time()
    data = te.generate_dd_training_data(num_games, include_oracle=include_oracle)
    elapsed = time.time() - t0

    n = data["num_experiences"]
    print(f"Generated {n:,} DD-labeled experiences in {elapsed:.1f}s")
    print(f"Speed: {num_games / elapsed:.1f} games/sec ({n / elapsed:.0f} exps/sec)")
    print(f"DD values range: [{data['dd_values'].min():.3f}, {data['dd_values'].max():.3f}]")


async def run_dd_pretrain():
    """Pre-train network using DD-solved perfect-play labels."""
    import time

    import numpy as np
    import tarok_engine as te
    import torch
    import torch.nn.functional as F

    from tarok_model.network import TarokNet

    num_games = 10_000
    num_epochs = 20
    batch_size = 256
    lr = 1e-3
    save_path = None
    resume_from = None
    include_oracle = True

    for i, arg in enumerate(sys.argv):
        if arg == "--games" and i + 1 < len(sys.argv):
            num_games = int(sys.argv[i + 1])
        elif arg == "--epochs" and i + 1 < len(sys.argv):
            num_epochs = int(sys.argv[i + 1])
        elif arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
        elif arg == "--lr" and i + 1 < len(sys.argv):
            lr = float(sys.argv[i + 1])
        elif arg == "--save" and i + 1 < len(sys.argv):
            save_path = sys.argv[i + 1]
        elif arg == "--resume" and i + 1 < len(sys.argv):
            resume_from = sys.argv[i + 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate DD training data
    print(f"Generating DD training data from {num_games:,} games...")
    t0 = time.time()
    data = te.generate_dd_training_data(num_games, include_oracle=include_oracle)
    elapsed = time.time() - t0
    n = data["num_experiences"]
    print(f"Generated {n:,} DD-labeled experiences in {elapsed:.1f}s")

    # Convert to tensors
    states = torch.tensor(np.array(data["states"]), dtype=torch.float32, device=device)
    dd_values = torch.tensor(np.array(data["dd_values"]), dtype=torch.float32, device=device)
    dd_best_moves = torch.tensor(np.array(data["dd_best_moves"]), dtype=torch.long, device=device)
    dd_move_values = torch.tensor(
        np.array(data["dd_move_values"]), dtype=torch.float32, device=device
    )
    legal_masks = torch.tensor(np.array(data["legal_masks"]), dtype=torch.float32, device=device)

    oracle_states = None
    if data["oracle_states"] is not None:
        oracle_states = torch.tensor(
            np.array(data["oracle_states"]), dtype=torch.float32, device=device
        )

    # Create or load network
    net = TarokNet(hidden_size=256, oracle_critic=include_oracle).to(device)
    if resume_from:
        print(f"Resuming from {resume_from}")
        net.load_state_dict(torch.load(resume_from, map_location=device, weights_only=True))

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print(f"\nDD Pre-training: {num_epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Training data: {n:,} experiences")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(n, device=device)
        states_s = states[perm]
        dd_values_s = dd_values[perm]
        dd_best_moves_s = dd_best_moves[perm]
        dd_move_values_s = dd_move_values[perm]
        legal_masks_s = legal_masks[perm]
        oracle_s = oracle_states[perm] if oracle_states is not None else None

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_rank_loss = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            s_batch = states_s[start:end]
            v_batch = dd_values_s[start:end]
            bm_batch = dd_best_moves_s[start:end]
            mv_batch = dd_move_values_s[start:end]
            lm_batch = legal_masks_s[start:end]

            # Forward pass — card play head only (DD data is card play only)
            features = net.shared(s_batch)
            features = net.res_blocks(features)

            # Card attention
            card_feats = net._extract_card_features(s_batch)
            attn_out = net.card_attention(card_feats)
            features = net.fuse(torch.cat([features, attn_out], dim=-1))

            # Card play logits
            logits = net.card_head(features)

            # Mask illegal moves
            logits = logits + (lm_batch - 1) * 1e9

            # Policy loss: cross-entropy with DD best move
            policy_loss = F.cross_entropy(logits, bm_batch)

            # Ranking loss: encourage move ordering to match DD values
            # Use pairwise ranking — softmax over DD values as soft targets
            dd_probs = F.softmax(mv_batch * 10.0, dim=-1)  # sharpen DD values
            dd_probs = dd_probs * lm_batch  # zero out illegal
            dd_probs = dd_probs / (dd_probs.sum(dim=-1, keepdim=True) + 1e-8)
            log_probs = F.log_softmax(logits, dim=-1)
            rank_loss = F.kl_div(log_probs, dd_probs, reduction="batchmean")

            # Value loss: MSE on DD value
            if oracle_s is not None:
                o_batch = oracle_s[start:end]
                critic_features = net.critic_backbone(o_batch)
                critic_features = net.critic_res_blocks(critic_features)
            else:
                critic_features = features

            value_pred = net.critic(critic_features).squeeze(-1)
            value_loss = F.mse_loss(value_pred, v_batch)

            # Total loss
            loss = policy_loss + 0.5 * value_loss + 0.5 * rank_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_rank_loss += rank_loss.item()
            num_batches += 1

        avg_ploss = epoch_policy_loss / num_batches
        avg_vloss = epoch_value_loss / num_batches
        avg_rloss = epoch_rank_loss / num_batches
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"policy={avg_ploss:.4f} value={avg_vloss:.4f} rank={avg_rloss:.4f}"
        )

    # Save checkpoint
    if save_path is None:
        save_path = "checkpoints/dd_pretrain.pt"

    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"\nSaved DD-pretrained model to {save_path}")


def run_arena_duplicate(argv: list[str]) -> None:
    """Head-to-head duplicate match between two checkpoints.

    Usage::

        python -m tarok arena-duplicate \
            --challenger <ckpt_A.ts> \
            --defender <ckpt_B.ts> \
            --boards 1000 \
            --seed 42 \
            [--explore-rate 0.0] [--concurrency 1] \
            [--pairing rotation_8game] [--bootstrap 1000] \
            [--output out.json]

    Reuses ``SeededSelfPlayAdapter`` + ``RotationPairingAdapter`` +
    ``NumpyDuplicateArenaStats`` — the same ports that drive duplicate-RL
    training. One engine path, two call sites (see ``docs/double_rl.md``
    §8).
    """
    import argparse
    import json

    from training.adapters.duplicate.numpy_arena_stats import (
        NumpyDuplicateArenaStats,
    )
    from training.adapters.duplicate.rotation_pairing import (
        RotationPairingAdapter,
    )
    from training.adapters.duplicate.seeded_self_play_adapter import (
        SeededSelfPlayAdapter,
    )
    from training.adapters.self_play import RustSelfPlay
    from training.use_cases.run_duplicate_arena import RunDuplicateArena

    parser = argparse.ArgumentParser(
        prog="python -m tarok arena-duplicate",
        description="Head-to-head duplicate match between two TorchScript checkpoints.",
    )
    parser.add_argument("--challenger", required=True, help="Path to challenger .ts")
    parser.add_argument("--defender", required=True, help="Path to defender .ts")
    parser.add_argument(
        "--boards", type=int, default=1000, help="Target paired games (default 1000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Pairing + bootstrap RNG seed")
    parser.add_argument(
        "--explore-rate",
        type=float,
        default=0.0,
        help="Policy exploration rate (default 0 = greedy)",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Rust self-play concurrency")
    parser.add_argument(
        "--pairing",
        choices=["rotation_8game", "rotation_4game", "single_seat_2game"],
        default="rotation_8game",
        help="Pod rotation scheme (default rotation_8game)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap CI resamples (default 1000; 0 to skip)",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output file")
    args = parser.parse_args(argv)

    selfplay = SeededSelfPlayAdapter(inner=RustSelfPlay())
    pairing = RotationPairingAdapter(pairing=args.pairing)
    stats = NumpyDuplicateArenaStats()

    use_case = RunDuplicateArena(selfplay=selfplay, pairing=pairing, stats=stats)

    print(
        f"Duplicate arena: challenger={args.challenger} defender={args.defender} "
        f"boards={args.boards} seed={args.seed} pairing={args.pairing}"
    )
    result = use_case.execute(
        challenger_path=args.challenger,
        defender_path=args.defender,
        n_boards=args.boards,
        rng_seed=args.seed,
        explore_rate=args.explore_rate,
        concurrency=args.concurrency,
        bootstrap_samples=args.bootstrap,
    )

    summary = {
        "challenger": args.challenger,
        "defender": args.defender,
        "pairing": args.pairing,
        "seed": args.seed,
        "boards_played": result.boards_played,
        "challenger_mean_score": result.challenger_mean_score,
        "defender_mean_score": result.defender_mean_score,
        "mean_duplicate_advantage": result.mean_duplicate_advantage,
        "duplicate_advantage_std": result.duplicate_advantage_std,
        "ci_low_95": result.ci_low_95,
        "ci_high_95": result.ci_high_95,
        "imps_per_board": result.imps_per_board,
    }

    print("")
    print(f"  Boards played:          {result.boards_played}")
    print(f"  Challenger mean score:  {result.challenger_mean_score:+.3f}")
    print(f"  Defender mean score:    {result.defender_mean_score:+.3f}")
    print(f"  Mean duplicate advantage: {result.mean_duplicate_advantage:+.3f}")
    print(f"  Advantage std:          {result.duplicate_advantage_std:.3f}")
    print(f"  95% CI:                 [{result.ci_low_95:+.3f}, {result.ci_high_95:+.3f}]")
    print(f"  IMPs / board:           {result.imps_per_board:+.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
