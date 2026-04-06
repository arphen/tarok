"""Warmup pre-training — teach the value network from random play.

The Rust engine plays hundreds of thousands of random games, recording
(state, reward) at every decision point.  We then train the critic head
to predict game outcomes from mid-game states.  This gives the value
function a meaningful head start before self-play:

  • The agent learns which game states tend to be good or bad
  • Value predictions stabilize faster, so early PPO updates are cleaner
  • Card-play patterns (suit following, trumping) emerge from the data

The policy network is NOT pre-trained — random play has no useful policy
signal.  PPO self-play shapes the policy once the value head is warm.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import tarok_engine as te
except ImportError:
    te = None  # type: ignore[assignment]

from tarok.adapters.ai.encoding import DecisionType, STATE_SIZE, ORACLE_STATE_SIZE
from tarok.adapters.ai.network import TarokNet


# Map Rust decision type constants to Python DecisionType enum
_DT_MAP = {
    0: DecisionType.BID,
    1: DecisionType.KING_CALL,
    2: DecisionType.TALON_PICK,
    3: DecisionType.CARD_PLAY,
    4: DecisionType.ANNOUNCE,
}


def warmup_value_network(
    network: TarokNet,
    num_games: int = 500_000,
    batch_size: int = 2048,
    epochs: int = 3,
    lr: float = 1e-3,
    chunk_size: int = 10_000,
    device: str = "cpu",
    include_oracle: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Pre-train the value (critic) head from random game experiences.

    Args:
        network: TarokNet to train (only critic weights are updated).
        num_games: Total random games to generate.
        batch_size: Mini-batch size for SGD.
        epochs: Passes over each chunk of data.
        lr: Learning rate for the critic.
        chunk_size: Games per Rust generation batch (controls peak memory).
        device: "cpu" or "cuda".
        include_oracle: Also train the oracle critic backbone.
        progress_callback: Called with progress dict after each chunk.

    Returns:
        Dict with training stats: total_experiences, avg_loss, elapsed_secs.
    """
    assert te is not None, "tarok_engine Rust extension not installed"

    dev = torch.device(device)
    network = network.to(dev)

    # Only optimize critic parameters (not actor heads)
    critic_params = list(network.critic.parameters())
    if include_oracle and network.oracle_critic_enabled:
        critic_params += list(network.critic_backbone.parameters())
    # Also include shared backbone since critic uses it for non-oracle path
    critic_params += list(network.shared.parameters())

    optimizer = optim.Adam(critic_params, lr=lr)
    mse = nn.MSELoss()

    total_exps = 0
    total_loss = 0.0
    num_chunks = 0
    t0 = time.time()

    games_remaining = num_games
    while games_remaining > 0:
        n = min(chunk_size, games_remaining)
        games_remaining -= n

        # Generate experiences in Rust (fast!)
        gen_t0 = time.perf_counter()
        data = te.generate_warmup_data(n, include_oracle=include_oracle)
        gen_elapsed = time.perf_counter() - gen_t0

        n_exp = data["num_experiences"]
        states = torch.from_numpy(data["states"]).float().to(dev)
        rewards = torch.from_numpy(data["rewards"]).float().to(dev)
        dt_arr = data["decision_types"]  # numpy uint8

        oracle_states = None
        if include_oracle and data["oracle_states"] is not None:
            oracle_states = torch.from_numpy(data["oracle_states"]).float().to(dev)

        # Pre-compute decision type tensor
        dt_tensor = torch.from_numpy(dt_arr).long()

        chunk_loss = 0.0
        chunk_updates = 0

        for _epoch in range(epochs):
            perm = torch.randperm(n_exp)
            for start in range(0, n_exp, batch_size):
                end = min(start + batch_size, n_exp)
                idx = perm[start:end]

                b_states = states[idx]
                b_rewards = rewards[idx]
                b_dt = dt_tensor[idx]
                b_oracle = oracle_states[idx] if oracle_states is not None else None

                # Forward through shared backbone → critic head
                # Group by decision type for correct routing
                # For value training, we just need the value output
                # We can use the shared backbone directly since all decision
                # types share the same state encoding

                # Use the shared backbone
                features = network.shared(b_states)

                # Critic output
                if (
                    include_oracle
                    and network.oracle_critic_enabled
                    and b_oracle is not None
                ):
                    critic_features = network.critic_backbone(b_oracle)
                else:
                    critic_features = features

                values = network.critic(critic_features).squeeze(-1)
                loss = mse(values, b_rewards)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic_params, 1.0)
                optimizer.step()

                chunk_loss += loss.item()
                chunk_updates += 1

        total_exps += n_exp
        avg_chunk_loss = chunk_loss / max(chunk_updates, 1)
        total_loss += avg_chunk_loss
        num_chunks += 1

        if progress_callback:
            progress_callback({
                "chunk": num_chunks,
                "games_done": num_games - games_remaining,
                "total_games": num_games,
                "experiences": total_exps,
                "chunk_loss": round(avg_chunk_loss, 6),
                "gen_speed": round(n / gen_elapsed) if gen_elapsed > 0 else 0,
                "elapsed": round(time.time() - t0, 1),
            })

    elapsed = time.time() - t0
    return {
        "total_experiences": total_exps,
        "avg_loss": round(total_loss / max(num_chunks, 1), 6),
        "elapsed_secs": round(elapsed, 1),
        "games_per_sec": round(num_games / elapsed) if elapsed > 0 else 0,
    }
