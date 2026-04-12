"""Warmup pre-training — teach the value network from random play.

The Rust engine plays hundreds of thousands of random games, recording
(state, reward) at every decision point.  We train the critic head to
predict game outcomes from mid-game states.

The policy network is NOT pre-trained — random play has no useful policy
signal.  PPO self-play shapes the policy once the value head is warm.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

import tarok_engine as te

from training_lab.entities.encoding import DecisionType, RUST_DT_MAP
from training_lab.entities.network import TarokNet
from training_lab.ports.compute_backend import ComputeBackendPort

log = logging.getLogger(__name__)


class RunWarmup:
    """Pre-train the value (critic) head from random game experiences."""

    def __init__(
        self,
        compute: ComputeBackendPort,
        network: TarokNet | None = None,
        hidden_size: int = 256,
        oracle_critic: bool = False,
    ):
        self.compute = compute
        if network is not None:
            self.network = network
        else:
            self.network = TarokNet(hidden_size=hidden_size, oracle_critic=oracle_critic)
        self.network = self.compute.prepare_network(self.network)

    def run(
        self,
        num_games: int = 500_000,
        batch_size: int = 2048,
        epochs: int = 3,
        lr: float = 1e-3,
        chunk_size: int = 10_000,
        include_oracle: bool = False,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        """Run warmup pre-training. Returns stats dict."""
        dev = self.compute.device
        self.network.to(dev)

        # Only optimize critic + shared backbone (not actor heads)
        critic_params = list(self.network.critic.parameters())
        if include_oracle and self.network.oracle_critic_enabled:
            critic_params += list(self.network.critic_backbone.parameters())
        critic_params += list(self.network.shared.parameters())

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

            gen_t0 = time.perf_counter()
            data = te.generate_warmup_data(n, include_oracle=include_oracle)
            gen_elapsed = time.perf_counter() - gen_t0

            n_exp = data["num_experiences"]
            states = torch.from_numpy(data["states"]).float().to(dev)
            rewards = torch.from_numpy(data["rewards"]).float().to(dev)

            oracle_states = None
            if include_oracle and data["oracle_states"] is not None:
                oracle_states = torch.from_numpy(data["oracle_states"]).float().to(dev)

            chunk_loss = 0.0
            chunk_updates = 0

            for _epoch in range(epochs):
                perm = torch.randperm(n_exp)
                for start in range(0, n_exp, batch_size):
                    end = min(start + batch_size, n_exp)
                    idx = perm[start:end]

                    b_states = states[idx]
                    b_rewards = rewards[idx]
                    b_oracle = oracle_states[idx] if oracle_states is not None else None

                    features = self.network.shared(b_states)

                    if (
                        include_oracle
                        and self.network.oracle_critic_enabled
                        and b_oracle is not None
                    ):
                        critic_features = self.network.critic_backbone(b_oracle)
                    else:
                        critic_features = features

                    values = self.network.critic(critic_features).squeeze(-1)
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
