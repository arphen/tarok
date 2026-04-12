"""Imitation learning — bootstrap the policy from StockŠkis expert games.

The Rust engine plays millions of games with all 4 players using StockŠkis
heuristic bots, recording (state, action, legal_mask, reward) at every
decision point.  We then train BOTH the policy heads (imitation) and the
value head (regression) simultaneously.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import tarok_engine as te

from training_lab.entities.encoding import (
    DecisionType,
    ACTION_SIZES,
    RUST_DT_MAP,
)
from training_lab.entities.network import TarokNet
from training_lab.ports.compute_backend import ComputeBackendPort

log = logging.getLogger(__name__)


class RunImitationPretraining:
    """Pre-train both policy and value heads from StockŠkis expert games."""

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
        num_games: int = 1_000_000,
        batch_size: int = 2048,
        epochs: int = 3,
        lr: float = 1e-3,
        value_coef: float = 0.5,
        chunk_size: int = 50_000,
        include_oracle: bool = False,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        """Run imitation pre-training. Returns stats dict."""
        dev = self.compute.device
        self.network.train()

        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss_fn = nn.MSELoss()

        total_exps = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_chunks = 0
        t0 = time.time()

        games_remaining = num_games
        while games_remaining > 0:
            n = min(chunk_size, games_remaining)
            games_remaining -= n

            gen_t0 = time.perf_counter()
            data = te.generate_expert_data(n, include_oracle=include_oracle)
            gen_elapsed = time.perf_counter() - gen_t0

            n_exp = data["num_experiences"]
            states = torch.from_numpy(data["states"]).float().to(dev)
            actions = torch.from_numpy(
                np.asarray(data["actions"], dtype=np.int64)
            ).long().to(dev)
            rewards = torch.from_numpy(data["rewards"]).float().to(dev)
            dt_arr = data["decision_types"]

            oracle_states = None
            if include_oracle and data["oracle_states"] is not None:
                oracle_states = torch.from_numpy(data["oracle_states"]).float().to(dev)

            # Group indices by decision type
            dt_indices: dict[DecisionType, torch.Tensor] = {}
            dt_numpy = np.asarray(dt_arr, dtype=np.uint8)
            for rust_dt, py_dt in RUST_DT_MAP.items():
                mask = dt_numpy == rust_dt
                if mask.any():
                    dt_indices[py_dt] = torch.from_numpy(np.where(mask)[0]).long().to(dev)

            chunk_policy = 0.0
            chunk_value = 0.0
            chunk_updates = 0

            for _epoch in range(epochs):
                for dt, indices in dt_indices.items():
                    if len(indices) == 0:
                        continue

                    dt_states = states[indices]
                    dt_actions = actions[indices]
                    dt_rewards = rewards[indices]
                    dt_oracle = oracle_states[indices] if oracle_states is not None else None

                    perm = torch.randperm(len(indices), device=dev)
                    for start in range(0, len(indices), batch_size):
                        end = min(start + batch_size, len(indices))
                        idx = perm[start:end]

                        b_states = dt_states[idx]
                        b_actions = dt_actions[idx]
                        b_rewards = dt_rewards[idx]
                        b_oracle = dt_oracle[idx] if dt_oracle is not None else None

                        logits, values = self.network(b_states, dt, oracle_state=b_oracle)

                        action_size = ACTION_SIZES[dt]
                        b_actions_clamped = b_actions.clamp(0, action_size - 1)
                        policy_loss = ce_loss_fn(logits[:, :action_size], b_actions_clamped)

                        value_loss = mse_loss_fn(values.squeeze(-1), b_rewards)

                        loss = policy_loss + value_coef * value_loss

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                        optimizer.step()

                        chunk_policy += policy_loss.item()
                        chunk_value += value_loss.item()
                        chunk_updates += 1

            total_exps += n_exp
            avg_policy = chunk_policy / max(chunk_updates, 1)
            avg_value = chunk_value / max(chunk_updates, 1)
            total_policy_loss += avg_policy
            total_value_loss += avg_value
            num_chunks += 1

            if progress_callback:
                progress_callback({
                    "chunk": num_chunks,
                    "games_done": num_games - games_remaining,
                    "total_games": num_games,
                    "experiences": total_exps,
                    "policy_loss": round(avg_policy, 6),
                    "value_loss": round(avg_value, 6),
                    "gen_speed": round(n / gen_elapsed) if gen_elapsed > 0 else 0,
                    "elapsed": round(time.time() - t0, 1),
                })

        elapsed = time.time() - t0
        return {
            "total_experiences": total_exps,
            "avg_policy_loss": round(total_policy_loss / max(num_chunks, 1), 6),
            "avg_value_loss": round(total_value_loss / max(num_chunks, 1), 6),
            "elapsed_secs": round(elapsed, 1),
            "games_per_sec": round(num_games / elapsed) if elapsed > 0 else 0,
        }
