"""Adapter: PPO training — self-contained, no dependency on deleted training_lab.py.

Owns the optimizer, compute backend, and the PPO update loop.
Extracted from the deleted PPOTrainer._ppo_update with zero behavioral changes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from tarok.adapters.ai.agent import Experience
from tarok.adapters.ai.compute import create_backend
from tarok.adapters.ai.encoding import (
    ANNOUNCE_ACTION_SIZE,
    BID_ACTION_SIZE,
    CARD_ACTION_SIZE,
    DecisionType,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
)
from tarok.adapters.ai.network import TarokNet

from training.entities import TrainingConfig
from training.ports import PPOPort

log = logging.getLogger(__name__)

_DT_MAP = {
    0: DecisionType.BID,
    1: DecisionType.KING_CALL,
    2: DecisionType.TALON_PICK,
    3: DecisionType.CARD_PLAY,
}

_ACTION_SIZES = {
    DecisionType.BID: BID_ACTION_SIZE,
    DecisionType.KING_CALL: KING_ACTION_SIZE,
    DecisionType.TALON_PICK: TALON_ACTION_SIZE,
    DecisionType.CARD_PLAY: CARD_ACTION_SIZE,
    DecisionType.ANNOUNCE: ANNOUNCE_ACTION_SIZE,
}


class PPOAdapter(PPOPort):
    def __init__(self) -> None:
        self._network: TarokNet | None = None
        self._optimizer: optim.Adam | None = None
        self._compute: Any = None
        self._config: TrainingConfig | None = None
        # PPO hyperparams (standard defaults)
        self._gamma = 0.99
        self._gae_lambda = 0.95
        self._clip_epsilon = 0.2
        self._value_coef = 0.5
        self._entropy_coef = 0.01

    def setup(self, weights: dict, config: TrainingConfig, device: str) -> None:
        self._config = config
        self._compute = create_backend(device)

        hidden_size = weights["shared.0.weight"].shape[0]
        oracle = any(k.startswith("critic_backbone") for k in weights)

        self._network = TarokNet(hidden_size=hidden_size, oracle_critic=oracle)
        self._network.load_state_dict(weights)
        self._network = self._compute.prepare_network(self._network)
        self._optimizer = optim.Adam(self._network.parameters(), lr=config.lr)

    def set_lr(self, lr: float) -> None:
        assert self._optimizer is not None
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def update(self, raw_experiences: dict[str, Any]) -> tuple[dict[str, float], dict]:
        assert self._network is not None
        assert self._optimizer is not None
        assert self._config is not None

        all_exps = _raw_to_experiences(raw_experiences)
        metrics = self._ppo_update(all_exps)

        new_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
        return metrics, new_weights

    def _ppo_update(self, experiences: list[Experience]) -> dict[str, float]:
        if not experiences:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        network = self._network
        compute = self._compute
        optimizer = self._optimizer

        # --- 1. Compute GAE per-game in temporal order ---
        games: dict[int, list[Experience]] = defaultdict(list)
        for exp in experiences:
            games[exp.game_id].append(exp)

        gae_advantages: dict[int, float] = {}
        gae_returns: dict[int, float] = {}

        for game_exps in games.values():
            game_exps.sort(key=lambda e: e.step_in_game)
            n = len(game_exps)
            advantages = [0.0] * n
            returns = [0.0] * n
            last_gae = 0.0

            for t in reversed(range(n)):
                exp = game_exps[t]
                next_value = 0.0 if t == n - 1 else game_exps[t + 1].value.item()
                delta = exp.reward + self._gamma * next_value - exp.value.item()
                last_gae = delta + self._gamma * self._gae_lambda * last_gae
                advantages[t] = last_gae
                returns[t] = last_gae + exp.value.item()

            for t, exp in enumerate(game_exps):
                gae_advantages[id(exp)] = advantages[t]
                gae_returns[id(exp)] = returns[t]

        # --- 2. Group by decision type for correct head routing ---
        grouped: dict[DecisionType, list[Experience]] = defaultdict(list)
        for exp in experiences:
            grouped[exp.decision_type].append(exp)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for dt, dt_exps in grouped.items():
            if not dt_exps:
                continue

            action_size = _ACTION_SIZES[dt]
            states = compute.stack_to_device([e.state for e in dt_exps])
            actions = compute.tensor_to_device([e.action for e in dt_exps], dtype=torch.long)
            old_log_probs = compute.stack_to_device([e.log_prob for e in dt_exps])

            has_oracle = dt_exps[0].oracle_state is not None
            oracle_states = (
                compute.stack_to_device([e.oracle_state for e in dt_exps])
                if has_oracle else None
            )

            advantages = compute.tensor_to_device(
                [gae_advantages[id(e)] for e in dt_exps], dtype=torch.float32,
            )
            returns = compute.tensor_to_device(
                [gae_returns[id(e)] for e in dt_exps], dtype=torch.float32,
            )

            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            if dt_exps[0].legal_mask is not None:
                legal_masks = compute.stack_to_device([e.legal_mask for e in dt_exps])
            else:
                legal_masks = torch.ones(len(dt_exps), action_size, dtype=torch.float32)
                legal_masks = compute.to_device(legal_masks)

            old_values = compute.stack_to_device([e.value for e in dt_exps])

            for _ in range(self._config.ppo_epochs):
                indices = torch.randperm(len(dt_exps))

                for start in range(0, len(dt_exps), self._config.batch_size):
                    end = min(start + self._config.batch_size, len(dt_exps))
                    batch_idx = indices[start:end]

                    b_states = states[batch_idx]
                    b_actions = actions[batch_idx]
                    b_old_log_probs = old_log_probs[batch_idx]
                    b_advantages = advantages[batch_idx]
                    b_returns = returns[batch_idx]
                    b_masks = legal_masks[batch_idx]
                    b_oracle = oracle_states[batch_idx] if oracle_states is not None else None

                    new_log_probs, new_values, entropy = network.evaluate_action(
                        b_states, b_actions, b_masks, dt,
                        oracle_state=b_oracle,
                    )

                    ratio = torch.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(
                        ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon,
                    ) * b_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.functional.mse_loss(new_values, b_returns)

                    loss = (
                        policy_loss
                        + self._value_coef * value_loss
                        - self._entropy_coef * entropy.mean()
                    )

                    if not torch.isfinite(loss):
                        log.warning("Non-finite loss (%.4g), skipping batch", loss.item())
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                    optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates += 1

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "total_loss": (total_policy_loss + total_value_loss) / n,
        }


def _raw_to_experiences(raw: dict) -> list[Experience]:
    exps_by_game: dict[int, list[Experience]] = {}

    for i in range(len(raw["actions"])):
        if int(raw["players"][i]) != 0:
            continue
        gid = int(raw["game_ids"][i])
        scores = raw["scores"]
        reward = float(scores[gid % scores.shape[0], 0]) / 100.0
        exps_by_game.setdefault(gid, []).append(
            Experience(
                state=torch.tensor(raw["states"][i], dtype=torch.float32),
                action=torch.tensor(int(raw["actions"][i]), dtype=torch.long),
                log_prob=torch.tensor(float(raw["log_probs"][i]), dtype=torch.float32),
                value=torch.tensor(float(raw["values"][i]), dtype=torch.float32),
                reward=reward,
                decision_type=_DT_MAP[int(raw["decision_types"][i])],
                legal_mask=torch.tensor(raw["legal_masks"][i], dtype=torch.float32),
                game_id=gid,
            )
        )

    all_exps: list[Experience] = []
    for game_exps in exps_by_game.values():
        game_exps.sort(key=lambda e: e.step_in_game)
        for j, exp in enumerate(game_exps):
            exp.done = (j == len(game_exps) - 1)
        all_exps.extend(game_exps)

    return all_exps
