"""Adapter: PPO training — self-contained, no dependency on deleted training_lab.py.

Owns the optimizer, compute backend, and the PPO update loop.
Extracted from the deleted PPOTrainer._ppo_update with zero behavioral changes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tarok.core.compute import create_backend
from tarok.core.encoding import (
    ANNOUNCE_ACTION_SIZE,
    BID_ACTION_SIZE,
    CARD_ACTION_SIZE,
    DecisionType,
    GameMode,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
)
from tarok.core.network import TarokNet, TarokNetV3

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

_MODE_ID_TO_ENUM = {
    0: GameMode.SOLO,
    1: GameMode.KLOP_BERAC,
    2: GameMode.PARTNER_PLAY,
    3: GameMode.COLOR_VALAT,
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

        model_arch = config.model_arch
        if any(k.startswith("card_heads.") for k in weights):
            model_arch = "v3"
        model_cls = TarokNetV3 if model_arch == "v3" else TarokNet
        self._network = model_cls(hidden_size=hidden_size, oracle_critic=oracle)
        if model_cls is TarokNetV3:
            _validate_v3_contract_indices_with_rust()
        self._network.load_state_dict(weights)
        self._network = self._compute.prepare_network(self._network)
        self._optimizer = optim.Adam(self._network.parameters(), lr=config.lr)

    def set_lr(self, lr: float) -> None:
        assert self._optimizer is not None
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def update(self, raw_experiences: dict[str, Any], nn_seats: list[int], bot_seats: list[int]) -> tuple[dict[str, float], dict]:
        assert self._network is not None
        assert self._optimizer is not None
        assert self._config is not None

        # Learn from every seat regardless of who sits there.  The seat args are
        # kept for interface compatibility with the surrounding training code.
        del nn_seats, bot_seats
        prepped = _prepare_batched(raw_experiences)
        metrics = self._ppo_update_batched(prepped)
        metrics["il_loss"] = 0.0

        new_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
        return metrics, new_weights

    def _ppo_update_batched(self, data: dict[str, Any]) -> dict[str, float]:
        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]
        decision_types = data["decision_types"]
        legal_masks = data["legal_masks"]
        game_modes = data.get("game_modes")

        if len(states) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        network = self._network
        compute = self._compute
        optimizer = self._optimizer

        is_v3 = isinstance(network, TarokNetV3)
        groups: dict[tuple[DecisionType, GameMode | None], np.ndarray] = {}

        for dt_int, dt_enum in _DT_MAP.items():
            dt_mask = decision_types == dt_int
            if not np.any(dt_mask):
                continue
            if is_v3 and dt_enum == DecisionType.CARD_PLAY and game_modes is not None:
                dt_indices = np.where(dt_mask)[0]
                gm_vals = game_modes[dt_indices]
                for gm_int in np.unique(gm_vals):
                    groups[(dt_enum, _MODE_ID_TO_ENUM[int(gm_int)])] = dt_indices[gm_vals == gm_int]
            else:
                groups[(dt_enum, None)] = np.where(dt_mask)[0]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for (dt, gm), idx in groups.items():
            n = len(idx)
            action_size = _ACTION_SIZES[dt]
            g_states = compute.to_device(states[idx])
            g_actions = compute.to_device(actions[idx])
            g_old_log_probs = compute.to_device(old_log_probs[idx])
            g_advantages = advantages[idx].clone()
            if g_advantages.numel() > 1:
                g_advantages = (g_advantages - g_advantages.mean()) / (g_advantages.std() + 1e-8)
            g_advantages = compute.to_device(g_advantages)
            g_returns = compute.to_device(returns[idx])
            g_masks = compute.to_device(legal_masks[idx, :action_size])

            for _ in range(self._config.ppo_epochs):
                indices = torch.randperm(n, device=g_states.device)

                for start in range(0, n, self._config.batch_size):
                    end = min(start + self._config.batch_size, n)
                    batch_idx = indices[start:end]

                    b_states = g_states[batch_idx]
                    b_actions = g_actions[batch_idx]
                    b_old_log_probs = g_old_log_probs[batch_idx]
                    b_advantages = g_advantages[batch_idx]
                    b_returns = g_returns[batch_idx]
                    b_masks = g_masks[batch_idx]

                    new_log_probs, new_values, entropy = network.evaluate_action(
                        b_states, b_actions, b_masks, dt,
                        oracle_state=None,
                        game_mode=gm,
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

def _prepare_batched(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert raw Rust arrays into batched tensors and vectorized GAE inputs."""

    states_np = np.asarray(raw["states"])
    actions_np = np.asarray(raw["actions"])
    log_probs_np = np.asarray(raw["log_probs"])
    values_np = np.asarray(raw["values"])
    decision_types_np = np.asarray(raw["decision_types"])
    game_ids_np = np.asarray(raw["game_ids"])
    players_np = np.asarray(raw["players"])
    scores_np = np.asarray(raw["scores"])
    legal_masks_np = np.asarray(raw["legal_masks"])

    # Precompute v3 card-head modes from the contract one-hot in state.
    # This avoids expensive per-batch mode inference in TarokNetV3.
    game_modes_np: np.ndarray | None
    if states_np.shape[1] >= 230:
        contract_slice = states_np[:, 220:230]
        max_idx = np.argmax(contract_slice, axis=1)
        max_vals = np.max(contract_slice, axis=1)
        game_modes_np = np.full(len(states_np), 2, dtype=np.int8)  # partner_play default
        game_modes_np[(max_idx >= 4) & (max_idx <= 7)] = 0          # solo
        game_modes_np[(max_idx == 0) | (max_idx == 8)] = 1          # klop / berac
        game_modes_np[max_idx == 9] = 3                             # color valat
        game_modes_np[max_vals <= 0.0] = 2
    else:
        game_modes_np = None

    n_total = len(actions_np)
    gids = game_ids_np % scores_np.shape[0]
    rewards_np = scores_np[gids, players_np].astype(np.float32) / 100.0

    traj_keys = game_ids_np.astype(np.int64) * 4 + players_np.astype(np.int64)
    sort_idx = np.lexsort((np.arange(n_total), traj_keys))

    sorted_keys = traj_keys[sort_idx]
    sorted_values = values_np[sort_idx]
    sorted_rewards = rewards_np[sort_idx]

    boundaries = np.where(np.diff(sorted_keys) != 0)[0] + 1
    traj_starts = np.concatenate([[0], boundaries])
    traj_ends = np.concatenate([boundaries, [n_total]])

    gamma = 0.99
    gae_lambda = 0.95
    advantages_sorted = np.zeros(n_total, dtype=np.float32)
    returns_sorted = np.zeros(n_total, dtype=np.float32)

    for start, end in zip(traj_starts, traj_ends):
        traj_len = end - start
        last_gae = 0.0
        for offset in reversed(range(traj_len)):
            idx = start + offset
            next_val = 0.0 if offset == traj_len - 1 else sorted_values[idx + 1]
            delta = sorted_rewards[idx] + gamma * next_val - sorted_values[idx]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages_sorted[idx] = last_gae
            returns_sorted[idx] = last_gae + sorted_values[idx]

    advantages_np = np.empty(n_total, dtype=np.float32)
    returns_np = np.empty(n_total, dtype=np.float32)
    advantages_np[sort_idx] = advantages_sorted
    returns_np[sort_idx] = returns_sorted

    return {
        "states": torch.from_numpy(states_np),
        "actions": torch.from_numpy(actions_np.astype(np.int64)),
        "log_probs": torch.from_numpy(log_probs_np),
        "advantages": torch.from_numpy(advantages_np),
        "returns": torch.from_numpy(returns_np),
        "decision_types": decision_types_np,
        "legal_masks": torch.from_numpy(legal_masks_np),
        "game_modes": game_modes_np,
    }


def _validate_v3_contract_indices_with_rust() -> None:
    """Fail fast if Python and Rust contract indices diverge."""
    try:
        import tarok_engine as te
    except Exception:
        return

    expected = {
        "CONTRACT_KLOP": TarokNetV3._KLOP_IDX,
        "CONTRACT_THREE": TarokNetV3._THREE_IDX,
        "CONTRACT_TWO": TarokNetV3._TWO_IDX,
        "CONTRACT_ONE": TarokNetV3._ONE_IDX,
        "CONTRACT_SOLO_THREE": TarokNetV3._SOLO_THREE_IDX,
        "CONTRACT_SOLO_TWO": TarokNetV3._SOLO_TWO_IDX,
        "CONTRACT_SOLO_ONE": TarokNetV3._SOLO_ONE_IDX,
        "CONTRACT_SOLO": TarokNetV3._SOLO_IDX,
        "CONTRACT_BERAC": TarokNetV3._BERAC_IDX,
        "CONTRACT_BARVNI_VALAT": TarokNetV3._BARVNI_VALAT_IDX,
    }

    mismatches: list[str] = []
    for name, py_value in expected.items():
        rust_value = getattr(te, name, None)
        if rust_value is None or int(rust_value) != int(py_value):
            mismatches.append(f"{name}: rust={rust_value} python={py_value}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise RuntimeError(
            "Rust/Python contract index mismatch detected. "
            f"Refusing to train with ambiguous v3 mode routing: {mismatch_text}"
        )
