"""Adapter: PPO training — self-contained, no dependency on deleted training_lab.py.

Owns the optimizer, compute backend, and the PPO update loop.
Extracted from the deleted PPOTrainer._ppo_update with zero behavioral changes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tarok_engine as te
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

        new_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
        return metrics, new_weights

    def _ppo_update_batched(self, data: dict[str, Any]) -> dict[str, float]:
        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        old_values = data["values"]
        advantages = data["advantages"]
        returns = data["returns"]
        decision_types = data["decision_types"]
        legal_masks = data["legal_masks"]
        oracle_states = data.get("oracle_states")
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
        total_il_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_updates = 0

        use_oracle_distill = (
            getattr(network, "oracle_critic_enabled", False)
            and oracle_states is not None
            and float(getattr(self._config, "imitation_coef", 0.0)) > 0.0
        )

        for (dt, gm), idx in groups.items():
            n = len(idx)
            action_size = _ACTION_SIZES[dt]
            g_states = compute.to_device(states[idx])
            g_actions = compute.to_device(actions[idx])
            g_old_log_probs = compute.to_device(old_log_probs[idx])
            g_old_values = compute.to_device(old_values[idx])
            g_advantages = compute.to_device(advantages[idx])
            g_returns = compute.to_device(returns[idx])
            g_masks = compute.to_device(legal_masks[idx, :action_size])
            g_oracle_states = compute.to_device(oracle_states[idx]) if oracle_states is not None else None

            for _ in range(self._config.ppo_epochs):
                indices = torch.randperm(n, device=g_states.device)

                for start in range(0, n, self._config.batch_size):
                    end = min(start + self._config.batch_size, n)
                    batch_idx = indices[start:end]

                    b_states = g_states[batch_idx]
                    b_actions = g_actions[batch_idx]
                    b_old_log_probs = g_old_log_probs[batch_idx]
                    b_old_values = g_old_values[batch_idx]
                    b_advantages = g_advantages[batch_idx]
                    b_returns = g_returns[batch_idx]
                    b_masks = g_masks[batch_idx]
                    b_oracle_states = g_oracle_states[batch_idx] if g_oracle_states is not None else None

                    new_log_probs, new_values, entropy = network.evaluate_action(
                        b_states, b_actions, b_masks, dt,
                        oracle_state=b_oracle_states,
                        game_mode=gm,
                    )

                    ratio = torch.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(
                        ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon,
                    ) * b_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    v_pred_clipped = b_old_values + torch.clamp(
                        new_values - b_old_values,
                        -self._clip_epsilon,
                        self._clip_epsilon,
                    )
                    v_loss_unclipped = nn.functional.mse_loss(new_values, b_returns, reduction="none")
                    v_loss_clipped = nn.functional.mse_loss(v_pred_clipped, b_returns, reduction="none")
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    il_loss = torch.zeros((), device=b_states.device, dtype=b_states.dtype)
                    if use_oracle_distill and b_oracle_states is not None:
                        actor_features = network.get_actor_features(b_states)
                        with torch.no_grad():
                            critic_target = network.get_critic_features(b_oracle_states)
                        actor_features = nn.functional.normalize(actor_features, p=2, dim=-1, eps=1e-8)
                        critic_target = nn.functional.normalize(critic_target, p=2, dim=-1, eps=1e-8)
                        il_loss = 1.0 - nn.functional.cosine_similarity(
                            actor_features,
                            critic_target,
                            dim=-1,
                            eps=1e-8,
                        ).mean()

                    loss = (
                        policy_loss
                        + self._value_coef * value_loss
                        + self._config.imitation_coef * il_loss
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
                    total_il_loss += il_loss.item()
                    total_entropy += entropy.mean().item()
                    total_loss += loss.item()
                    num_updates += 1

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "il_loss": total_il_loss / n,
            "entropy": total_entropy / n,
            "total_loss": total_loss / n,
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
    oracle_states_raw = raw.get("oracle_states")
    oracle_states_np = np.asarray(oracle_states_raw, dtype=np.float32) if oracle_states_raw is not None else None

    # Precompute v3 card-head modes from the contract one-hot in state.
    # This avoids expensive per-batch mode inference in TarokNetV3.
    game_modes_np: np.ndarray | None
    contract_offset = int(te.CONTRACT_OFFSET)
    contract_size = int(te.CONTRACT_SIZE)
    contract_end = contract_offset + contract_size
    if states_np.shape[1] >= contract_end:
        contract_slice = states_np[:, contract_offset:contract_end]
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

    sorted_keys = np.asarray(traj_keys[sort_idx], dtype=np.int64)
    sorted_values = np.asarray(values_np[sort_idx], dtype=np.float32)
    sorted_rewards = np.asarray(rewards_np[sort_idx], dtype=np.float32)

    advantages_sorted, returns_sorted = te.compute_gae(
        sorted_values,
        sorted_rewards,
        sorted_keys,
        gamma=0.99,
        gae_lambda=0.95,
    )
    advantages_sorted = np.asarray(advantages_sorted, dtype=np.float32)
    returns_sorted = np.asarray(returns_sorted, dtype=np.float32)

    advantages_np = np.empty(n_total, dtype=np.float32)
    returns_np = np.empty(n_total, dtype=np.float32)
    advantages_np[sort_idx] = advantages_sorted
    returns_np[sort_idx] = returns_sorted

    # Normalize advantages once globally to avoid over-scaling rare
    # decision-type / game-mode subgroups during PPO updates.
    if advantages_np.size > 1:
        adv_mean = float(advantages_np.mean())
        adv_std = float(advantages_np.std())
        advantages_np = (advantages_np - adv_mean) / (adv_std + 1e-8)

    return {
        "states": torch.from_numpy(states_np),
        "actions": torch.from_numpy(actions_np.astype(np.int64)),
        "log_probs": torch.from_numpy(log_probs_np),
        "values": torch.from_numpy(values_np.astype(np.float32)),
        "advantages": torch.from_numpy(advantages_np),
        "returns": torch.from_numpy(returns_np),
        "decision_types": decision_types_np,
        "legal_masks": torch.from_numpy(legal_masks_np),
        "oracle_states": torch.from_numpy(oracle_states_np) if oracle_states_np is not None else None,
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
