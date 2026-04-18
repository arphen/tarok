"""Torch-based PPO adapter implementation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tarok_model.compute import create_backend
from tarok_model.encoding import (
    ANNOUNCE_ACTION_SIZE,
    BID_ACTION_SIZE,
    CARD_ACTION_SIZE,
    DecisionType,
    GameMode,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
)
from tarok_model.network import TarokNetV4

from training.adapters.ppo.batching import prepare_batched, release_allocator_memory
from training.adapters.ppo.contracts import validate_v4_contract_indices_with_rust
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
        self._network: TarokNetV4 | None = None
        self._optimizer: optim.Adam | None = None
        self._compute: Any = None
        self._config: TrainingConfig | None = None
        # PPO hyperparams — assigned from validated config in setup().
        self._gamma: float = 0.99
        self._gae_lambda: float = 0.95
        self._clip_epsilon: float = 0.2
        self._value_coef: float = 0.5
        self._entropy_coef: float = 0.01
        self._imitation_coef: float = 0.0

    def setup(self, weights: dict, config: TrainingConfig, device: str) -> None:
        self._config = config
        self._compute = create_backend(device)
        self._imitation_coef = float(config.imitation_coef)
        self._gamma = float(config.gamma)
        self._gae_lambda = float(config.gae_lambda)
        self._clip_epsilon = float(config.clip_epsilon)
        self._value_coef = float(config.value_coef)
        self._entropy_coef = float(config.entropy_coef)

        hidden_size = weights["shared.0.weight"].shape[0]
        oracle = any(k.startswith("critic_backbone") for k in weights)

        if config.model_arch != "v4":
            raise ValueError(f"Unsupported model_arch={config.model_arch}. Only 'v4' is supported.")

        self._network = TarokNetV4(hidden_size=hidden_size, oracle_critic=oracle)
        validate_v4_contract_indices_with_rust()
        self._network.load_state_dict(weights)
        self._network = self._compute.prepare_network(self._network)
        self._optimizer = optim.Adam(self._network.parameters(), lr=config.lr)

    def set_entropy_coef(self, coef: float) -> None:
        self._entropy_coef = float(coef)

    def set_lr(self, lr: float) -> None:
        assert self._optimizer is not None
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def set_imitation_coef(self, coef: float) -> None:
        self._imitation_coef = float(coef)

    def update(self, raw_experiences: dict[str, Any], nn_seats: list[int]) -> tuple[dict[str, float], dict]:
        assert self._network is not None
        assert self._optimizer is not None
        assert self._config is not None

        # Fail fast if the self-play payload contains non-learner seats.
        # Rust run_self_play is expected to emit learner-only experiences.
        players_np = np.asarray(raw_experiences["players"])
        if players_np.size > 0 and not np.isin(players_np, nn_seats).all():
            unique_players = sorted(set(int(x) for x in players_np.tolist()))
            raise RuntimeError(
                "run_self_play returned non-learner experiences "
                f"(players={unique_players}, learner_seats={sorted(nn_seats)}). "
                "Rebuild engine-rs so learner-only emission is active."
            )
        del nn_seats

        prepped = prepare_batched(raw_experiences, gamma=self._gamma, gae_lambda=self._gae_lambda)
        metrics = self._ppo_update_batched(**prepped)

        new_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
        del prepped
        release_allocator_memory()
        return metrics, new_weights

    def _ppo_update_batched(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        vad: torch.Tensor,
        decision_types: np.ndarray,
        legal_masks: torch.Tensor,
        oracle_states: torch.Tensor | None,
        game_modes: np.ndarray,
    ) -> dict[str, float]:
        old_log_probs = log_probs

        if len(states) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        network = self._network
        compute = self._compute
        optimizer = self._optimizer

        groups: dict[tuple[DecisionType, GameMode | None], np.ndarray] = {}

        for dt_int, dt_enum in _DT_MAP.items():
            dt_mask = decision_types == dt_int
            if not np.any(dt_mask):
                continue
            if dt_enum == DecisionType.CARD_PLAY:
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
            network.oracle_critic_enabled
            and oracle_states is not None
            and self._imitation_coef > 0.0
        )

        for (dt, gm), idx in groups.items():
            n = len(idx)
            action_size = _ACTION_SIZES[dt]
            g_states = compute.to_device(states[idx])
            g_actions = compute.to_device(actions[idx])
            g_old_log_probs = compute.to_device(old_log_probs[idx])
            g_vad = compute.to_device(vad[idx])  # (n, 3): old_values | advantages | returns
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
                    b_vad = g_vad[batch_idx]  # (batch, 3)
                    b_old_values = b_vad[:, 0]
                    b_advantages = b_vad[:, 1]
                    b_returns = b_vad[:, 2]
                    b_masks = g_masks[batch_idx]
                    b_oracle_states = g_oracle_states[batch_idx] if g_oracle_states is not None else None

                    new_log_probs, new_values, entropy = network.evaluate_action(
                        b_states,
                        b_actions,
                        b_masks,
                        dt,
                        oracle_state=b_oracle_states,
                        game_mode=gm,
                    )

                    ratio = torch.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1 - self._clip_epsilon,
                        1 + self._clip_epsilon,
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
                        + self._imitation_coef * il_loss
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
