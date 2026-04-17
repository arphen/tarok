"""Unit tests proving oracle distillation is active in PPO updates."""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training-lab"))
sys.path.insert(0, str(ROOT / "model" / "src"))

# The PPO adapter imports tarok_engine at module import time, but this test
# exercises only the isolated PPO update path and does not need the extension.
sys.modules.setdefault("tarok_engine", types.ModuleType("tarok_engine"))

from tarok_model.encoding import DecisionType
from tarok_model.network import ORACLE_STATE_SIZE, STATE_SIZE, TarokNetV4
from training.adapters.compute.cpu_backend import CpuBackend
from training.adapters.ppo import PPOAdapter
from training.entities.training_config import TrainingConfig


def _make_adapter(state_dict: dict[str, torch.Tensor], imitation_coef: float) -> PPOAdapter:
    hidden_size = state_dict["shared.0.weight"].shape[0]
    adapter = PPOAdapter()
    adapter._config = TrainingConfig(  # type: ignore[attr-defined]
        ppo_epochs=6,
        batch_size=4,
        lr=0.05,
        imitation_coef=imitation_coef,
        entropy_coef=0.0,
    )
    adapter._compute = CpuBackend()  # type: ignore[attr-defined]
    adapter._network = TarokNetV4(hidden_size=hidden_size, oracle_critic=True)  # type: ignore[attr-defined]
    adapter._network.load_state_dict(copy.deepcopy(state_dict))  # type: ignore[attr-defined]
    adapter._optimizer = torch.optim.Adam(adapter._network.parameters(), lr=0.05)  # type: ignore[attr-defined]
    adapter._clip_epsilon = 0.2  # type: ignore[attr-defined]
    adapter._value_coef = 0.5  # type: ignore[attr-defined]
    adapter._entropy_coef = 0.0  # type: ignore[attr-defined]
    adapter._imitation_coef = imitation_coef  # type: ignore[attr-defined]
    return adapter


def _make_neutral_batch(network: TarokNetV4, include_oracle_states: bool) -> dict:
    torch.manual_seed(7)
    batch_size = 4
    states = torch.randn(batch_size, STATE_SIZE)
    oracle_states = torch.randn(batch_size, ORACLE_STATE_SIZE)

    legal_masks = torch.zeros(batch_size, 54)
    legal_masks[:, :9] = 1.0
    actions = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    with torch.no_grad():
        old_log_probs, old_values, _entropy = network.evaluate_action(
            states,
            actions,
            legal_masks[:, :9],
            DecisionType.BID,
            oracle_state=oracle_states if include_oracle_states else None,
        )

    return {
        "states": states,
        "actions": actions,
        "log_probs": old_log_probs.clone(),
        "values": old_values.clone(),
        "advantages": torch.zeros(batch_size),
        "returns": old_values.clone(),
        "decision_types": np.zeros(batch_size, dtype=np.int8),
        "legal_masks": legal_masks,
        "oracle_states": oracle_states if include_oracle_states else None,
        "game_modes": np.zeros(batch_size, dtype=np.int8),
    }


def _mean_actor_critic_cosine(network: TarokNetV4, states: torch.Tensor, oracle_states: torch.Tensor) -> float:
    with torch.no_grad():
        actor = torch.nn.functional.normalize(network.get_actor_features(states), p=2, dim=-1, eps=1e-8)
        critic = torch.nn.functional.normalize(network.get_critic_features(oracle_states), p=2, dim=-1, eps=1e-8)
        return float(torch.nn.functional.cosine_similarity(actor, critic, dim=-1, eps=1e-8).mean().item())


def _actor_weight_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, before_tensor in before.items():
        if key.startswith("critic_backbone") or key.startswith("critic."):
            continue
        total += float((after[key] - before_tensor).abs().sum().item())
    return total


def test_oracle_distillation_moves_actor_toward_critic_features() -> None:
    torch.manual_seed(1234)
    base_network = TarokNetV4(hidden_size=32, oracle_critic=True)
    initial_state = copy.deepcopy(base_network.state_dict())

    adapter_with_oracle = _make_adapter(initial_state, imitation_coef=1.0)
    adapter_without_oracle = _make_adapter(initial_state, imitation_coef=1.0)

    batch_with_oracle = _make_neutral_batch(adapter_with_oracle._network, include_oracle_states=True)  # type: ignore[arg-type]
    batch_without_oracle = _make_neutral_batch(adapter_without_oracle._network, include_oracle_states=False)  # type: ignore[arg-type]

    states = batch_with_oracle["states"]
    oracle_states = batch_with_oracle["oracle_states"]

    before_similarity = _mean_actor_critic_cosine(adapter_with_oracle._network, states, oracle_states)  # type: ignore[arg-type]
    before_with = copy.deepcopy(adapter_with_oracle._network.state_dict())  # type: ignore[union-attr]
    before_without = copy.deepcopy(adapter_without_oracle._network.state_dict())  # type: ignore[union-attr]

    metrics_with = adapter_with_oracle._ppo_update_batched(**batch_with_oracle)
    metrics_without = adapter_without_oracle._ppo_update_batched(**batch_without_oracle)

    after_similarity = _mean_actor_critic_cosine(adapter_with_oracle._network, states, oracle_states)  # type: ignore[arg-type]
    after_with = adapter_with_oracle._network.state_dict()  # type: ignore[union-attr]
    after_without = adapter_without_oracle._network.state_dict()  # type: ignore[union-attr]

    assert metrics_with["il_loss"] > 1e-4
    assert metrics_without["il_loss"] == 0.0
    assert after_similarity > before_similarity + 1e-4
    assert _actor_weight_delta(before_with, after_with) > 0.0
    assert _actor_weight_delta(before_without, after_without) == 0.0
