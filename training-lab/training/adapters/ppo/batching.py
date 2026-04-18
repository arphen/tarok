"""Batch preparation helpers for PPO."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import tarok_engine as te
import torch


def prepare_batched(raw: dict[str, Any], gamma: float = 0.99, gae_lambda: float = 0.95) -> dict[str, Any]:
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

    game_modes_np = np.asarray(raw["game_modes"], dtype=np.int8)

    n_total = len(actions_np)
    gids = game_ids_np % scores_np.shape[0]
    rewards_np = scores_np[gids, players_np].astype(np.float32) / 100.0

    traj_keys = game_ids_np.astype(np.int64) * 4 + players_np.astype(np.int64)
    sort_idx = np.lexsort((np.arange(n_total), traj_keys))

    sorted_keys = np.asarray(traj_keys[sort_idx], dtype=np.int64)
    sorted_values = np.asarray(values_np[sort_idx], dtype=np.float32)
    sorted_rewards = np.asarray(rewards_np[sort_idx], dtype=np.float32)

    # Zero out non-terminal rewards so the critic learns position value,
    # not a countdown timer. A step is terminal if it's the last element
    # or if the next step belongs to a different (game, player) trajectory.
    is_terminal = np.ones(n_total, dtype=bool)
    if n_total > 0:
        is_terminal[:-1] = sorted_keys[:-1] != sorted_keys[1:]
    sorted_rewards[~is_terminal] = 0.0

    advantages_sorted, returns_sorted = te.compute_gae(
        sorted_values,
        sorted_rewards,
        sorted_keys,
        gamma=gamma,
        gae_lambda=gae_lambda,
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

    # Stack values / advantages / returns into a single (N, 3) matrix so that
    # one advanced-index copy on the CPU and one PCIe transfer replace three.
    # Column layout: 0=old_values, 1=advantages (normalised), 2=returns.
    vad_np = np.stack([values_np.astype(np.float32), advantages_np, returns_np], axis=1)

    return {
        "states": torch.from_numpy(states_np),
        "actions": torch.from_numpy(actions_np.astype(np.int64)),
        "log_probs": torch.from_numpy(log_probs_np),
        "vad": torch.from_numpy(vad_np),
        "decision_types": decision_types_np,
        "legal_masks": torch.from_numpy(legal_masks_np),
        "oracle_states": torch.from_numpy(oracle_states_np) if oracle_states_np is not None else None,
        "game_modes": game_modes_np,
    }


def release_allocator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
