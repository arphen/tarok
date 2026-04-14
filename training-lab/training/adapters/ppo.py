"""Adapter: PPO training — self-contained, no dependency on deleted training_lab.py.

Owns the optimizer, compute backend, and the PPO update loop.
Extracted from the deleted PPOTrainer._ppo_update with zero behavioral changes.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import tarok_engine as te
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

        if config.model_arch != "v4":
            raise ValueError(f"Unsupported model_arch={config.model_arch}. Only 'v4' is supported.")

        self._network = TarokNetV4(hidden_size=hidden_size, oracle_critic=oracle)
        _validate_v4_contract_indices_with_rust()
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
        game_modes = data["game_modes"]

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
    # not a countdown timer.  A step is terminal if it's the last element
    # or if the next step belongs to a different (game, player) trajectory.
    is_terminal = np.ones(n_total, dtype=bool)
    if n_total > 0:
        is_terminal[:-1] = sorted_keys[:-1] != sorted_keys[1:]
    sorted_rewards[~is_terminal] = 0.0

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


_DT_NAME_MAP = {"BID": 0, "KING_CALL": 1, "TALON_PICK": 2, "CARD_PLAY": 3}


def _mode_id_from_state(state: np.ndarray) -> int:
    contract_offset = int(te.CONTRACT_OFFSET)
    contract_size = int(te.CONTRACT_SIZE)
    contract_end = contract_offset + contract_size
    if state.shape[0] < contract_end:
        return 2

    contract_slice = state[contract_offset:contract_end]
    max_idx = int(np.argmax(contract_slice))
    max_val = float(np.max(contract_slice))
    if max_val <= 0.0:
        return 2
    if 4 <= max_idx <= 7:
        return 0
    if max_idx == 0 or max_idx == 8:
        return 1
    if max_idx == 9:
        return 3
    return 2


def load_human_experiences(data_dir: str | Path) -> dict[str, Any] | None:
    """Load all JSONL human-game files and return a raw-experiences dict.

    The format mirrors what the Rust self-play engine emits so it can be merged
    directly into the self-play batch before the PPO update.  Human actions get
    ``log_prob = log(1 / n_legal)`` (uniform prior), ``value = 0``.  GAE will
    give them proper advantage estimates from the actual game outcomes.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        log.warning("human_data_dir=%s has no .jsonl files — skipping", data_dir)
        return None

    states_list: list[np.ndarray] = []
    actions_list: list[int] = []
    log_probs_list: list[float] = []
    values_list: list[float] = []
    decision_types_list: list[int] = []
    legal_masks_list: list[np.ndarray] = []
    game_ids_list: list[int] = []
    players_list: list[int] = []
    game_modes_list: list[int] = []

    # scores: game_id -> {player -> final_score}
    game_scores: dict[int, dict[int, float]] = {}

    # Assign synthetic integer game IDs sequentially
    file_to_gid: dict[str, int] = {}
    gid_counter = 0

    for f in files:
        rows = []
        try:
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception as exc:
            log.warning("Could not read %s: %s", f, exc)
            continue

        if not rows:
            continue

        # Use (game_id, round) as the unique game key
        first = rows[0]
        key = f"{first.get('game_id', f.stem)}_r{first.get('round', 0)}"
        if key not in file_to_gid:
            file_to_gid[key] = gid_counter
            gid_counter += 1
        gid = file_to_gid[key]

        # Collect per-player final scores for this game
        for row in rows:
            p = int(row.get("player", 0))
            score = float(row.get("final_score", 0))
            game_scores.setdefault(gid, {})[p] = score

        for row in rows:
            state = row.get("state")
            action = row.get("action")
            legal_mask = row.get("legal_mask")
            dt_str = row.get("decision_type", "CARD_PLAY")
            player = int(row.get("player", 0))

            if state is None or action is None or legal_mask is None:
                continue

            state_arr = np.asarray(state, dtype=np.float32)
            mask_arr = np.asarray(legal_mask, dtype=np.float32)
            n_legal = max(int(mask_arr.sum()), 1)
            lp = -math.log(n_legal)  # uniform log-prob over legal actions

            states_list.append(state_arr)
            actions_list.append(int(action))
            log_probs_list.append(lp)
            values_list.append(0.0)
            decision_types_list.append(_DT_NAME_MAP.get(str(dt_str), 3))
            game_modes_list.append(_mode_id_from_state(state_arr))
            legal_masks_list.append(mask_arr)
            game_ids_list.append(gid)
            players_list.append(player)

    if not states_list:
        return None

    n_games = gid_counter
    scores_arr = np.zeros((n_games, 4), dtype=np.float32)
    for gid, player_scores in game_scores.items():
        for p, s in player_scores.items():
            if 0 <= p < 4:
                scores_arr[gid, p] = s

    log.info("Loaded %d human decisions from %d games in %s", len(states_list), n_games, data_dir)
    return {
        "states": np.stack(states_list),
        "actions": np.asarray(actions_list, dtype=np.int64),
        "log_probs": np.asarray(log_probs_list, dtype=np.float32),
        "values": np.asarray(values_list, dtype=np.float32),
        "decision_types": np.asarray(decision_types_list, dtype=np.int8),
        "game_modes": np.asarray(game_modes_list, dtype=np.int8),
        "legal_masks": np.stack(legal_masks_list),
        "game_ids": np.asarray(game_ids_list, dtype=np.int64),
        "players": np.asarray(players_list, dtype=np.int8),
        "scores": scores_arr,
        "oracle_states": None,
    }


def merge_experiences(primary: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Concatenate two raw-experience dicts along the sample axis.

    ``extra`` game IDs are offset so they don't collide with ``primary``.
    """
    n_primary_games = int(primary["scores"].shape[0])
    extra_game_ids = np.asarray(extra["game_ids"], dtype=np.int64) + n_primary_games
    # Stack scores (primary rows first, then extra)
    merged_scores = np.concatenate([primary["scores"], extra["scores"]], axis=0)

    def _cat(k: str) -> np.ndarray:
        a = primary[k]
        b = extra[k] if k != "game_ids" else extra_game_ids
        if a is None or b is None:
            return None  # type: ignore[return-value]
        return np.concatenate([np.asarray(a), np.asarray(b)], axis=0)

    return {
        "states": _cat("states"),
        "actions": _cat("actions"),
        "log_probs": _cat("log_probs"),
        "values": _cat("values"),
        "decision_types": _cat("decision_types"),
        "game_modes": _cat("game_modes"),
        "legal_masks": _cat("legal_masks"),
        "game_ids": extra_game_ids if "game_ids" not in primary else _cat("game_ids"),
        "players": _cat("players"),
        "scores": merged_scores,
        "oracle_states": primary.get("oracle_states"),
    }


def _validate_v4_contract_indices_with_rust() -> None:
    """Fail fast if Python and Rust contract indices diverge."""
    try:
        import tarok_engine as te
    except Exception:
        return

    expected = {
        "CONTRACT_KLOP": TarokNetV4._KLOP_IDX,
        "CONTRACT_THREE": TarokNetV4._THREE_IDX,
        "CONTRACT_TWO": TarokNetV4._TWO_IDX,
        "CONTRACT_ONE": TarokNetV4._ONE_IDX,
        "CONTRACT_SOLO_THREE": TarokNetV4._SOLO_THREE_IDX,
        "CONTRACT_SOLO_TWO": TarokNetV4._SOLO_TWO_IDX,
        "CONTRACT_SOLO_ONE": TarokNetV4._SOLO_ONE_IDX,
        "CONTRACT_SOLO": TarokNetV4._SOLO_IDX,
        "CONTRACT_BERAC": TarokNetV4._BERAC_IDX,
        "CONTRACT_BARVNI_VALAT": TarokNetV4._BARVNI_VALAT_IDX,
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
            f"Refusing to train with ambiguous v4 mode routing: {mismatch_text}"
        )
