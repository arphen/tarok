"""JSONL human replay-data loading and merge helpers for PPO."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import tarok_engine as te

log = logging.getLogger(__name__)

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
    directly into the self-play batch before the PPO update. Human actions get
    ``log_prob = log(1 / n_legal)``, ``value = 0``.
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

    game_scores: dict[int, dict[int, float]] = {}
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

        first = rows[0]
        key = f"{first.get('game_id', f.stem)}_r{first.get('round', 0)}"
        if key not in file_to_gid:
            file_to_gid[key] = gid_counter
            gid_counter += 1
        gid = file_to_gid[key]

        for row in rows:
            player = int(row.get("player", 0))
            score = float(row.get("final_score", 0))
            game_scores.setdefault(gid, {})[player] = score

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
            lp = -math.log(n_legal)

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
        for player, score in player_scores.items():
            if 0 <= player < 4:
                scores_arr[gid, player] = score

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
    merged_scores = np.concatenate([primary["scores"], extra["scores"]], axis=0)

    def _cat(key: str) -> np.ndarray:
        current = primary[key]
        incoming = extra[key] if key != "game_ids" else extra_game_ids
        if current is None or incoming is None:
            return None  # type: ignore[return-value]
        return np.concatenate([np.asarray(current), np.asarray(incoming)], axis=0)

    return {
        "states": _cat("states"),
        "actions": _cat("actions"),
        "log_probs": _cat("log_probs"),
        "values": _cat("values"),
        "decision_types": _cat("decision_types"),
        "game_modes": _cat("game_modes"),
        "legal_masks": _cat("legal_masks"),
        "game_ids": _cat("game_ids"),
        "players": _cat("players"),
        "scores": merged_scores,
        "oracle_states": primary.get("oracle_states"),
    }