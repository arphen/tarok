"""Expert replay-data loading helpers for behavioral cloning."""

from __future__ import annotations

from typing import Any

import numpy as np
import tarok_engine as te

from tarok_model.encoding import CARD_ACTION_SIZE


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


def load_expert_experiences(teacher: str, num_games: int) -> dict[str, Any] | None:
    """Load expert trajectories from Rust and convert to raw PPO payload shape.

    The current implementation supports only ``bot_v5`` teacher data via
    ``tarok_engine.generate_expert_data``.
    """
    if num_games <= 0:
        return None
    if teacher != "bot_v5":
        raise ValueError(
            f"Unsupported behavioral_clone_teacher={teacher!r}. Only 'bot_v5' is supported."
        )

    batch = te.generate_expert_data(num_games, include_oracle=False)
    states = np.asarray(batch["states"], dtype=np.float32)
    actions = np.asarray(batch["actions"], dtype=np.int64)
    decision_types = np.asarray(batch["decision_types"], dtype=np.int8)

    n_total = int(actions.shape[0])
    if n_total == 0:
        return None

    legal_masks = np.asarray(batch["legal_masks"], dtype=np.float32)

    # generate_expert_data returns variable-length masks per decision type
    # (BID=9, KING_CALL=4, TALON_PICK=6, CARD_PLAY=54). Walk the flat array
    # and pad each mask to CARD_ACTION_SIZE to match run_self_play output.
    _DT_MASK_SIZE: dict[int, int] = {
        int(te.DT_BID): 9,
        int(te.DT_KING_CALL): 4,
        int(te.DT_TALON_PICK): 6,
        int(te.DT_CARD_PLAY): CARD_ACTION_SIZE,
    }
    padded_masks = np.zeros((n_total, CARD_ACTION_SIZE), dtype=np.float32)
    cursor = 0
    for i, dt in enumerate(decision_types):
        size = _DT_MASK_SIZE.get(int(dt), CARD_ACTION_SIZE)
        padded_masks[i, :size] = legal_masks[cursor : cursor + size]
        cursor += size
    legal_masks = padded_masks
    game_modes = np.asarray([_mode_id_from_state(s) for s in states], dtype=np.int8)

    # Use one pseudo-game per sample to avoid accidental trajectory coupling.
    game_ids = np.arange(n_total, dtype=np.int64)
    players = np.zeros(n_total, dtype=np.int8)

    # Expert samples are supervised via behavioral cloning CE; PPO terms are neutral.
    return {
        "states": states,
        "actions": actions,
        "log_probs": np.zeros(n_total, dtype=np.float32),
        "values": np.zeros(n_total, dtype=np.float32),
        "decision_types": decision_types,
        "game_modes": game_modes,
        "legal_masks": legal_masks,
        "game_ids": game_ids,
        "players": players,
        "scores": np.zeros((n_total, 4), dtype=np.float32),
        "oracle_states": None,
        "behavioral_clone_mask": np.ones(n_total, dtype=bool),
    }
