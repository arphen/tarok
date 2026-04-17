"""Coverage tests for PPO human-experience loading and merging helpers."""

from __future__ import annotations

import json

import numpy as np

from tarok_model.encoding import CARD_ACTION_SIZE, STATE_SIZE
from training.adapters.ppo import load_human_experiences, merge_experiences


def test_load_and_merge_human_experiences(tmp_path) -> None:
    # 1) Create a tiny valid human-data JSONL corpus.
    data_dir = tmp_path / "human_data"
    data_dir.mkdir()
    file_path = data_dir / "game_1.jsonl"

    dummy_row = {
        "game_id": "game1",
        "round": 1,
        "player": 0,
        "final_score": 55.0,
        "action": 12,
        "decision_type": "CARD_PLAY",
        "state": [0.0] * STATE_SIZE,
        "legal_mask": [1.0] * CARD_ACTION_SIZE,
    }

    with file_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(dummy_row) + "\n")

    # 2) Loader behavior.
    loaded = load_human_experiences(data_dir)
    assert loaded is not None
    assert loaded["actions"].shape == (1,)
    assert loaded["actions"][0] == 12
    assert loaded["scores"].shape == (1, 4)
    assert loaded["scores"][0, 0] == 55.0

    # log_prob should be -log(n_legal); here all actions are legal.
    assert loaded["log_probs"].shape == (1,)
    assert loaded["log_probs"][0] == np.float32(-np.log(CARD_ACTION_SIZE))

    # 3) Merge behavior with synthetic primary self-play batch.
    primary = {
        "states": np.zeros((2, STATE_SIZE), dtype=np.float32),
        "actions": np.array([1, 2], dtype=np.int64),
        "log_probs": np.array([-0.1, -0.2], dtype=np.float32),
        "values": np.array([0.5, 0.6], dtype=np.float32),
        "decision_types": np.array([3, 3], dtype=np.int8),
        "game_modes": np.array([2, 2], dtype=np.int8),
        "legal_masks": np.ones((2, CARD_ACTION_SIZE), dtype=np.float32),
        "game_ids": np.array([0, 0], dtype=np.int64),
        "players": np.array([1, 2], dtype=np.int8),
        "scores": np.zeros((1, 4), dtype=np.float32),
        "oracle_states": None,
    }

    merged = merge_experiences(primary, loaded)

    assert merged["actions"].shape == (3,)
    assert merged["scores"].shape == (2, 4)
    # Loaded game IDs are offset by primary game count (1).
    assert merged["game_ids"][2] == 1
    # Last merged action comes from loaded sample.
    assert merged["actions"][2] == 12
