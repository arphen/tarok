"""Tests for expert replay-data loading (behavioral cloning input pipeline)."""

from __future__ import annotations

import numpy as np
# Pre-load torch so tarok_engine's native extension resolves its Torch deps.
import torch  # noqa: F401
import pytest
import tarok_engine as te

from tarok_model.encoding import CARD_ACTION_SIZE, STATE_SIZE
from training.adapters.ppo import expert_replay


_DT_SIZES = {
    int(te.DT_BID): 9,
    int(te.DT_KING_CALL): 4,
    int(te.DT_TALON_PICK): 6,
    int(te.DT_CARD_PLAY): CARD_ACTION_SIZE,
}


def _build_expert_batch(decision_types: list[int]) -> dict:
    """Synthesise a raw `generate_expert_data`-shaped batch.

    ``decision_types`` drives per-sample mask sizes. We use distinctive mask
    values so padding correctness can be checked directly.
    """
    n = len(decision_types)
    states = np.zeros((n, STATE_SIZE), dtype=np.float32)
    actions = np.arange(n, dtype=np.int64)
    dts = np.asarray(decision_types, dtype=np.int8)

    flat_masks: list[float] = []
    for dt in decision_types:
        size = _DT_SIZES[int(dt)]
        # Pack 1.0 at each index < size so we can verify post-pad equality.
        flat_masks.extend([1.0] * size)
    legal_masks = np.asarray(flat_masks, dtype=np.float32)

    return {
        "states": states,
        "actions": actions,
        "decision_types": dts,
        "legal_masks": legal_masks,
    }


def test_returns_none_when_num_games_is_zero() -> None:
    assert expert_replay.load_expert_experiences("bot_v5", 0) is None


def test_rejects_unsupported_teacher() -> None:
    with pytest.raises(ValueError, match="Unsupported behavioral_clone_teacher"):
        expert_replay.load_expert_experiences("bot_v99", 10)


def test_pads_variable_length_masks_to_card_action_size(monkeypatch: pytest.MonkeyPatch) -> None:
    dts = [int(te.DT_BID), int(te.DT_KING_CALL), int(te.DT_TALON_PICK), int(te.DT_CARD_PLAY)]
    batch = _build_expert_batch(dts)
    monkeypatch.setattr(te, "generate_expert_data", lambda n, include_oracle=False: batch)

    out = expert_replay.load_expert_experiences("bot_v5", 1)

    assert out is not None
    legal_masks = out["legal_masks"]
    assert legal_masks.shape == (4, CARD_ACTION_SIZE)

    # Bid: first 9 cells set; remaining CARD_ACTION_SIZE-9 are zero.
    assert legal_masks[0, :9].sum() == pytest.approx(9.0)
    assert legal_masks[0, 9:].sum() == pytest.approx(0.0)
    # King-call: first 4 cells set.
    assert legal_masks[1, :4].sum() == pytest.approx(4.0)
    assert legal_masks[1, 4:].sum() == pytest.approx(0.0)
    # Talon-pick: first 6 cells set.
    assert legal_masks[2, :6].sum() == pytest.approx(6.0)
    assert legal_masks[2, 6:].sum() == pytest.approx(0.0)
    # Card-play already matches CARD_ACTION_SIZE.
    assert legal_masks[3].sum() == pytest.approx(CARD_ACTION_SIZE)


def test_produces_bc_only_payload_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    dts = [int(te.DT_CARD_PLAY), int(te.DT_CARD_PLAY)]
    batch = _build_expert_batch(dts)
    monkeypatch.setattr(te, "generate_expert_data", lambda n, include_oracle=False: batch)

    out = expert_replay.load_expert_experiences("bot_v5", 42)

    assert out is not None
    assert out["behavioral_clone_mask"].shape == (2,)
    assert out["behavioral_clone_mask"].dtype == bool
    # Expert samples always flow as BC supervised signal.
    assert out["behavioral_clone_mask"].all()
    # PPO channels must be neutral so expert samples don't pollute RL losses.
    assert np.allclose(out["log_probs"], 0.0)
    assert np.allclose(out["values"], 0.0)
    assert out["oracle_states"] is None
    # game_ids must be unique so they don't collide with self-play trajectories.
    assert len(np.unique(out["game_ids"])) == len(out["game_ids"])
    # scores array is shaped for merge-time rebase, but all zeros.
    assert out["scores"].shape == (2, 4)
    assert np.all(out["scores"] == 0.0)


def test_returns_none_when_batch_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = {
        "states": np.zeros((0, STATE_SIZE), dtype=np.float32),
        "actions": np.zeros(0, dtype=np.int64),
        "decision_types": np.zeros(0, dtype=np.int8),
        "legal_masks": np.zeros(0, dtype=np.float32),
    }
    monkeypatch.setattr(te, "generate_expert_data", lambda n, include_oracle=False: empty)

    assert expert_replay.load_expert_experiences("bot_v5", 5) is None


def test_game_mode_derived_from_contract_slice_in_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Construct two samples:
    #   sample 0 — contract slice empty (max_val == 0) → game_mode default 2
    #   sample 1 — contract argmax at index 5 (inside 4..7 range) → game_mode 0
    dts = [int(te.DT_CARD_PLAY), int(te.DT_CARD_PLAY)]
    batch = _build_expert_batch(dts)
    offset = int(te.CONTRACT_OFFSET)
    batch["states"][1, offset + 5] = 0.8  # force argmax at idx 5

    monkeypatch.setattr(te, "generate_expert_data", lambda n, include_oracle=False: batch)

    out = expert_replay.load_expert_experiences("bot_v5", 2)

    assert out is not None
    assert out["game_modes"].tolist() == [2, 0]
