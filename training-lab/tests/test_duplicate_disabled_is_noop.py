"""Regression: duplicate-disabled must be a perfect no-op.

The core guarantee of ``docs/double_rl.md`` Phase 1/2 is that when
``duplicate.enabled=False`` the training stack behaves *exactly* as it did
before any duplicate code existed. These tests lock that down.
"""

from __future__ import annotations

import numpy as np

from tarok_model.encoding import CARD_ACTION_SIZE, STATE_SIZE
from training.adapters.ppo.ppo_batch_preparation import prepare_batched
from training.entities.duplicate_config import DuplicateConfig
from training.use_cases.resolve_config import _parse_duplicate


def _make_raw(n: int = 6) -> dict:
    rng = np.random.default_rng(0)
    game_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)[:n]
    players = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)[:n]
    n_games = int(game_ids.max()) + 1
    scores = rng.integers(-40, 41, size=(n_games, 4)).astype(np.float32)
    return {
        "states": np.zeros((n, STATE_SIZE), dtype=np.float32),
        "actions": np.arange(n, dtype=np.int64) % CARD_ACTION_SIZE,
        "log_probs": np.zeros(n, dtype=np.float32),
        "values": np.zeros(n, dtype=np.float32),
        "decision_types": np.full(n, 3, dtype=np.int8),
        "game_modes": np.full(n, 2, dtype=np.int8),
        "legal_masks": np.ones((n, CARD_ACTION_SIZE), dtype=np.float32),
        "game_ids": game_ids,
        "players": players,
        "scores": scores,
        "oracle_states": None,
        "behavioral_clone_mask": None,
        "traces": None,
        "declarers": np.full((n_games,), -1, dtype=np.int8),
        "partners": np.full((n_games,), -1, dtype=np.int8),
    }


def test_duplicate_default_config_is_disabled():
    cfg = DuplicateConfig()
    assert cfg.enabled is False
    assert cfg.actor_only is False


def test_parse_duplicate_empty_yields_disabled():
    assert _parse_duplicate({}) == DuplicateConfig()


def test_ppo_batch_prep_without_precomputed_rewards_is_finite():
    raw = _make_raw()
    out = prepare_batched(raw)
    returns = out["vad"][:, 2].detach().cpu().numpy()
    assert np.all(np.isfinite(returns))


def test_ppo_batch_prep_none_precomputed_matches_missing_key():
    raw_missing = _make_raw()
    raw_none = _make_raw()
    raw_none["precomputed_rewards"] = None

    out_missing = prepare_batched(raw_missing)
    out_none = prepare_batched(raw_none)

    for col in (1, 2):  # advantages, returns
        arr_missing = out_missing["vad"][:, col].detach().cpu().numpy()
        arr_none = out_none["vad"][:, col].detach().cpu().numpy()
        assert np.allclose(arr_missing, arr_none)


def test_duplicate_yaml_enabled_does_not_affect_ppo_batch_prep():
    """Enabling duplicate in YAML changes config only; it must not leak into
    prepare_batched, which is pure and takes its reward source from
    ``raw['precomputed_rewards']`` or falls back to ``scores/100``.
    """
    enabled_cfg = _parse_duplicate({"enabled": True, "pods_per_iteration": 2})
    assert enabled_cfg.enabled is True
    raw = _make_raw()
    out = prepare_batched(raw)
    assert out["vad"].shape[0] == raw["players"].shape[0]
