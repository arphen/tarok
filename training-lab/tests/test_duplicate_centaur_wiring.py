"""Tests for duplicate-centaur wiring.

Locks in the plumbing that lets a training run set
``duplicate.learner_seat_token: centaur`` and forwards the top-level
``centaur_*`` knobs through ``CollectDuplicateExperiences`` into
``SelfPlayPort.run_seeded_pods``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from training.adapters.duplicate import seeded_self_play_adapter as ssa_mod
from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
from training.adapters.duplicate.seeded_self_play_adapter import (
    SeededSelfPlayAdapter,
    _render_seat_config,
)
from training.adapters.duplicate.shadow_score_reward import ShadowScoreRewardAdapter
from training.entities.duplicate_config import DuplicateConfig
from training.use_cases.collect_duplicate_experiences import CollectDuplicateExperiences


CARD_ACTION_SIZE = 54
STATE_SIZE = 8


def _det_score(seat_config: str, seed: int, seat: int, model_path: str) -> int:
    h = hashlib.blake2b(
        f"{seat_config}|{seed}|{seat}|{model_path}".encode(), digest_size=4
    ).digest()
    return int.from_bytes(h, "little", signed=True) % 201 - 100


def _make_fake_rsp(captured: list[dict[str, Any]]):
    """Fake run_self_play that treats both 'nn' and 'centaur' as learner seats.

    Captures every kwargs dict so tests can assert on the plumbing.
    """

    def _fake(**kwargs):
        captured.append(dict(kwargs))
        n_games = kwargs["n_games"]
        seeds = list(kwargs["deck_seeds"])
        seat_config = kwargs["seat_config"]
        model_path = kwargs.get("model_path") or ""
        seat_labels = seat_config.split(",")
        learner_seats = [
            i for i, s in enumerate(seat_labels) if s in ("nn", "centaur")
        ]

        rows_states: list = []
        rows_masks: list = []
        actions, log_probs, values = [], [], []
        decision_types, game_modes, game_ids, players = [], [], [], []
        scores = np.zeros((n_games, 4), dtype=np.int32)
        for g, seed in enumerate(seeds):
            for seat in range(4):
                scores[g, seat] = _det_score(seat_config, int(seed), seat, model_path)
            for seat in learner_seats:
                for step in range(3):
                    rows_states.append(np.full((STATE_SIZE,), 0.1, dtype=np.float32))
                    rows_masks.append(np.ones((CARD_ACTION_SIZE,), dtype=np.float32))
                    actions.append(step % CARD_ACTION_SIZE)
                    log_probs.append(-0.1)
                    values.append(0.0)
                    decision_types.append(3)
                    game_modes.append(2)
                    game_ids.append(g)
                    players.append(seat)

        return {
            "states": (
                np.asarray(rows_states, dtype=np.float32)
                if rows_states
                else np.zeros((0, STATE_SIZE), dtype=np.float32)
            ),
            "legal_masks": (
                np.asarray(rows_masks, dtype=np.float32)
                if rows_masks
                else np.zeros((0, CARD_ACTION_SIZE), dtype=np.float32)
            ),
            "actions": np.asarray(actions, dtype=np.uint16),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "decision_types": np.asarray(decision_types, dtype=np.uint8),
            "game_modes": np.asarray(game_modes, dtype=np.int8),
            "game_ids": np.asarray(game_ids, dtype=np.uint32),
            "players": np.asarray(players, dtype=np.uint8),
            "scores": scores,
        }

    return _fake


class _StubPresenter:
    def __getattr__(self, _name):
        return lambda *a, **kw: None


# ---- DuplicateConfig validation --------------------------------------------


def test_learner_seat_token_default_is_nn() -> None:
    assert DuplicateConfig().learner_seat_token == "nn"


def test_learner_seat_token_accepts_centaur() -> None:
    cfg = DuplicateConfig(learner_seat_token="centaur")
    assert cfg.learner_seat_token == "centaur"


def test_learner_seat_token_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="learner_seat_token"):
        DuplicateConfig(learner_seat_token="bot_v5")


# ---- _render_seat_config preserves centaur ---------------------------------


def test_render_seat_config_preserves_centaur_token() -> None:
    seating = ("centaur", "bot_v5", "bot_m6", "bot_v3")
    cfg, path = _render_seat_config(seating, learner_pos=0, actor_path="/tmp/l.pt")
    assert cfg == "centaur,bot_v5,bot_m6,bot_v3"
    assert path == "/tmp/l.pt"


def test_render_seat_config_preserves_nn_token() -> None:
    seating = ("bot_v5", "nn", "bot_m6", "bot_v3")
    cfg, _ = _render_seat_config(seating, learner_pos=1, actor_path="/tmp/l.pt")
    assert cfg == "bot_v5,nn,bot_m6,bot_v3"


# ---- End-to-end: centaur learner forwards through run_seeded_pods -----------


def test_collect_duplicate_centaur_learner_routes_centaur_seat(monkeypatch):
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(ssa_mod.te, "run_self_play", _make_fake_rsp(captured))

    selfplay = SeededSelfPlayAdapter(inner=object())  # type: ignore[arg-type]
    pairing = RotationPairingAdapter(pairing="rotation_8game")
    reward = ShadowScoreRewardAdapter(score_scale=100.0)
    uc = CollectDuplicateExperiences(selfplay, pairing, reward, _StubPresenter())

    cfg = DuplicateConfig(
        enabled=True,
        pods_per_iteration=1,
        rng_seed=7,
        learner_seat_token="centaur",
    )
    bundle = uc.execute(
        duplicate_config=cfg,
        concurrency=1,
        explore_rate=0.05,
        learner_path="/tmp/learner.pt",
        shadow_path="/tmp/shadow.pt",
        pool=None,
        outplace_session_size=1,
        centaur_handoff_trick=8,
        centaur_pimc_worlds=50,
        centaur_endgame_solver="pimc",
        centaur_alpha_mu_depth=2,
        centaur_deterministic_seed=1,
    )

    # Every engine invocation must have the centaur token at the learner
    # seat AND the centaur_* knobs forwarded unchanged.
    assert captured, "fake run_self_play was never invoked"
    for call in captured:
        labels = call["seat_config"].split(",")
        assert "centaur" in labels
        assert "nn" not in labels
        assert call["centaur_handoff_trick"] == 8
        assert call["centaur_pimc_worlds"] == 50
        assert call["centaur_endgame_solver"] == "pimc"
        assert call["centaur_alpha_mu_depth"] == 2
        assert call["centaur_deterministic_seed"] == 1

    # And the pipeline still produces a valid bundle with precomputed rewards.
    assert "precomputed_rewards" in bundle.raw
    assert bundle.raw["precomputed_rewards"].shape == bundle.raw["players"].shape


def test_collect_duplicate_nn_learner_still_routes_nn_seat(monkeypatch):
    """Default (no centaur) must keep pre-existing behavior: seat_config has 'nn'."""
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(ssa_mod.te, "run_self_play", _make_fake_rsp(captured))

    selfplay = SeededSelfPlayAdapter(inner=object())  # type: ignore[arg-type]
    pairing = RotationPairingAdapter(pairing="rotation_8game")
    reward = ShadowScoreRewardAdapter(score_scale=100.0)
    uc = CollectDuplicateExperiences(selfplay, pairing, reward, _StubPresenter())

    cfg = DuplicateConfig(enabled=True, pods_per_iteration=1, rng_seed=3)
    uc.execute(
        duplicate_config=cfg,
        concurrency=1,
        explore_rate=0.05,
        learner_path="/tmp/learner.pt",
        shadow_path="/tmp/shadow.pt",
        pool=None,
        outplace_session_size=1,
    )

    assert captured
    for call in captured:
        labels = call["seat_config"].split(",")
        assert "nn" in labels
        assert "centaur" not in labels
        # When not provided, centaur_* knobs are None.
        assert call["centaur_handoff_trick"] is None
        assert call["centaur_pimc_worlds"] is None
