"""End-to-end test: CollectDuplicateExperiences with fake Rust engine.

Wires together the real pairing adapter, real seeded self-play adapter (with
a mocked ``tarok_engine.run_self_play``), and the real reward adapter. This
asserts the Phase 2 flow produces a bundle whose ``raw["precomputed_rewards"]``
is consumed by the PPO batch path (§4.1) without any other behavior change.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from training.adapters.duplicate import seeded_self_play_adapter as ssa_mod
from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
from training.adapters.duplicate.seeded_self_play_adapter import SeededSelfPlayAdapter
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


def _fake_rsp(**kwargs):
    n_games = kwargs["n_games"]
    seeds = list(kwargs["deck_seeds"])
    seat_config = kwargs["seat_config"]
    model_path = kwargs.get("model_path") or ""
    seat_labels = seat_config.split(",")
    learner_seats = [i for i, s in enumerate(seat_labels) if s == "nn"]

    rows_states = []
    rows_masks = []
    actions, log_probs, values = [], [], []
    decision_types, game_modes, game_ids, players = [], [], [], []
    scores = np.zeros((n_games, 4), dtype=np.int32)
    bid_contracts = np.zeros((n_games, 4), dtype=np.int8)
    contracts = np.zeros((n_games,), dtype=np.uint8)
    declarers = np.zeros((n_games,), dtype=np.int8)
    partners = np.zeros((n_games,), dtype=np.int8)

    for g, seed in enumerate(seeds):
        for seat in range(4):
            scores[g, seat] = _det_score(seat_config, int(seed), seat, model_path)
        # Deterministic bid contract values based on seed
        bid_contracts[g, :] = np.array([g % 3, (g + 1) % 3, (g + 2) % 3, (g + 3) % 3], dtype=np.int8)
        contracts[g] = g % 8
        declarers[g] = g % 4
        partners[g] = (g + 1) % 4
        
        # 3 steps per learner seat per game.
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
        "states": np.asarray(rows_states, dtype=np.float32)
        if rows_states
        else np.zeros((0, STATE_SIZE), dtype=np.float32),
        "legal_masks": np.asarray(rows_masks, dtype=np.float32)
        if rows_masks
        else np.zeros((0, CARD_ACTION_SIZE), dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.uint16),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "decision_types": np.asarray(decision_types, dtype=np.uint8),
        "game_modes": np.asarray(game_modes, dtype=np.int8),
        "game_ids": np.asarray(game_ids, dtype=np.uint32),
        "players": np.asarray(players, dtype=np.uint8),
        "scores": scores,
        "bid_contracts": bid_contracts,
        "contracts": contracts,
        "declarers": declarers,
        "partners": partners,
    }


class _StubPresenter:
    def __init__(self):
        self.events: list[tuple] = []

    def on_selfplay_start(self, *a, **kw):
        self.events.append(("start", a, kw))

    def on_selfplay_done(self, *a, **kw):
        self.events.append(("done", a, kw))

    def __getattr__(self, name):
        def _noop(*a, **kw):
            self.events.append((name, a, kw))

        return _noop


def _build_use_case(monkeypatch):
    monkeypatch.setattr(ssa_mod.te, "run_self_play", _fake_rsp)
    inner = object()  # unused in duplicate path
    selfplay = SeededSelfPlayAdapter(inner=inner)  # type: ignore[arg-type]
    pairing = RotationPairingAdapter(pairing="rotation_8game")
    reward = ShadowScoreRewardAdapter(score_scale=100.0)
    presenter = _StubPresenter()
    return CollectDuplicateExperiences(selfplay, pairing, reward, presenter), presenter


def test_collect_duplicate_experiences_attaches_precomputed_rewards(monkeypatch):
    uc, presenter = _build_use_case(monkeypatch)

    cfg = DuplicateConfig(enabled=True, pods_per_iteration=2, rng_seed=7)
    bundle = uc.execute(
        duplicate_config=cfg,
        concurrency=2,
        explore_rate=0.05,
        learner_path="learner.pt",
        shadow_path="shadow.pt",
        pool=None,
        outplace_session_size=2,
    )

    assert "precomputed_rewards" in bundle.raw
    rewards = bundle.raw["precomputed_rewards"]
    n_total = bundle.raw["players"].shape[0]
    assert rewards.shape == (n_total,)
    # Terminal reward rows are non-zero for most rows; all non-terminal rows
    # contribute 0. At minimum there should be at least one non-zero.
    assert np.any(rewards != 0.0)

    # n_total = 2 pods * 4 variants * 1 learner seat * 3 steps = 24 rows.
    assert n_total == 24
    assert bundle.n_learner == 24

    # Presenter saw exactly one "done" event.
    done_events = [e for e in presenter.events if e[0] == "done"]
    assert len(done_events) == 1


def test_collect_duplicate_experiences_requires_enabled(monkeypatch):
    uc, _ = _build_use_case(monkeypatch)
    with pytest.raises(ValueError, match="duplicate.enabled=False"):
        uc.execute(
            duplicate_config=DuplicateConfig(enabled=False),
            concurrency=1,
            explore_rate=0.0,
            learner_path="l.pt",
            shadow_path="s.pt",
            pool=None,
            outplace_session_size=1,
        )


def test_collect_duplicate_experiences_rejects_zero_pods(monkeypatch):
    uc, _ = _build_use_case(monkeypatch)
    with pytest.raises(ValueError, match="pods_per_iteration must be > 0"):
        uc.execute(
            duplicate_config=DuplicateConfig(enabled=True, pods_per_iteration=0),
            concurrency=1,
            explore_rate=0.0,
            learner_path="l.pt",
            shadow_path="s.pt",
            pool=None,
            outplace_session_size=1,
        )


def test_collect_duplicate_experiences_reward_sign_is_active_minus_shadow(monkeypatch):
    """Verify reward sign: when shadow beats active by a known margin,
    rewards at terminal steps should equal (active − shadow) / 100 < 0.
    """

    def constant_rsp(**kwargs):
        base = _fake_rsp(**kwargs)
        base["scores"] = np.zeros_like(base["scores"])
        # Give shadow (explore_rate == 0) an extra +20 at every seat.
        if kwargs.get("explore_rate", 0.0) == 0.0:
            base["scores"][:] = 20
        return base

    monkeypatch.setattr(ssa_mod.te, "run_self_play", constant_rsp)
    selfplay = SeededSelfPlayAdapter(inner=object())  # type: ignore[arg-type]
    pairing = RotationPairingAdapter(pairing="rotation_8game")
    reward = ShadowScoreRewardAdapter(score_scale=100.0)
    presenter = _StubPresenter()
    uc = CollectDuplicateExperiences(selfplay, pairing, reward, presenter)

    cfg = DuplicateConfig(enabled=True, pods_per_iteration=1, rng_seed=1)
    bundle = uc.execute(
        duplicate_config=cfg,
        concurrency=1,
        explore_rate=0.05,
        learner_path="L.pt",
        shadow_path="S.pt",
        pool=None,
        outplace_session_size=1,
    )

    rewards = bundle.raw["precomputed_rewards"]
    nonzero = rewards[rewards != 0.0]
    # All terminal rewards should be (0 − 20) / 100 = −0.2.
    assert nonzero.size > 0
    assert np.allclose(nonzero, -0.2, atol=1e-6)


def test_collect_duplicate_experiences_includes_bid_contracts(monkeypatch):
    """Verify that bid_contracts metadata from engine is merged correctly."""
    monkeypatch.setattr(ssa_mod.te, "run_self_play", _fake_rsp)
    selfplay = SeededSelfPlayAdapter(inner=object())  # type: ignore[arg-type]
    pairing = RotationPairingAdapter(pairing="rotation_8game")
    reward = ShadowScoreRewardAdapter(score_scale=100.0)
    presenter = _StubPresenter()
    uc = CollectDuplicateExperiences(selfplay, pairing, reward, presenter)

    cfg = DuplicateConfig(enabled=True, pods_per_iteration=2, rng_seed=1)
    bundle = uc.execute(
        duplicate_config=cfg,
        concurrency=1,
        explore_rate=0.05,
        learner_path="L.pt",
        shadow_path="S.pt",
        pool=None,
        outplace_session_size=1,
    )

    # Verify bid_contracts is present and has correct shape
    assert "bid_contracts" in bundle.raw, "bid_contracts missing from merged data"
    bid_contracts = bundle.raw["bid_contracts"]
    assert bid_contracts.ndim == 2, f"Expected 2D array, got {bid_contracts.ndim}D"
    assert bid_contracts.shape[1] == 4, f"Expected 4 columns (seats), got {bid_contracts.shape[1]}"
    
    # Verify other metadata keys are also present
    for key in ["contracts", "declarers", "partners"]:
        assert key in bundle.raw, f"{key} missing from merged data"

