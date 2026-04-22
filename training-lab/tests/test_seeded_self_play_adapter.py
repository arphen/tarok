"""Unit tests for ``SeededSelfPlayAdapter`` using a fake Rust engine.

The real ``tarok_engine.run_self_play`` requires a genuine NN checkpoint, so
we patch the module-level reference in the adapter with a fake that
deterministically produces scores derived from ``(seat_config, deck_seeds)``.
What we're checking here is the *orchestration* logic of the adapter —
pod→group batching, result stitching, and shadow-score alignment — not the
Rust engine itself (which has its own Rust-side integration test in
``engine-rs/tests/seeded_deal_determinism.rs``).
"""

from __future__ import annotations

import hashlib

import numpy as np

from training.adapters.duplicate import seeded_self_play_adapter as ssa_mod
from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
from training.adapters.duplicate.seeded_self_play_adapter import SeededSelfPlayAdapter


STATE_SIZE = 8  # anything; we never index into it
CARD_ACTION_SIZE = 54


def _deterministic_score(seat_config: str, deck_seed: int, seat: int) -> int:
    h = hashlib.blake2b(
        f"{seat_config}|{deck_seed}|{seat}".encode(),
        digest_size=4,
    ).digest()
    # Return a small signed int in [-50, 50].
    return int.from_bytes(h, "little", signed=True) % 101 - 50


def _fake_run_self_play(**kwargs):
    n_games = kwargs["n_games"]
    seeds = kwargs["deck_seeds"]
    seat_config = kwargs["seat_config"]
    assert seeds is not None
    assert len(seeds) == n_games

    # Produce 2 experiences per game per learner seat.
    seat_labels = seat_config.split(",")
    learner_seats = [i for i, s in enumerate(seat_labels) if s == "nn"]

    states_rows: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    decision_types: list[int] = []
    game_modes: list[int] = []
    game_ids: list[int] = []
    players: list[int] = []
    legal_masks_rows: list[np.ndarray] = []

    scores = np.zeros((n_games, 4), dtype=np.int32)
    for g, seed in enumerate(seeds):
        for seat in range(4):
            scores[g, seat] = _deterministic_score(seat_config, int(seed), seat)
        # Emit 2 synthetic experience rows per learner seat per game.
        for seat in learner_seats:
            for step in range(2):
                states_rows.append(np.full((STATE_SIZE,), g * 10 + step, dtype=np.float32))
                legal_masks_rows.append(np.ones((CARD_ACTION_SIZE,), dtype=np.float32))
                actions.append((g + step) % CARD_ACTION_SIZE)
                log_probs.append(-0.5)
                values.append(0.0)
                decision_types.append(3)  # CARD
                game_modes.append(2)
                game_ids.append(g)
                players.append(seat)

    return {
        "states": np.asarray(states_rows, dtype=np.float32) if states_rows else np.zeros(
            (0, STATE_SIZE), dtype=np.float32
        ),
        "legal_masks": np.asarray(legal_masks_rows, dtype=np.float32) if legal_masks_rows else np.zeros(
            (0, CARD_ACTION_SIZE), dtype=np.float32
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


def _build_pods(n_pods: int, rng_seed: int = 42):
    adapter = RotationPairingAdapter(pairing="rotation_8game")
    return adapter.build_pods(
        pool=None,
        learner_seat_token="nn",
        shadow_seat_token="nn",
        n_pods=n_pods,
        rng_seed=rng_seed,
    )


def test_seeded_adapter_returns_correct_shapes(monkeypatch):
    monkeypatch.setattr(ssa_mod.te, "run_self_play", _fake_run_self_play)
    pods = _build_pods(n_pods=3)

    adapter = SeededSelfPlayAdapter(inner=None)  # inner unused for run_seeded_pods
    result = adapter.run_seeded_pods(
        learner_path="dummy/learner.pt",
        shadow_path="dummy/shadow.pt",
        pods=pods,
        explore_rate=0.05,
        concurrency=4,
    )

    assert result.shadow_scores.shape == (3, 4, 4)  # n_pods, 4 variants, 4 seats
    assert result.learner_positions.shape == (3, 4)
    assert result.active_game_ids.shape == (3, 4)

    # Learner positions should match what the pairing adapter picked.
    for pod_idx, p in enumerate(pods):
        for v in range(4):
            assert int(result.learner_positions[pod_idx, v]) == p.learner_positions[v]

    # Active game ids are globally unique.
    flat = result.active_game_ids.reshape(-1)
    assert len(set(flat.tolist())) == len(flat)


def test_seeded_adapter_active_steps_aligned(monkeypatch):
    monkeypatch.setattr(ssa_mod.te, "run_self_play", _fake_run_self_play)
    pods = _build_pods(n_pods=2)

    adapter = SeededSelfPlayAdapter(inner=None)
    result = adapter.run_seeded_pods(
        learner_path="learner.pt",
        shadow_path="shadow.pt",
        pods=pods,
        explore_rate=0.05,
        concurrency=2,
    )

    active = result.active
    # 2 pods * 4 variants * 1 learner seat * 2 steps = 16 rows.
    assert active["game_ids"].shape == (16,)
    assert result.pod_ids.shape == (16,)
    # Every (pod_id, variant) in active_game_ids matches rows in game_ids.
    for pod_idx in range(2):
        for v in range(4):
            gid = int(result.active_game_ids[pod_idx, v])
            rows = active["game_ids"] == gid
            assert rows.sum() == 2  # 2 steps per learner seat per variant
            assert np.all(result.pod_ids[rows] == pod_idx)


def test_seeded_adapter_rejects_empty_pods():
    adapter = SeededSelfPlayAdapter(inner=None)
    import pytest

    with pytest.raises(ValueError, match="at least one pod"):
        adapter.run_seeded_pods(
            learner_path="l.pt",
            shadow_path="s.pt",
            pods=[],
            explore_rate=0.0,
            concurrency=1,
        )


def test_seeded_adapter_shadow_scores_from_shadow_call(monkeypatch):
    """Shadow scores must come from the shadow (greedy) run, not the active one.

    We patch run_self_play to tag its return by model_path so we can verify
    that ``shadow_scores`` was populated from the shadow invocation.
    """
    captured: list[dict] = []

    def recorder(**kwargs):
        captured.append(
            {
                "seat_config": kwargs["seat_config"],
                "model_path": kwargs["model_path"],
                "explore_rate": kwargs["explore_rate"],
                "deck_seeds": list(kwargs["deck_seeds"]),
            }
        )
        return _fake_run_self_play(**kwargs)

    monkeypatch.setattr(ssa_mod.te, "run_self_play", recorder)
    pods = _build_pods(n_pods=1)

    adapter = SeededSelfPlayAdapter(inner=None)
    adapter.run_seeded_pods(
        learner_path="L.pt",
        shadow_path="S.pt",
        pods=pods,
        explore_rate=0.07,
        concurrency=1,
    )

    # 4 variants × 2 (active+shadow) = 8 calls.
    assert len(captured) == 8
    shadow_calls = [c for c in captured if c["model_path"] == "S.pt"]
    active_calls = [c for c in captured if c["model_path"] == "L.pt"]
    assert len(shadow_calls) == 4
    assert len(active_calls) == 4
    assert all(c["explore_rate"] == 0.0 for c in shadow_calls)
    assert all(c["explore_rate"] == 0.07 for c in active_calls)
    # Each pair of active/shadow calls uses the same deck_seeds list.
    for a, s in zip(active_calls, shadow_calls, strict=True):
        assert a["deck_seeds"] == s["deck_seeds"]
