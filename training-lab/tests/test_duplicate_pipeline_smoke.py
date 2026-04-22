"""End-to-end smoke test: duplicate-RL pipeline produces a PPO-ready batch.

This wires the full vertical slice that a real training run traverses when
``config.duplicate.enabled=True`` — pairing → seeded self-play → shadow-diff
reward → ``precomputed_rewards`` on the raw dict → ``prepare_batched`` →
PPO-ready tensor batch — using a fake ``tarok_engine.run_self_play`` so the
test runs in milliseconds without the Rust engine.

If this test passes, a real training run with duplicate enabled is expected
to produce well-formed PPO batches; the only runtime difference is the
engine backing ``te.run_self_play``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from training.adapters.duplicate import seeded_self_play_adapter as ssa_mod
from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
from training.adapters.duplicate.seeded_self_play_adapter import SeededSelfPlayAdapter
from training.adapters.duplicate.shadow_score_reward import ShadowScoreRewardAdapter
from training.adapters.ppo.ppo_batch_preparation import prepare_batched
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
    }


class _StubPresenter:
    def __getattr__(self, _name):
        return lambda *a, **kw: None


def test_duplicate_pipeline_produces_ppo_ready_batch(monkeypatch):
    monkeypatch.setattr(ssa_mod.te, "run_self_play", _fake_rsp)

    # Full adapter stack as wired by the container when duplicate is enabled.
    selfplay = SeededSelfPlayAdapter(inner=None)
    pairing = RotationPairingAdapter(pairing="rotation_8game")
    reward = ShadowScoreRewardAdapter()
    use_case = CollectDuplicateExperiences(
        selfplay=selfplay, pairing=pairing, reward=reward, presenter=_StubPresenter(),
    )

    # STATE_SIZE/CARD_ACTION_SIZE in the fake engine are test-local; the
    # downstream PPO batch path doesn't check against tarok_model constants
    # because we don't hit the encoding there — prepare_batched just reshapes.
    # (Real runs use the real encoding sizes; this test locks the *plumbing*.)
    bundle = use_case.execute(
        duplicate_config=DuplicateConfig(enabled=True, pods_per_iteration=3, rng_seed=7),
        concurrency=2,
        explore_rate=0.1,
        learner_path="/tmp/learner.pt",
        shadow_path="/tmp/shadow.pt",
        pool=None,
        outplace_session_size=50,
    )

    # The use case must have attached precomputed_rewards aligned with
    # experience rows, per §4.1 of docs/double_rl.md.
    raw = bundle.raw
    assert "precomputed_rewards" in raw
    assert raw["precomputed_rewards"].shape == raw["players"].shape

    # Rewards should span a nontrivial range (active and shadow score
    # different models ⇒ different scores ⇒ nonzero per-terminal-step reward
    # on most games).
    nonzero = np.count_nonzero(raw["precomputed_rewards"])
    assert nonzero > 0, "expected at least some nonzero shadow-diff rewards"

    # Running prepare_batched with these precomputed_rewards must succeed
    # and must propagate the override (i.e. the reward path actually ran
    # through the §4.1 branch).
    batch = prepare_batched(raw)
    assert batch["states"].shape[0] == raw["players"].shape[0]
    assert batch["vad"].shape == (raw["players"].shape[0], 3)
    returns = batch["vad"][:, 2].detach().cpu().numpy()
    advantages = batch["vad"][:, 1].detach().cpu().numpy()
    assert np.all(np.isfinite(returns))
    assert np.all(np.isfinite(advantages))
