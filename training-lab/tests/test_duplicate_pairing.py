"""Tests for RotationPairingAdapter."""

from __future__ import annotations

import pytest

from training.adapters.duplicate.rotation_pairing import RotationPairingAdapter
from training.entities.duplicate_pod import DuplicatePod


def test_build_pods_yields_requested_count() -> None:
    adapter = RotationPairingAdapter()
    pods = adapter.build_pods(
        pool=None,
        learner_seat_token="nn",
        shadow_seat_token="shadow.pt",
        n_pods=5,
        rng_seed=42,
    )
    assert len(pods) == 5
    for pod in pods:
        assert isinstance(pod, DuplicatePod)


def test_rotation_8game_has_learner_at_every_seat() -> None:
    adapter = RotationPairingAdapter("rotation_8game")
    pods = adapter.build_pods(
        pool=None,
        learner_seat_token="nn",
        shadow_seat_token="shadow.pt",
        n_pods=1,
        rng_seed=0,
    )
    pod = pods[0]
    assert len(pod.active_seatings) == 4
    assert len(pod.shadow_seatings) == 4
    assert pod.learner_positions == (0, 1, 2, 3)
    # Learner appears at its rotating seat in each active game and nowhere else.
    for seating, pos in zip(pod.active_seatings, pod.learner_positions, strict=True):
        assert seating[pos] == "nn"
        for i, tok in enumerate(seating):
            if i != pos:
                assert tok != "nn"
    # Shadow table: learner replaced by shadow at the same position.
    for seating, pos in zip(pod.shadow_seatings, pod.learner_positions, strict=True):
        assert seating[pos] == "shadow.pt"
        assert "nn" not in seating


def test_rotation_8game_opponents_are_constant_across_pod() -> None:
    adapter = RotationPairingAdapter("rotation_8game")
    pod = adapter.build_pods(None, "nn", "shadow", n_pods=1, rng_seed=0)[0]
    assert len(set(pod.opponents)) == 3  # three distinct opponent tokens
    opps_set = set(pod.opponents)
    for seating in pod.active_seatings:
        non_learner = [t for t in seating if t != "nn"]
        assert set(non_learner) == opps_set


def test_deterministic_under_same_seed() -> None:
    a = RotationPairingAdapter()
    b = RotationPairingAdapter()
    pods_a = a.build_pods(None, "nn", "shadow", n_pods=3, rng_seed=7)
    pods_b = b.build_pods(None, "nn", "shadow", n_pods=3, rng_seed=7)
    assert [p.deck_seed for p in pods_a] == [p.deck_seed for p in pods_b]
    assert [p.opponents for p in pods_a] == [p.opponents for p in pods_b]


def test_different_seeds_yield_different_decks() -> None:
    adapter = RotationPairingAdapter()
    pods_a = adapter.build_pods(None, "nn", "shadow", n_pods=3, rng_seed=1)
    pods_b = adapter.build_pods(None, "nn", "shadow", n_pods=3, rng_seed=2)
    assert [p.deck_seed for p in pods_a] != [p.deck_seed for p in pods_b]


def test_n_pods_zero_yields_empty_list() -> None:
    adapter = RotationPairingAdapter()
    assert adapter.build_pods(None, "nn", "shadow", n_pods=0, rng_seed=0) == []


def test_n_pods_negative_rejected() -> None:
    adapter = RotationPairingAdapter()
    with pytest.raises(ValueError, match="n_pods"):
        adapter.build_pods(None, "nn", "shadow", n_pods=-1, rng_seed=0)


def test_rotation_4game_has_two_games_at_same_seat() -> None:
    adapter = RotationPairingAdapter("rotation_4game")
    pod = adapter.build_pods(None, "nn", "shadow", n_pods=1, rng_seed=0)[0]
    assert len(pod.active_seatings) == 2
    assert pod.learner_positions == (0, 0)


def test_single_seat_2game_has_one_active_game() -> None:
    adapter = RotationPairingAdapter("single_seat_2game")
    pod = adapter.build_pods(None, "nn", "shadow", n_pods=1, rng_seed=0)[0]
    assert len(pod.active_seatings) == 1
    assert len(pod.shadow_seatings) == 1
    assert pod.learner_positions == (0,)


def test_unknown_pairing_rejected() -> None:
    with pytest.raises(ValueError, match="pairing"):
        RotationPairingAdapter("rotation_16game")
