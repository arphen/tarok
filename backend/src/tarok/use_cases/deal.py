"""Legacy functional deal API backed by the Rust game state."""

from __future__ import annotations

import random

import tarok_engine as te

from tarok.use_cases.rust_state import _build_py_state_from_rust
from tarok.entities import GameState


def deal(state: GameState, rng: random.Random | None = None) -> GameState:
    """Deal a fresh game and return a Python snapshot.

    The optional ``rng`` argument is kept for legacy test compatibility.
    Rust currently controls shuffle randomness internally.
    """
    _ = rng
    dealer = getattr(state, "dealer", 0)
    gs = te.RustGameState(dealer)
    gs.deal()

    snap = _build_py_state_from_rust(gs, completed_tricks=[], bids=[])
    snap._trick_in_progress = None
    snap._talon_groups = None
    snap._bid_passed = [False] * 4
    snap._bid_highest = None
    snap._bid_winner = None
    return snap
