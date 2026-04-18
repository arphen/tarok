"""Legacy king-calling API backed by the Rust game state."""

from __future__ import annotations

from tarok.use_cases.rust_state import _build_py_state_from_rust
from tarok.entities import Card, GameState


def call_king(state: GameState, king: Card) -> GameState:
    gs = state._rust_gs
    gs.apply_king_call(king._idx)
    snap = _build_py_state_from_rust(
        gs,
        completed_tricks=list(state.tricks),
        bids=list(getattr(state, "bids", [])),
        talon_revealed=getattr(state, "_talon_groups", None),
    )
    snap._trick_in_progress = getattr(state, "_trick_in_progress", None)
    snap._talon_groups = getattr(state, "_talon_groups", None)
    snap._bid_passed = list(getattr(state, "_bid_passed", [False] * 4))
    snap._bid_highest = getattr(state, "_bid_highest", None)
    snap._bid_winner = getattr(state, "_bid_winner", None)
    return snap
