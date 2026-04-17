"""Legacy king-calling API backed by the Rust game state."""

from __future__ import annotations

from tarok.use_cases.rust_state import _build_py_state_from_rust
from tarok.entities import Card, GameState


def call_king(state: GameState, king: Card) -> GameState:
    gs = state._rust_gs
    gs.apply_king_call(king._idx)
    snap = _build_py_state_from_rust(
        gs,
        completed_tricks=list(getattr(state, "_legacy_tricks", [])),
        bids=list(getattr(state, "bids", [])),
        talon_revealed=getattr(state, "_legacy_talon_revealed", None),
    )
    snap._legacy_tricks = list(getattr(state, "_legacy_tricks", []))
    snap._legacy_current_trick = getattr(state, "_legacy_current_trick", None)
    snap._legacy_talon_revealed = getattr(state, "_legacy_talon_revealed", None)
    snap._legacy_bid_passed = list(getattr(state, "_legacy_bid_passed", [False] * 4))
    snap._legacy_bid_highest = getattr(state, "_legacy_bid_highest", None)
    snap._legacy_bid_winner = getattr(state, "_legacy_bid_winner", None)
    return snap
