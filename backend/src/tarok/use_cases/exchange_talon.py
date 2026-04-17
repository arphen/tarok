"""Legacy talon-exchange API backed by the Rust game state."""

from __future__ import annotations

from tarok.adapters.ai.rust_game_loop import _build_py_state_from_rust
from tarok.entities import Card, DECK, GameState


def _snapshot(state: GameState) -> GameState:
    snap = _build_py_state_from_rust(
        state._rust_gs,
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


def reveal_talon(state: GameState) -> list[list[Card]]:
    groups = state._rust_gs.build_talon_groups()
    state._rust_gs.set_talon_revealed(groups)
    state._legacy_talon_revealed = [list(g) for g in groups]
    return [[DECK[idx] for idx in group] for group in groups]


def pick_talon_group(state: GameState, group_idx: int) -> GameState:
    state._rust_gs.apply_talon_pick(group_idx)
    snap = _snapshot(state)
    snap._legacy_talon_revealed = getattr(state, "_legacy_talon_revealed", None)
    return snap


def discard_cards(state: GameState, cards: list[Card]) -> GameState:
    state._rust_gs.apply_discards([c._idx for c in cards])
    return _snapshot(state)
