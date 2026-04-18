"""Legacy functional trick-play API backed by the Rust game state."""

from __future__ import annotations

from tarok.use_cases.rust_state import _build_py_state_from_rust
from tarok.entities import Card, GameState, Trick


def _carry_meta(prev: GameState, snap: GameState) -> GameState:
    snap._trick_in_progress = getattr(prev, "_trick_in_progress", None)
    snap._talon_groups = getattr(prev, "_talon_groups", None)
    snap._bid_passed = list(getattr(prev, "_bid_passed", [False] * 4))
    snap._bid_highest = getattr(prev, "_bid_highest", None)
    snap._bid_winner = getattr(prev, "_bid_winner", None)
    return snap


def start_trick(state: GameState) -> GameState:
    """Start a trick from the current player and return an updated snapshot."""
    gs = state._rust_gs
    lead = getattr(state, "current_player", 0)
    gs.start_trick(lead)

    snap = _build_py_state_from_rust(
        gs,
        completed_tricks=list(state.tricks),
        bids=list(getattr(state, "bids", [])),
        current_trick=(lead, []),
        talon_revealed=getattr(state, "_talon_groups", None),
    )
    snap = _carry_meta(state, snap)
    snap._trick_in_progress = (lead, [])
    return snap


def play_card(state: GameState, player_idx: int, card: Card) -> GameState:
    """Play one card and return an updated snapshot."""
    gs = state._rust_gs
    gs.play_card(player_idx, card._idx)

    lead, cards = getattr(state, "_trick_in_progress", (player_idx, []))
    cards = list(cards)
    cards.append((player_idx, card))

    completed = list(state.tricks)
    current_trick = (lead, cards)

    if len(cards) == 4:
        winner, _points = gs.finish_trick()
        completed.append(Trick(lead_player=lead, cards=cards, winner=winner))
        current_trick = None

    snap = _build_py_state_from_rust(
        gs,
        completed_tricks=completed,
        bids=list(getattr(state, "bids", [])),
        current_trick=current_trick,
        talon_revealed=getattr(state, "_talon_groups", None),
    )
    snap = _carry_meta(state, snap)
    snap._trick_in_progress = current_trick
    return snap
