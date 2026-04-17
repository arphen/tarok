"""Legacy functional trick-play API backed by the Rust game state."""

from __future__ import annotations

from tarok.adapters.ai.rust_game_loop import _build_py_state_from_rust
from tarok.entities import Card, GameState, Trick


def _carry_meta(prev: GameState, snap: GameState) -> GameState:
    snap._legacy_tricks = list(getattr(prev, "_legacy_tricks", []))
    snap._legacy_current_trick = getattr(prev, "_legacy_current_trick", None)
    snap._legacy_talon_revealed = getattr(prev, "_legacy_talon_revealed", None)
    snap._legacy_bid_passed = list(getattr(prev, "_legacy_bid_passed", [False] * 4))
    snap._legacy_bid_highest = getattr(prev, "_legacy_bid_highest", None)
    snap._legacy_bid_winner = getattr(prev, "_legacy_bid_winner", None)
    return snap


def start_trick(state: GameState) -> GameState:
    """Start a trick from the current player and return an updated snapshot."""
    gs = state._rust_gs
    lead = getattr(state, "current_player", 0)
    gs.start_trick(lead)

    snap = _build_py_state_from_rust(
        gs,
        completed_tricks=list(getattr(state, "_legacy_tricks", [])),
        bids=list(getattr(state, "bids", [])),
        current_trick=(lead, []),
        talon_revealed=getattr(state, "_legacy_talon_revealed", None),
    )
    snap = _carry_meta(state, snap)
    snap._legacy_current_trick = (lead, [])
    return snap


def play_card(state: GameState, player_idx: int, card: Card) -> GameState:
    """Play one card and return an updated snapshot."""
    gs = state._rust_gs
    gs.play_card(player_idx, card._idx)

    lead, cards = getattr(state, "_legacy_current_trick", (player_idx, []))
    cards = list(cards)
    cards.append((player_idx, card))

    completed = list(getattr(state, "_legacy_tricks", []))
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
        talon_revealed=getattr(state, "_legacy_talon_revealed", None),
    )
    snap = _carry_meta(state, snap)
    snap._legacy_tricks = completed
    snap._legacy_current_trick = current_trick
    return snap
