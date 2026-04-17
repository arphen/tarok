"""Legacy functional bidding API backed by the Rust game state."""

from __future__ import annotations

from tarok.adapters.ai.rust_game_loop import (
    _PY_CONTRACT_TO_RUST_U8,
    _build_py_state_from_rust,
)
from tarok.entities import Bid, Contract, GameState


def _next_active_bidder(current: int, passed: list[bool]) -> int:
    bidder = current
    for _ in range(4):
        bidder = (bidder + 1) % 4
        if not passed[bidder]:
            return bidder
    return current


def place_bid(state: GameState, player_idx: int, bid: Contract | None) -> GameState:
    """Apply one bid action and return an updated snapshot."""
    gs = state._rust_gs
    bids = list(getattr(state, "bids", []))
    passed = list(getattr(state, "_legacy_bid_passed", [False] * 4))
    highest = getattr(state, "_legacy_bid_highest", None)
    winner = getattr(state, "_legacy_bid_winner", None)

    gs.current_player = player_idx

    if bid is None:
        passed[player_idx] = True
        gs.add_bid(player_idx, None)
        bids.append(Bid(player=player_idx, contract=None))
    else:
        rust_bid = _PY_CONTRACT_TO_RUST_U8[bid]
        legal = list(gs.legal_bids(player_idx))
        if rust_bid not in legal:
            passed[player_idx] = True
            gs.add_bid(player_idx, None)
            bids.append(Bid(player=player_idx, contract=None))
        else:
            gs.add_bid(player_idx, rust_bid)
            bids.append(Bid(player=player_idx, contract=bid))
            highest = rust_bid
            winner = player_idx

    active = [i for i in range(4) if not passed[i]]
    bidding_done = (len(active) <= 1 and winner is not None) or len(active) == 0

    if bidding_done:
        gs.resolve_bidding(winner=winner, contract=highest)
        current_bidder = None
    else:
        next_bidder = _next_active_bidder(player_idx, passed)
        gs.current_player = next_bidder
        current_bidder = next_bidder

    snap = _build_py_state_from_rust(
        gs,
        completed_tricks=list(getattr(state, "_legacy_tricks", [])),
        bids=bids,
        current_bidder=current_bidder,
        talon_revealed=getattr(state, "_legacy_talon_revealed", None),
    )
    snap._legacy_tricks = list(getattr(state, "_legacy_tricks", []))
    snap._legacy_current_trick = getattr(state, "_legacy_current_trick", None)
    snap._legacy_talon_revealed = getattr(state, "_legacy_talon_revealed", None)
    snap._legacy_bid_passed = passed
    snap._legacy_bid_highest = highest
    snap._legacy_bid_winner = winner
    return snap
