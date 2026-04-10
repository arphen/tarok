"""Tests for the WebSocket observer card tracker caching."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tarok.adapters.api.ws_observer import (
    WebSocketObserver,
    _build_card_tracker,
    _state_for_player,
)
from tarok.entities.card import Card, CardType, Suit, SuitRank, DECK
from tarok.entities.game_state import GameState, Phase, Contract, Trick
from tarok.use_cases.deal import deal


def _make_trick_play_state() -> GameState:
    """Create a minimal GameState in trick_play phase by dealing cards."""
    import random
    rng = random.Random(42)
    state = GameState()
    state = deal(state, rng=rng)
    # Advance to trick play phase
    state.phase = Phase.TRICK_PLAY
    state.contract = Contract.THREE
    state.declarer = 0
    return state


def test_card_tracker_returns_remaining_cards():
    """Card tracker should list cards not in player hand or already played."""
    state = _make_trick_play_state()
    tracker = _build_card_tracker(state)
    assert "remaining_by_group" in tracker
    assert "remaining_count" in tracker
    assert tracker["remaining_count"] > 0


def test_card_tracker_tracks_player_voids():
    """Card tracker should have player info for opponents."""
    state = _make_trick_play_state()
    tracker = _build_card_tracker(state)
    assert "player_info" in tracker
    # Should have info for players 1, 2, 3 (not 0)
    assert "1" in tracker["player_info"]
    assert "2" in tracker["player_info"]
    assert "3" in tracker["player_info"]
    assert "0" not in tracker["player_info"]


def test_state_for_player_includes_tracker_during_trick_play():
    """_state_for_player should include card_tracker during trick play."""
    state = _make_trick_play_state()
    names = ["You", "AI-1", "AI-2", "AI-3"]
    result = _state_for_player(state, 0, names)
    assert result["card_tracker"] is not None


def test_state_for_player_accepts_precomputed_tracker():
    """When a precomputed tracker is passed, it should be used directly."""
    state = _make_trick_play_state()
    names = ["You", "AI-1", "AI-2", "AI-3"]
    fake_tracker = {"remaining_by_group": {}, "remaining_count": 0, "player_info": {}}
    result = _state_for_player(state, 0, names, card_tracker=fake_tracker)
    assert result["card_tracker"] is fake_tracker


@pytest.mark.asyncio
async def test_observer_caches_tracker():
    """WebSocketObserver should cache the card tracker and not rebuild it
    when the trick count hasn't changed."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()

    observer = WebSocketObserver(ws, player_idx=0, player_names=["You", "AI-1", "AI-2", "AI-3"])
    state = _make_trick_play_state()

    # Send first event — cache should be populated
    await observer._send("card_played", {}, state)
    first_tracker = observer._cached_tracker
    assert first_tracker is not None

    # Send again with same trick count — should reuse cache
    await observer._send("card_played", {}, state)
    assert observer._cached_tracker is first_tracker
