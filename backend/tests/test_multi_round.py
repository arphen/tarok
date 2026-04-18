"""Tests for multi-round games, scoreboard data, and card tracking."""

import random
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from tarok.adapters.api.server import app
from tarok.adapters.api.ws_observer import _build_card_tracker, _state_for_player
from tarok.entities import (
    Card,
    Suit,
    SuitRank,
    DECK,
    GameState,
    Phase,
    Trick,
    tarok,
    suit_card,
)
from tarok.use_cases.game_loop import RustGameLoop as GameLoop, NullObserver
from tarok.adapters.players.stockskis_player import StockskisPlayer


# ---- Fixtures ----


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---- API tests: num_rounds ----


async def test_new_game_with_num_rounds(client):
    """POST /api/game/new with num_rounds creates a game with round config."""
    resp = await client.post(
        "/api/game/new",
        json={
            "opponents": ["random", "random", "random"],
            "num_rounds": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "game_id" in data


async def test_new_game_default_rounds(client):
    """Default num_rounds should be 1."""
    resp = await client.post(
        "/api/game/new",
        json={
            "opponents": ["random", "random", "random"],
        },
    )
    assert resp.status_code == 200


async def test_new_game_single_round(client):
    """Explicit single round works."""
    resp = await client.post(
        "/api/game/new",
        json={
            "num_rounds": 1,
        },
    )
    assert resp.status_code == 200


# ---- Card tracker tests ----


def _make_trick(
    cards_with_players: list[tuple[int, Card]], trump_value: int | None = None
) -> Trick:
    """Helper to create a trick from cards."""
    trick = Trick()
    for p, c in cards_with_players:
        trick.cards.append((p, c))
    return trick


def test_card_tracker_initial():
    """At the start of trick play, tracker shows all non-hand cards as remaining."""
    state = GameState(dealer=0)
    state.phase = Phase.TRICK_PLAY
    # Give player 0 some cards
    rng = random.Random(42)
    deck_list = list(DECK)
    rng.shuffle(deck_list)
    state.hands = [deck_list[:12], deck_list[12:24], deck_list[24:36], deck_list[36:48]]
    state.talon = list(deck_list[48:])

    tracker = _build_card_tracker(state)

    # Remaining should be 54 - 12 (our hand) = 42
    assert tracker["remaining_count"] == 42
    # All groups should be present
    assert "taroks" in tracker["remaining_by_group"]
    assert "hearts" in tracker["remaining_by_group"]
    # No player info yet (no tricks played)
    for p in ["1", "2", "3"]:
        info = tracker["player_info"][p]
        assert info["void_suits"] == []
        assert info["taroks_played_count"] == 0


def test_card_tracker_after_trick():
    """After a trick, played cards are removed from remaining."""
    state = GameState(dealer=0)
    state.phase = Phase.TRICK_PLAY
    rng = random.Random(42)
    deck_list = list(DECK)
    rng.shuffle(deck_list)
    state.hands = [deck_list[:12], deck_list[12:24], deck_list[24:36], deck_list[36:48]]

    # Play a trick with known cards
    hearts_king = suit_card(Suit.HEARTS, SuitRank.KING)
    tarok_1 = tarok(1)
    tarok_2 = tarok(2)
    tarok_21 = tarok(21)

    trick = Trick(lead_player=0)
    trick.cards = [(0, hearts_king), (1, tarok_1), (2, tarok_2), (3, tarok_21)]
    state.tricks = [trick]

    tracker = _build_card_tracker(state)

    # Played cards should be gone from remaining
    remaining_labels = [
        c["label"] for group in tracker["remaining_by_group"].values() for c in group
    ]
    assert hearts_king.label not in remaining_labels
    assert tarok_1.label not in remaining_labels

    # Player 1,2,3 played tarok on hearts lead → they're void in hearts
    for p in ["1", "2", "3"]:
        assert "hearts" in tracker["player_info"][p]["void_suits"]

    # Tarok stats
    assert tracker["player_info"]["1"]["taroks_played_count"] == 1
    assert tracker["player_info"]["1"]["highest_tarok"] == 1
    assert tracker["player_info"]["3"]["highest_tarok"] == 21


def test_card_tracker_not_shown_before_trick_play():
    """Card tracker should be None outside trick play."""
    state = GameState(dealer=0)
    state.phase = Phase.BIDDING
    state.hands = [[], [], [], []]

    result = _state_for_player(state, 0, ["P0", "P1", "P2", "P3"])
    assert result["card_tracker"] is None


def test_card_tracker_shown_during_trick_play():
    """Card tracker should be present during trick play."""
    state = GameState(dealer=0)
    state.phase = Phase.TRICK_PLAY
    rng = random.Random(42)
    deck_list = list(DECK)
    rng.shuffle(deck_list)
    state.hands = [deck_list[:12], deck_list[12:24], deck_list[24:36], deck_list[36:48]]

    result = _state_for_player(state, 0, ["P0", "P1", "P2", "P3"])
    assert result["card_tracker"] is not None
    assert "remaining_by_group" in result["card_tracker"]
    assert "player_info" in result["card_tracker"]


def test_card_tracker_put_down_excluded():
    """Cards in put_down should not appear in remaining."""
    state = GameState(dealer=0)
    state.phase = Phase.TRICK_PLAY
    rng = random.Random(42)
    deck_list = list(DECK)
    rng.shuffle(deck_list)
    state.hands = [deck_list[:12], deck_list[12:24], deck_list[24:36], deck_list[36:48]]

    # Put down some cards
    put_card = deck_list[48]
    state.put_down = [put_card]

    tracker = _build_card_tracker(state)
    remaining_labels = [
        c["label"] for group in tracker["remaining_by_group"].values() for c in group
    ]
    assert put_card.label not in remaining_labels


# ---- Match info in state tests ----


def test_state_includes_match_info():
    """When match info is set, state includes it."""
    state = GameState(dealer=0)
    state.phase = Phase.TRICK_PLAY
    state.hands = [[], [], [], []]
    match_info = {
        "round_num": 2,
        "total_rounds": 5,
        "cumulative_scores": {"0": 10, "1": -5, "2": 3, "3": -8},
        "caller_counts": {"0": 1, "1": 0, "2": 1, "3": 0},
        "called_counts": {"0": 0, "1": 1, "2": 0, "3": 0},
        "round_history": [],
    }

    result = _state_for_player(state, 0, ["P0", "P1", "P2", "P3"], match_info)
    assert result["match_info"] is not None
    assert result["match_info"]["round_num"] == 2
    assert result["match_info"]["total_rounds"] == 5


def test_state_match_info_none_by_default():
    """Without match info, state has match_info=None."""
    state = GameState(dealer=0)
    state.phase = Phase.BIDDING
    state.hands = [[], [], [], []]

    result = _state_for_player(state, 0, ["P0", "P1", "P2", "P3"])
    assert result["match_info"] is None


# ---- Multi-round game loop test ----


class RecordingObserver(NullObserver):
    """Observer that records events for testing."""

    def __init__(self):
        self.events: list[str] = []
        self.game_end_scores: list[dict] = []

    async def on_game_start(self, state):
        self.events.append("game_start")

    async def on_game_end(self, scores, state, breakdown=None):
        self.events.append("game_end")
        self.game_end_scores.append(dict(scores))


async def test_single_round_game_loop():
    """A single round game completes normally."""

    players = [StockskisPlayer(variant="m6", name=f"P{i}") for i in range(4)]
    obs = RecordingObserver()
    loop = GameLoop(players, observer=cast(Any, obs))
    state, scores = await loop.run(dealer=0)

    assert state.phase == Phase.FINISHED
    assert len(scores) == 4
    assert "game_start" in obs.events
    assert "game_end" in obs.events


async def test_multi_round_cumulative_scores():
    """Running multiple rounds accumulates scores correctly."""

    players = [StockskisPlayer(variant="m6", name=f"P{i}") for i in range(4)]
    cumulative = {i: 0 for i in range(4)}
    num_rounds = 3

    for r in range(num_rounds):
        loop = GameLoop(players)
        state, scores = await loop.run(dealer=r % 4)
        for p, s in scores.items():
            cumulative[p] += s

    # After 3 rounds, we should have cumulative scores
    assert len(cumulative) == 4
    # Scores should generally not all be zero after 3 rounds
    total = sum(abs(v) for v in cumulative.values())
    assert total > 0
