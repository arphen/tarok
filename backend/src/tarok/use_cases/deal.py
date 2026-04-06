"""Deal use case — shuffle and distribute 54 cards."""

from __future__ import annotations

import random

from tarok.entities.card import DECK, Card, CardType
from tarok.entities.game_state import GameState, Phase

CARDS_PER_PLAYER = 12
TALON_SIZE = 6


def deal(state: GameState, rng: random.Random | None = None) -> GameState:
    """Deal cards to 4 players + talon. Returns new state."""
    assert state.phase == Phase.DEALING

    rng = rng or random.Random()
    cards = list(DECK)
    rng.shuffle(cards)

    hands: list[list[Card]] = []
    idx = 0
    for _ in range(state.num_players):
        hands.append(sorted(cards[idx : idx + CARDS_PER_PLAYER], key=lambda c: c.sort_key))
        idx += CARDS_PER_PLAYER

    talon = cards[idx : idx + TALON_SIZE]
    assert idx + TALON_SIZE == len(cards)

    state.hands = hands
    state.talon = talon
    state.initial_tarok_counts = [
        sum(1 for c in h if c.card_type == CardType.TAROK) for h in hands
    ]
    state.phase = Phase.BIDDING
    # First bidder is the player after the dealer (forehand)
    state.current_bidder = (state.dealer + 1) % state.num_players
    state.current_player = state.current_bidder
    return state
