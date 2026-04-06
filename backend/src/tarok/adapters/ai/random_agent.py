"""Random player adapter — baseline agent that makes random legal moves."""

from __future__ import annotations

import random

from tarok.entities.card import Card, CardType
from tarok.entities.game_state import Announcement, Contract, GameState


class RandomPlayer:
    def __init__(self, name: str = "Random", rng: random.Random | None = None):
        self._name = name
        self._rng = rng or random.Random()

    @property
    def name(self) -> str:
        return self._name

    async def choose_bid(
        self, state: GameState, player_idx: int, legal_bids: list[Contract | None]
    ) -> Contract | None:
        # Bias toward passing slightly
        if None in legal_bids and self._rng.random() < 0.6:
            return None
        return self._rng.choice(legal_bids)

    async def choose_king(
        self, state: GameState, player_idx: int, callable_kings: list[Card]
    ) -> Card:
        return self._rng.choice(callable_kings)

    async def choose_talon_group(
        self, state: GameState, player_idx: int, talon_groups: list[list[Card]]
    ) -> int:
        return self._rng.randint(0, len(talon_groups) - 1)

    async def choose_discard(
        self, state: GameState, player_idx: int, must_discard: int
    ) -> list[Card]:
        hand = state.hands[player_idx]
        # Discard lowest non-king, non-tarok cards
        discardable = [
            c for c in hand
            if not c.is_king and c.card_type != CardType.TAROK
        ]
        if len(discardable) < must_discard:
            # Must discard taroks (allowed when no suit cards)
            discardable = [c for c in hand if not c.is_king]
        discardable.sort(key=lambda c: c.points)
        return discardable[:must_discard]

    async def choose_announcements(
        self, state: GameState, player_idx: int
    ) -> list[Announcement]:
        return []  # Random player never announces

    async def choose_card(
        self, state: GameState, player_idx: int, legal_plays: list[Card]
    ) -> Card:
        return self._rng.choice(legal_plays)
