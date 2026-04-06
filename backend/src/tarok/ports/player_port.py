"""Port for player decision-making.

Implemented by: HumanPlayer (via WebSocket), AIPlayer (RL agent), RandomPlayer.
"""

from __future__ import annotations

from typing import Protocol

from tarok.entities.card import Card
from tarok.entities.game_state import Contract, GameState, Announcement


class PlayerPort(Protocol):
    """Interface for any player (human, AI, random)."""

    @property
    def name(self) -> str: ...

    async def choose_bid(
        self,
        state: GameState,
        player_idx: int,
        legal_bids: list[Contract | None],
    ) -> Contract | None:
        """Choose a bid or pass (None)."""
        ...

    async def choose_king(
        self,
        state: GameState,
        player_idx: int,
        callable_kings: list[Card],
    ) -> Card:
        """Choose which king to call."""
        ...

    async def choose_talon_group(
        self,
        state: GameState,
        player_idx: int,
        talon_groups: list[list[Card]],
    ) -> int:
        """Choose which talon group to pick up (index)."""
        ...

    async def choose_discard(
        self,
        state: GameState,
        player_idx: int,
        must_discard: int,
    ) -> list[Card]:
        """Choose which cards to put down after picking up talon."""
        ...

    async def choose_announcements(
        self,
        state: GameState,
        player_idx: int,
    ) -> list[Announcement]:
        """Choose announcements (can be empty)."""
        ...

    async def choose_announce_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> int:
        """Choose a single announcement/kontra action.

        0=pass, 1=trula, 2=kings, 3=pagat, 4=valat,
        5=kontra_game, 6=kontra_trula, 7=kontra_kings, 8=kontra_pagat, 9=kontra_valat.
        Called repeatedly until the player passes.
        """
        ...

    async def choose_card(
        self,
        state: GameState,
        player_idx: int,
        legal_plays: list[Card],
    ) -> Card:
        """Choose a card to play in the current trick."""
        ...
