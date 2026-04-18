"""Human player adapter — bridges WebSocket input to PlayerPort."""

from __future__ import annotations

import asyncio

from tarok.entities import Announcement, Card, Contract, GameState


class HumanPlayer:
    """Player controlled via WebSocket. Waits for human input."""

    def __init__(self, name: str = "Human"):
        self._name = name
        self._pending_action: asyncio.Future | None = None
        self._queued_action: object | None = None
        self._has_queued: bool = False
        self._expected_action: str | None = None

    @property
    def name(self) -> str:
        return self._name

    def submit_action(self, action, *, action_type: str | None = None) -> None:
        if self._expected_action and action_type and action_type != self._expected_action:
            return
        if self._pending_action and not self._pending_action.done():
            self._pending_action.set_result(action)
        else:
            self._queued_action = action
            self._has_queued = True

    def drain_queue(self) -> None:
        self._queued_action = None
        self._has_queued = False

    async def _wait_for_input(self, expected: str):
        self._expected_action = expected
        if self._has_queued:
            result = self._queued_action
            self._queued_action = None
            self._has_queued = False
            self._expected_action = None
            return result
        loop = asyncio.get_event_loop()
        self._pending_action = loop.create_future()
        result = await self._pending_action
        self._pending_action = None
        self._expected_action = None
        return result

    async def choose_bid(
        self, state: GameState, player_idx: int, legal_bids: list[Contract | None]
    ) -> Contract | None:
        return await self._wait_for_input("bid")

    async def choose_king(
        self, state: GameState, player_idx: int, callable_kings: list[Card]
    ) -> Card:
        return await self._wait_for_input("king")

    async def choose_talon_group(
        self, state: GameState, player_idx: int, talon_groups: list[list[Card]]
    ) -> int:
        return await self._wait_for_input("talon")

    async def choose_discard(
        self, state: GameState, player_idx: int, must_discard: int
    ) -> list[Card]:
        return await self._wait_for_input("discard")

    async def choose_announcements(self, state: GameState, player_idx: int) -> list[Announcement]:
        return []

    async def choose_card(self, state: GameState, player_idx: int, legal_plays: list[Card]) -> Card:
        return await self._wait_for_input("card")
