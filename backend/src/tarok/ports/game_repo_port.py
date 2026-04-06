"""Port for game state persistence."""

from __future__ import annotations

from typing import Protocol

from tarok.entities.game_state import GameState


class GameRepoPort(Protocol):
    async def save(self, game_id: str, state: GameState) -> None: ...
    async def load(self, game_id: str) -> GameState | None: ...
    async def list_games(self) -> list[str]: ...
