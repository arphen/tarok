"""Port: league pool state persistence interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.entities.league import LeaguePool


class LeagueStatePersistencePort(ABC):
    @abstractmethod
    def save(self, pool: LeaguePool, path: Path) -> None: ...

    @abstractmethod
    def restore(self, pool: LeaguePool, path: Path) -> bool: ...
