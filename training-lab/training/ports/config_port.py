"""Port: config loading interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConfigPort(ABC):
    @abstractmethod
    def load(self, path: str) -> dict[str, Any]:
        """Load config from file → raw dict."""
