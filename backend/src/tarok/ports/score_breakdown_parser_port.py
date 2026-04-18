from __future__ import annotations

from typing import Any, Protocol


class ScoreBreakdownParserPort(Protocol):
    """Parses engine score payload into a mapping for use-case consumption."""

    def parse(self, payload: str) -> dict[str, Any]: ...
