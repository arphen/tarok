from __future__ import annotations

import json
from typing import Any


class JsonScoreBreakdownParser:
    """Adapter that parses Rust score payloads encoded as JSON."""

    def parse(self, payload: str) -> dict[str, Any]:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Score breakdown payload must decode to an object")
        return data
