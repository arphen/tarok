"""StockSkis v5 — currently aliased to v4.

This version exists so callers can request ``stockskis_v5`` consistently.
If you later add true v5 heuristics, keep the exported class name
``StockSkisPlayerV5`` stable so API consumers don't need changes.
"""

from __future__ import annotations

from tarok.adapters.ai.stockskis_v4 import StockSkisPlayerV4


class StockSkisPlayerV5(StockSkisPlayerV4):
    """Heuristic bot v5 (currently same logic as v4)."""

    def __init__(
        self,
        name: str = "StockŠkis-v5",
        strength: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(name=name, strength=strength, seed=seed)

