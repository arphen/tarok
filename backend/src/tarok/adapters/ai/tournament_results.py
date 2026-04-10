"""Loader for tournament result data.

Usage::

    from tarok.adapters.ai.tournament_results import load_results, top_models

    results = load_results()
    best = top_models(results, n=3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "tournament_results.json"


def load_results(path: Path | None = None) -> dict[str, Any]:
    """Load tournament results JSON."""
    p = path or _DATA_PATH
    with open(p) as f:
        return json.load(f)


def top_models(results: dict[str, Any], n: int = 3) -> list[str]:
    """Return the *n* best model names by combined average placement.

    Only models that appear in **all** tournaments are ranked.
    """
    model_placements: dict[str, list[float]] = {}
    for t in results["tournaments"]:
        for s in t["standings"]:
            model_placements.setdefault(s["model"], []).append(s["avg_placement"])

    num_tournaments = len(results["tournaments"])
    combined = {
        m: sum(avgs) / len(avgs)
        for m, avgs in model_placements.items()
        if len(avgs) == num_tournaments
    }
    ranked = sorted(combined, key=lambda m: combined[m])
    return ranked[:n]
