"""Persistence for duplicate-arena run history (data/duplicate_arena_results.json)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HISTORY_PATH = Path(__file__).resolve().parents[4] / "data" / "duplicate_arena_results.json"


def load() -> dict:
    if not _HISTORY_PATH.exists():
        return {"runs": []}
    try:
        with open(_HISTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"runs": []}
        runs = data.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        return {"runs": runs}
    except Exception:
        return {"runs": []}


def save(data: dict) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _HISTORY_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(_HISTORY_PATH)


def persist_run(
    *,
    challenger: str,
    defender: str,
    boards: int,
    seed: int,
    pairing: str,
    status: str,
    result: dict[str, Any] | None,
    error: str | None = None,
) -> dict:
    run = {
        "run_id": f"duparena-{time.time_ns()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "challenger": challenger,
        "defender": defender,
        "boards": int(boards),
        "seed": int(seed),
        "pairing": pairing,
        "status": status,
        "result": result,
        "error": error,
    }
    history = load()
    runs = history.get("runs", [])
    runs.append(run)
    if len(runs) > 200:
        runs = runs[-200:]
    history["runs"] = runs
    save(history)
    return run
