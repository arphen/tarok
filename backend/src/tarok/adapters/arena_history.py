"""Persistence for arena run history (data/arena_results.json)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

_HISTORY_PATH = Path(__file__).resolve().parents[4] / "data" / "arena_results.json"


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
    req_agents: list[dict],
    total_games: int,
    session_size: int,
    payload: dict,
) -> None:
    analytics = payload.get("analytics") or {}
    checkpoints = sorted(
        {
            str(a.get("checkpoint", "")).strip()
            for a in req_agents
            if str(a.get("type", "")).strip().lower() == "rl"
            and str(a.get("checkpoint", "")).strip()
        }
    )
    run = {
        "run_id": f"arena-{time.time_ns()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": payload.get("status", "done"),
        "games_done": int(payload.get("games_done", 0)),
        "total_games": int(total_games),
        "session_size": int(session_size),
        "agents": [
            {
                "name": str(a.get("name", "")),
                "type": str(a.get("type", "")),
                "checkpoint": str(a.get("checkpoint", "")),
            }
            for a in req_agents
        ],
        "checkpoints": checkpoints,
        "analytics": analytics,
    }
    history = load()
    runs = history.get("runs", [])
    runs.append(run)
    if len(runs) > 200:
        runs = runs[-200:]
    history["runs"] = runs
    save(history)
