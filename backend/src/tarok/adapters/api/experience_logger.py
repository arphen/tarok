"""Persists supervised-learning data collected from human vs. AI games."""

from __future__ import annotations

import json
import time
from pathlib import Path


_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "human_play"


class HumanPlayExperienceLogger:
    """Writes per-round experience files to data/human_play/."""

    def __init__(self, data_dir: Path | None = None):
        self._dir = Path(data_dir) if data_dir else _DATA_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def write_round(
        self,
        *,
        game_id: str,
        round_num: int,
        player_names: list[str],
        decisions: list[dict],
        scores: list[int],
        contract: str | None,
        declarer: int | None,
        partner: int | None,
    ) -> Path:
        """Persist one round of experience and return the file path."""
        ts = int(time.time() * 1000)
        filename = f"{game_id}_r{round_num:03d}_{ts}.json"
        path = self._dir / filename
        payload = {
            "game_id": game_id,
            "round_num": round_num,
            "player_names": player_names,
            "decisions": decisions,
            "scores": scores,
            "contract": contract,
            "declarer": declarer,
            "partner": partner,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return path
