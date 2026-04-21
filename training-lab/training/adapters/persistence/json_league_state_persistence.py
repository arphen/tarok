"""Adapter: JSON-based league pool state persistence."""

from __future__ import annotations

import json
from pathlib import Path

from training.entities.league import LeagueOpponent, LeaguePool, LeaguePoolEntry
from training.ports.league_persistence_port import LeagueStatePersistencePort


class JsonLeagueStatePersistence(LeagueStatePersistencePort):
    def save(self, pool: LeaguePool, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learner_elo": pool.learner_elo,
            "entries": [
                {
                    "opponent": {
                        "name": entry.opponent.name,
                        "type": entry.opponent.type,
                        "path": entry.opponent.path,
                        "initial_elo": entry.opponent.initial_elo,
                    },
                    "elo": entry.elo,
                    "games_played": entry.games_played,
                    "learner_outplaces": entry.learner_outplaces,
                    "recent_outplace_rate": entry.recent_outplace_rate,
                    "recent_outplace_samples": entry.recent_outplace_samples,
                }
                for entry in pool.entries
            ],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def restore(self, pool: LeaguePool, path: Path) -> bool:
        if not path.exists():
            return False

        raw = json.loads(path.read_text(encoding="utf-8"))
        pool.learner_elo = float(raw.get("learner_elo", pool.learner_elo))

        restored_entries = [
            LeaguePoolEntry(
                opponent=LeagueOpponent(
                    name=item["opponent"]["name"],
                    type=item["opponent"]["type"],
                    path=item["opponent"].get("path"),
                    initial_elo=float(item["opponent"].get("initial_elo", item.get("elo", 1500.0))),
                ),
                elo=float(item.get("elo", 1500.0)),
                games_played=int(item.get("games_played", 0)),
                learner_outplaces=int(item.get("learner_outplaces", 0)),
                recent_outplace_rate=(
                    float(item["recent_outplace_rate"])
                    if item.get("recent_outplace_rate") is not None
                    else None
                ),
                recent_outplace_samples=int(item.get("recent_outplace_samples", 0)),
            )
            for item in raw.get("entries", [])
        ]

        restored_by_key = {
            _entry_key(entry.opponent): entry
            for entry in restored_entries
        }

        merged_entries: list[LeaguePoolEntry] = []
        for opp in pool.config.opponents:
            restored = restored_by_key.pop(_entry_key(opp), None)
            if restored is None:
                merged_entries.append(LeaguePoolEntry(opponent=opp, elo=opp.initial_elo))
                continue
            merged_entries.append(
                LeaguePoolEntry(
                    opponent=opp,
                    elo=restored.elo,
                    games_played=restored.games_played,
                    learner_outplaces=restored.learner_outplaces,
                    recent_outplace_rate=restored.recent_outplace_rate,
                    recent_outplace_samples=restored.recent_outplace_samples,
                )
            )

        for entry in restored_by_key.values():
            if entry.opponent.type != "nn_checkpoint":
                continue
            snap_path = entry.opponent.path
            if snap_path is None or not Path(snap_path).exists():
                continue
            merged_entries.append(entry)

        pool.entries = merged_entries
        return True


def _entry_key(opponent: LeagueOpponent) -> tuple[str, str, str | None]:
    return opponent.name, opponent.type, opponent.path
