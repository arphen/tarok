"""League play domain types."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Frozen config types (from YAML)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeagueOpponent:
    name: str
    type: Literal["nn_checkpoint", "bot_v1", "bot_v3", "bot_v5", "bot_v6", "bot_m6"]
    path: str | None = None  # required when type == "nn_checkpoint"
    initial_elo: float = 1500.0

    def seat_token(self) -> str:
        """The string passed to run_self_play's seat_config for this opponent."""
        if self.type == "nn_checkpoint":
            if self.path is None:
                raise ValueError(f"LeagueOpponent '{self.name}' is nn_checkpoint but has no path")
            return self.path
        return self.type


@dataclass(frozen=True)
class LeagueConfig:
    enabled: bool = False
    opponents: tuple[LeagueOpponent, ...] = ()
    min_nn_per_game: int = 1  # learner seat 0 counts as 1
    sampling: Literal["uniform", "pfsp", "hardest", "matchmaking"] = "pfsp"
    pfsp_alpha: float = 1.5
    snapshot_interval: int = 5  # save snapshot every N iterations
    snapshot_elo_delta: float = 50.0
    max_active_snapshots: int = 3


# ---------------------------------------------------------------------------
# Mutable runtime types
# ---------------------------------------------------------------------------

@dataclass
class LeaguePoolEntry:
    opponent: LeagueOpponent
    elo: float = 1500.0
    games_played: int = 0
    learner_outplaces: int = 0  # games where learner scored above this opponent

    @property
    def outplace_rate(self) -> float:
        """Fraction of games where the learner placed above this opponent."""
        if self.games_played == 0:
            return 0.5
        return self.learner_outplaces / self.games_played


@dataclass
class LeaguePool:
    config: LeagueConfig
    entries: list[LeaguePoolEntry] = field(default_factory=list)
    learner_elo: float = 800.0

    def __post_init__(self) -> None:
        for opp in self.config.opponents:
            self.entries.append(LeaguePoolEntry(opponent=opp, elo=opp.initial_elo))

    def add_snapshot(self, name: str, path: str) -> None:
        """Register an auto-generated checkpoint snapshot as a new pool entry."""
        opp = LeagueOpponent(name=name, type="nn_checkpoint", path=path)
        self.entries.append(LeaguePoolEntry(opponent=opp, elo=self.learner_elo))

    def save(self, state_path: Path) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learner_elo": self.learner_elo,
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
                }
                for entry in self.entries
            ],
        }
        state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def restore(self, state_path: Path) -> bool:
        if not state_path.exists():
            return False

        raw = json.loads(state_path.read_text(encoding="utf-8"))
        self.learner_elo = float(raw.get("learner_elo", self.learner_elo))

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
            )
            for item in raw.get("entries", [])
        ]

        restored_by_key = {
            self._entry_key(entry.opponent): entry
            for entry in restored_entries
        }

        merged_entries: list[LeaguePoolEntry] = []
        for opp in self.config.opponents:
            restored = restored_by_key.pop(self._entry_key(opp), None)
            if restored is None:
                merged_entries.append(LeaguePoolEntry(opponent=opp, elo=opp.initial_elo))
                continue
            merged_entries.append(
                LeaguePoolEntry(
                    opponent=opp,
                    elo=restored.elo,
                    games_played=restored.games_played,
                    learner_outplaces=restored.learner_outplaces,
                )
            )

        for entry in restored_by_key.values():
            if entry.opponent.type != "nn_checkpoint":
                continue
            snap_path = entry.opponent.path
            if snap_path is None or not Path(snap_path).exists():
                continue
            merged_entries.append(entry)

        self.entries = merged_entries
        return True

    @staticmethod
    def _entry_key(opponent: LeagueOpponent) -> tuple[str, str, str | None]:
        return opponent.name, opponent.type, opponent.path

    def sampling_weights(self) -> list[float]:
        """Return a weight per entry for weighted random sampling."""
        if not self.entries:
            return []
        sampling = self.config.sampling
        if sampling == "uniform":
            return [1.0] * len(self.entries)
        if sampling == "hardest":
            # Only the entry with the highest Elo gets all the weight
            max_elo = max(e.elo for e in self.entries)
            return [1.0 if e.elo == max_elo else 0.0 for e in self.entries]
        if sampling == "matchmaking":
            # Gaussian window around learner Elo prefers similarly rated opponents.
            window = 200.0
            raw = []
            for e in self.entries:
                distance = e.elo - self.learner_elo
                raw.append(math.exp(-((distance ** 2) / (2 * (window ** 2)))))
            total = sum(raw)
            if total == 0:
                return [1.0 / len(self.entries)] * len(self.entries)
            return [w / total for w in raw]
        # pfsp: weight_i = exp(alpha * (elo_i - 1500) / 400)
        # Higher alpha concentrates on higher-Elo (harder) opponents.
        alpha = self.config.pfsp_alpha
        raw = [math.exp(alpha * (e.elo - 1500.0) / 400.0) for e in self.entries]
        total = sum(raw)
        if total == 0:
            return [1.0 / len(self.entries)] * len(self.entries)
        return [w / total for w in raw]
