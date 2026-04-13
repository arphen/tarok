"""Opponent Pool — modular self-play opponent selection with statistics.

Provides a unified interface for mixing different opponent types during
self-play training: neural network models (HoF, FSP bank) and heuristic bots
(StockSkis v5). Each opponent type implements
the SelfPlayOpponent protocol and tracks per-game statistics.

Usage in trainer:
    pool = OpponentPool()
    pool.add(HoFOpponent(hof_dir))
    pool.add(StockSkisOpponent(version=5))
    pool.add(FSPOpponent(network_bank))
    pool.add(SelfPlayPureOpponent())

    for game in session:
        opponent = pool.choose()
        agents[1:] = opponent.make_players(shared_network)
        ...play game...
        opponent.record_result(raw_score, won, contract, place)

    stats = pool.stats()
"""

from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from tarok.adapters.ai.network import TarokNet

log = logging.getLogger(__name__)


@dataclass
class OpponentGameResult:
    """Result of a single game against an opponent."""
    raw_score: int = 0
    won: bool = False
    contract_name: str = ""
    place: float = 4.0
    declarer_p0: bool = False
    bid: bool = False


@dataclass
class OpponentStats:
    """Aggregate stats for an opponent type across games."""
    games: int = 0
    wins: int = 0
    total_score: int = 0
    total_place: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.games, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.games, 1)

    @property
    def avg_place(self) -> float:
        return self.total_place / max(self.games, 1)

    def record(self, result: OpponentGameResult) -> None:
        self.games += 1
        if result.won:
            self.wins += 1
        self.total_score += result.raw_score
        self.total_place += result.place

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "avg_score": round(self.avg_score, 2),
            "avg_place": round(self.avg_place, 2),
        }


@runtime_checkable
class SelfPlayOpponent(Protocol):
    """Common interface for all self-play opponent types.

    Implementations wrap around PlayerPort-compatible players but add
    weight-loading, statistics tracking, and opponent-type metadata.
    """

    @property
    def name(self) -> str:
        """Human-readable label for this opponent type (e.g. 'HoF', 'v5')."""
        ...

    @property
    def weight(self) -> float:
        """Sampling weight (higher = more likely to be chosen)."""
        ...

    @weight.setter
    def weight(self, value: float) -> None: ...

    def is_available(self) -> bool:
        """Whether this opponent can currently supply players."""
        ...

    def make_players(self, shared_network: TarokNet | None = None) -> list:
        """Create 3 PlayerPort-compatible players for seats 1-3.

        For NN opponents, shared_network is provided so they can clone
        architecture. For heuristic bots, it's ignored.
        """
        ...

    def record_result(self, result: OpponentGameResult) -> None:
        """Record the outcome of a game played against this opponent."""
        ...

    @property
    def stats(self) -> OpponentStats:
        """Aggregate stats across all games."""
        ...

    def requires_external_experience(self) -> bool:
        """If True, only agent 0 records experiences (opponents are external).

        Self-play (shared network) returns False so all 4 agents contribute.
        """
        ...


class PureSelfPlayOpponent:
    """Self-play using the same shared network (no opponent swap needed)."""

    def __init__(self, weight: float = 1.0):
        self._weight = weight
        self._stats = OpponentStats()

    @property
    def name(self) -> str:
        return "self-play"

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value

    def is_available(self) -> bool:
        return True

    def make_players(self, shared_network: TarokNet | None = None) -> list:
        # No swap needed — trainer keeps agents using shared network
        return []

    def record_result(self, result: OpponentGameResult) -> None:
        self._stats.record(result)

    @property
    def stats(self) -> OpponentStats:
        return self._stats

    def requires_external_experience(self) -> bool:
        return False


class FSPOpponent:
    """Fictitious Self-Play — random historical snapshot from the network bank."""

    def __init__(self, network_bank, weight: float = 0.3):
        self._bank = network_bank
        self._weight = weight
        self._stats = OpponentStats()
        self._opponent_network: TarokNet | None = None
        self._last_snapshot_id: str = ""
        self._per_snapshot_stats: dict[str, OpponentStats] = {}

    @property
    def name(self) -> str:
        return "fsp"

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value

    def is_available(self) -> bool:
        return self._bank.is_ready

    def make_players(self, shared_network: TarokNet | None = None) -> list:
        from tarok.adapters.ai.agent import RLAgent

        snap, snapshot_id = self._bank.sample()
        if snap is None or shared_network is None:
            return []

        self._last_snapshot_id = snapshot_id

        if self._opponent_network is None:
            hidden_size = shared_network.shared[0].out_features
            self._opponent_network = TarokNet(
                hidden_size=hidden_size,
                oracle_critic=shared_network.oracle_critic_enabled,
            )

        self._opponent_network.load_state_dict(snap)
        players = []
        for i in range(3):
            agent = RLAgent(name=f"FSP-{i}", hidden_size=shared_network.shared[0].out_features)
            agent.network = self._opponent_network
            agent.set_training(False)
            players.append(agent)
        return players

    def record_result(self, result: OpponentGameResult) -> None:
        self._stats.record(result)
        if self._last_snapshot_id:
            if self._last_snapshot_id not in self._per_snapshot_stats:
                self._per_snapshot_stats[self._last_snapshot_id] = OpponentStats()
            self._per_snapshot_stats[self._last_snapshot_id].record(result)

    @property
    def stats(self) -> OpponentStats:
        return self._stats

    @property
    def per_instance_stats(self) -> dict[str, dict]:
        return {sid: s.to_dict() for sid, s in self._per_snapshot_stats.items()}

    def requires_external_experience(self) -> bool:
        return True


class StockSkisOpponent:
    """Heuristic StockSkis v5 bot as opponent."""

    def __init__(self, version: int = 5, strength: float = 1.0, weight: float = 0.5):
        self._version = version
        self._strength = strength
        self._weight = weight
        self._stats = OpponentStats()
        self._players: list | None = None

    @property
    def name(self) -> str:
        return f"stockskis-v{self._version}"

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value

    def is_available(self) -> bool:
        return True

    def make_players(self, shared_network: TarokNet | None = None) -> list:
        if self._players is None:
            from tarok.adapters.ai.stockskis_v5 import StockSkisPlayerV5

            self._players = [
                StockSkisPlayerV5(name=f"StockŠkis-v5-{i}", strength=self._strength)
                for i in range(3)
            ]
        return self._players

    def record_result(self, result: OpponentGameResult) -> None:
        self._stats.record(result)

    @property
    def stats(self) -> OpponentStats:
        return self._stats

    def requires_external_experience(self) -> bool:
        return True


class HoFOpponent:
    """Hall of Fame model loaded as a frozen opponent.

    Picks a random HoF model each game. Loads its weights into an
    opponent network and creates 3 agents using those frozen weights.
    """

    def __init__(self, hof_dir: str | Path, weight: float = 0.2):
        self._hof_dir = Path(hof_dir)
        self._weight = weight
        self._stats = OpponentStats()
        self._opponent_network: TarokNet | None = None
        self._cached_models: list[Path] = []
        self._cache_time: float = 0.0
        self._last_used: str = ""
        self._per_model_stats: dict[str, OpponentStats] = {}

    @property
    def name(self) -> str:
        return "hof"

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value

    def _refresh_cache(self) -> None:
        """Scan HoF dir (+ pinned/ subdir) for .pt files, cache for 60s."""
        now = time.time()
        if now - self._cache_time < 60 and self._cached_models:
            return
        models: list[Path] = []
        if self._hof_dir.exists():
            models.extend(sorted(self._hof_dir.glob("hof_*.pt")))
        pinned_dir = self._hof_dir / "pinned"
        if pinned_dir.exists():
            models.extend(sorted(pinned_dir.glob("hof_*.pt")))
        self._cached_models = models
        self._cache_time = now

    def is_available(self) -> bool:
        self._refresh_cache()
        return len(self._cached_models) > 0

    def make_players(self, shared_network: TarokNet | None = None) -> list:
        from tarok.adapters.ai.agent import RLAgent

        self._refresh_cache()
        if not self._cached_models or shared_network is None:
            return []

        # Pick a random HoF model
        chosen_path = random.choice(self._cached_models)
        self._last_used = chosen_path.name

        try:
            data = torch.load(chosen_path, map_location="cpu", weights_only=False)
        except Exception:
            log.warning("Failed to load HoF model %s", chosen_path.name)
            return []

        state_dict = data.get("model_state_dict")
        if state_dict is None:
            return []

        hidden_size = data.get("hidden_size", shared_network.shared[0].out_features)

        if self._opponent_network is None or self._opponent_network.shared[0].out_features != hidden_size:
            self._opponent_network = TarokNet(hidden_size=hidden_size)

        try:
            self._opponent_network.load_state_dict(state_dict)
        except Exception:
            log.warning("Failed to load state dict from HoF model %s", chosen_path.name)
            return []

        players = []
        for i in range(3):
            agent = RLAgent(name=f"HoF-{i}", hidden_size=hidden_size)
            agent.network = self._opponent_network
            agent.set_training(False)
            players.append(agent)
        return players

    def record_result(self, result: OpponentGameResult) -> None:
        self._stats.record(result)
        if self._last_used:
            if self._last_used not in self._per_model_stats:
                self._per_model_stats[self._last_used] = OpponentStats()
            self._per_model_stats[self._last_used].record(result)

    @property
    def per_instance_stats(self) -> dict[str, dict]:
        return {model: s.to_dict() for model, s in self._per_model_stats.items()}

    @property
    def stats(self) -> OpponentStats:
        return self._stats

    def requires_external_experience(self) -> bool:
        return True


class OpponentPool:
    """Manages a weighted pool of opponent types for self-play training.

    Opponents are sampled proportionally to their weights. Only available
    opponents are considered for sampling. If no opponents are available,
    falls back to pure self-play.
    """

    def __init__(self, rng: random.Random | None = None):
        self._opponents: list[SelfPlayOpponent] = []
        self._rng = rng or random.Random()
        self._session_stats: dict[str, OpponentStats] = {}

    def add(self, opponent: SelfPlayOpponent) -> None:
        self._opponents.append(opponent)

    def remove(self, name: str) -> None:
        self._opponents = [o for o in self._opponents if o.name != name]

    @property
    def opponents(self) -> list[SelfPlayOpponent]:
        return list(self._opponents)

    def begin_session(self) -> None:
        """Start a new session — reset per-session stats."""
        self._session_stats = {o.name: OpponentStats() for o in self._opponents}

    def record_session_result(self, opponent_name: str, result: OpponentGameResult) -> None:
        """Record a game result for a specific opponent within the current session."""
        if opponent_name in self._session_stats:
            self._session_stats[opponent_name].record(result)
        # Also record to the aggregate opponent stats
        for opp in self._opponents:
            if opp.name == opponent_name:
                opp.record_result(result)
                return

    def session_stats_dict(self) -> dict[str, dict]:
        """Return per-opponent-type statistics for the current session only."""
        return {name: s.to_dict() for name, s in self._session_stats.items() if s.games > 0}

    def choose(self) -> SelfPlayOpponent:
        """Sample an opponent proportional to weights (only available ones)."""
        available = [o for o in self._opponents if o.is_available()]
        if not available:
            # Fallback: pure self-play if nothing else available
            fallback = PureSelfPlayOpponent()
            return fallback

        total = sum(o.weight for o in available)
        if total <= 0:
            return available[0]

        r = self._rng.random() * total
        cumulative = 0.0
        for opp in available:
            cumulative += opp.weight
            if r <= cumulative:
                return opp
        return available[-1]

    def stats_dict(self) -> dict[str, dict]:
        """Return per-opponent-type statistics with per-instance breakdowns."""
        result = {}
        for o in self._opponents:
            entry = o.stats.to_dict()
            if hasattr(o, 'per_instance_stats'):
                instances = o.per_instance_stats
                if instances:
                    entry["instances"] = instances
            result[o.name] = entry
        return result

    def summary(self) -> str:
        """Human-readable summary of the pool configuration."""
        lines = []
        for o in self._opponents:
            avail = "ready" if o.is_available() else "not ready"
            lines.append(f"  {o.name}: weight={o.weight:.2f} ({avail}) games={o.stats.games}")
        return "Opponent Pool:\n" + "\n".join(lines)
