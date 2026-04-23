"""Adapters implementing :class:`DuplicateShadowSourcePort`.

Three implementations cover the three useful sources of a frozen shadow
policy for duplicate-RL:

* :class:`PreviousIterationShadowSource` — the learner one iteration ago.
* :class:`LeaguePoolShadowSource` — a random ``nn_checkpoint`` entry from
  the live league pool (Gaussian-matchmaking-weighted).
* :class:`BestSnapshotShadowSource` — the highest-Elo ``nn_checkpoint``
  entry in the league pool (the "best ghost" seen so far).

All three gracefully fall back to the current learner's TorchScript path
when they have no better option (iteration 0, empty pool, pool with only
bot entries). This makes them safe defaults even mid-run, before the
league has accumulated snapshots.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from training.ports.duplicate_shadow_source_port import DuplicateShadowSourcePort

if TYPE_CHECKING:
    from training.entities.league import LeaguePool, LeaguePoolEntry


# ----------------------------------------------------------------------


class PreviousIterationShadowSource(DuplicateShadowSourcePort):
    """Frozen copy of the learner from the previous iteration.

    Relies on the fact that the learner's TorchScript path is a single
    fixed file overwritten at the *end* of each iteration (by
    :class:`ExportModel`). During duplicate self-play — which runs
    *before* that export — the file on disk still contains the previous
    iteration's weights, so the Rust engine loading ``learner_ts_path``
    for the shadow seat reads the correct frozen snapshot.

    On iteration 0 (before any training has happened) both seats load
    the same freshly-initialised weights — the conventional
    "self-duplicate bootstrap".
    """

    def resolve(
        self,
        *,
        iteration: int,
        learner_ts_path: str,
        pool: "LeaguePool | None",
    ) -> str:
        del iteration, pool
        return learner_ts_path


# ----------------------------------------------------------------------


def _nn_checkpoint_entries(pool: "LeaguePool | None") -> list["LeaguePoolEntry"]:
    """Return only entries whose opponent is a usable ``nn_checkpoint``."""
    if pool is None:
        return []
    out = []
    for entry in pool.entries:
        opp = entry.opponent
        if opp.type != "nn_checkpoint":
            continue
        if opp.path is None:
            continue
        out.append(entry)
    return out


class LeaguePoolShadowSource(DuplicateShadowSourcePort):
    """Random ``nn_checkpoint`` entry from the league pool.

    Uses a Gaussian window around the learner's current Elo (``window=200``)
    to prefer similarly-rated snapshots — identical to the
    ``matchmaking`` league sampling, so duplicate pods see the same kind
    of opponent the arena benchmark trains against.

    Adapters own their own RNG so pod schedules stay reproducible across
    runs (seeded from :class:`DuplicateConfig.rng_seed`).
    """

    _WINDOW = 200.0

    def __init__(self, rng_seed: int = 0) -> None:
        self._rng = random.Random(rng_seed)

    def resolve(
        self,
        *,
        iteration: int,
        learner_ts_path: str,
        pool: "LeaguePool | None",
    ) -> str:
        candidates = _nn_checkpoint_entries(pool)
        if not candidates or pool is None:
            return learner_ts_path

        learner_elo = pool.learner_elo
        raw = [
            math.exp(-(((e.elo - learner_elo) ** 2) / (2 * (self._WINDOW ** 2))))
            for e in candidates
        ]
        total = sum(raw)
        if total <= 0.0:
            chosen = self._rng.choice(candidates)
        else:
            weights = [w / total for w in raw]
            chosen = self._rng.choices(candidates, weights=weights, k=1)[0]

        # ``_nn_checkpoint_entries`` already filtered out None paths.
        assert chosen.opponent.path is not None
        return chosen.opponent.path


# ----------------------------------------------------------------------


class BestSnapshotShadowSource(DuplicateShadowSourcePort):
    """Highest-Elo ``nn_checkpoint`` entry in the league pool.

    Picks the "best ghost" the learner has produced so far. Ties on Elo
    are broken by most games played (oldest-stable snapshot wins),
    ensuring the selection is deterministic.
    """

    def resolve(
        self,
        *,
        iteration: int,
        learner_ts_path: str,
        pool: "LeaguePool | None",
    ) -> str:
        del iteration
        candidates = _nn_checkpoint_entries(pool)
        if not candidates:
            return learner_ts_path
        chosen = max(candidates, key=lambda e: (e.elo, e.games_played))
        assert chosen.opponent.path is not None
        return chosen.opponent.path


# ----------------------------------------------------------------------


def create_shadow_source(
    shadow_source: str, *, rng_seed: int = 0
) -> DuplicateShadowSourcePort:
    """Factory: map ``duplicate.shadow_source`` string to an adapter."""
    if shadow_source == "previous_iteration":
        return PreviousIterationShadowSource()
    if shadow_source == "league_pool":
        return LeaguePoolShadowSource(rng_seed=rng_seed)
    if shadow_source == "best_snapshot":
        return BestSnapshotShadowSource()
    raise ValueError(
        f"Unknown duplicate.shadow_source={shadow_source!r}; "
        "expected one of 'previous_iteration', 'league_pool', 'best_snapshot'."
    )
