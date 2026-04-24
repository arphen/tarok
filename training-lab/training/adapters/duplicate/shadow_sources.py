"""Adapters implementing :class:`DuplicateShadowSourcePort`.

Four implementations cover the useful sources of a frozen shadow policy
for duplicate-RL:

* :class:`PreviousIterationShadowSource` — the learner one iteration ago.
* :class:`TrailingShadowSource` — the learner ``N`` iterations ago;
  keeps a cached copy of the TorchScript file on disk and only refreshes
  it every ``N`` iterations. Used to let the learner accumulate a large
  positive duplicate advantage before the baseline catches up.
* :class:`LeaguePoolShadowSource` — a random ``nn_checkpoint`` entry from
  the live league pool (Gaussian-matchmaking-weighted).
* :class:`BestSnapshotShadowSource` — the highest-Elo ``nn_checkpoint``
    entry in the league pool (the "best ghost" seen so far).
* :class:`WeakestSnapshotShadowSource` — the lowest-Elo ``nn_checkpoint``
    entry in the league pool (the "weakest ghost" seen so far).

All gracefully fall back to the current learner's TorchScript path when
they have no better option (iteration 0, empty pool, pool with only bot
entries). This makes them safe defaults even mid-run, before the league
has accumulated snapshots.
"""

from __future__ import annotations

import math
import os
import random
import shutil
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


class TrailingShadowSource(DuplicateShadowSourcePort):
    """Frozen copy of the learner from ``refresh_interval`` iterations ago.

    Snapshots ``learner_ts_path`` into a sibling file
    ``<stem>.trailing_shadow<ext>`` the first time it is asked for a
    shadow, and refreshes that copy only every ``refresh_interval``
    iterations. Between refreshes the Rust engine loads the stale file,
    so the shadow policy really does lag the learner by up to N-1
    iterations.

    Motivation (see ``docs/double_rl.md``): with
    :class:`PreviousIterationShadowSource` the shadow updates every
    single iteration, so any tiny edge the learner gains is
    immediately absorbed into the baseline. Meanwhile the learner still
    pays the exploration-tax (``explore_rate``) that the shadow avoids,
    leaving the measured duplicate advantage persistently negative and
    collapsing the policy gradient. A trailing shadow lets the learner
    open up a clear, stable positive advantage for several iterations
    before the baseline catches up.

    On iteration 0 (or any time the TorchScript file does not yet
    exist) falls back to ``learner_ts_path`` — identical to
    :class:`PreviousIterationShadowSource`.
    """

    def __init__(self, refresh_interval: int) -> None:
        if refresh_interval < 1:
            raise ValueError(
                f"TrailingShadowSource.refresh_interval must be >= 1, "
                f"got {refresh_interval}"
            )
        self._interval = int(refresh_interval)
        self._last_refresh_iter: int | None = None

    @property
    def last_refresh_iteration(self) -> int | None:
        """Iteration index of the currently cached trailing shadow weights."""
        return self._last_refresh_iter

    @staticmethod
    def _cached_path(learner_ts_path: str) -> str:
        dirpath, fname = os.path.split(learner_ts_path)
        stem, ext = os.path.splitext(fname)
        return os.path.join(dirpath or ".", f"{stem}.trailing_shadow{ext}")

    def resolve(
        self,
        *,
        iteration: int,
        learner_ts_path: str,
        pool: "LeaguePool | None",
    ) -> str:
        del pool
        cached = self._cached_path(learner_ts_path)
        needs_refresh = (
            self._last_refresh_iter is None
            or (iteration - self._last_refresh_iter) >= self._interval
            or not os.path.exists(cached)
        )
        if needs_refresh:
            if not os.path.exists(learner_ts_path):
                # Bootstrap: no TS saved yet (iteration 0 before export).
                # Fall back to the learner path like the previous-iteration
                # source does — engine loads the freshly-initialised weights.
                return learner_ts_path
            shutil.copy2(learner_ts_path, cached)
            self._last_refresh_iter = iteration
        return cached


class RelativeTrailingShadowSource(DuplicateShadowSourcePort):
    """Fixed-lag shadow: always use learner weights from ``iteration - lag``.

    Unlike :class:`TrailingShadowSource` (stepwise refresh), this adapter tracks
    a per-iteration cache of learner snapshots and resolves the shadow path for
    every iteration to exactly ``max(iteration - lag_iterations, 0)`` whenever
    the corresponding cached file exists.
    """

    def __init__(self, lag_iterations: int) -> None:
        if lag_iterations < 1:
            raise ValueError(
                f"RelativeTrailingShadowSource.lag_iterations must be >= 1, "
                f"got {lag_iterations}"
            )
        self._lag = int(lag_iterations)
        self._last_target_iter: int | None = None

    @property
    def last_target_iteration(self) -> int | None:
        """Iteration index of the shadow weights selected on last resolve."""
        return self._last_target_iter

    @staticmethod
    def _cache_path_for_iter(learner_ts_path: str, iter_idx: int) -> str:
        dirpath, fname = os.path.split(learner_ts_path)
        stem, ext = os.path.splitext(fname)
        return os.path.join(dirpath or ".", f"{stem}.relative_shadow_iter_{iter_idx:06d}{ext}")

    def resolve(
        self,
        *,
        iteration: int,
        learner_ts_path: str,
        pool: "LeaguePool | None",
    ) -> str:
        del pool

        # At iteration start, learner_ts_path holds weights from iteration-1.
        # Persist that snapshot so future fixed-lag lookups can resolve exactly.
        source_iter = max(iteration - 1, 0)
        if os.path.exists(learner_ts_path):
            current_cache = self._cache_path_for_iter(learner_ts_path, source_iter)
            shutil.copy2(learner_ts_path, current_cache)

        target_iter = max(iteration - self._lag, 0)
        target_path = self._cache_path_for_iter(learner_ts_path, target_iter)
        if os.path.exists(target_path):
            self._last_target_iter = target_iter
            return target_path

        # Bootstrap/fallback before cache history exists.
        self._last_target_iter = source_iter if os.path.exists(learner_ts_path) else None
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


class WeakestSnapshotShadowSource(DuplicateShadowSourcePort):
    """Lowest-Elo ``nn_checkpoint`` entry in the league pool.

    Picks the "weakest ghost" currently available. Ties on Elo are broken
    by most games played (older/stabler snapshot wins), keeping selection
    deterministic.
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
        chosen = min(candidates, key=lambda e: (e.elo, -e.games_played))
        assert chosen.opponent.path is not None
        return chosen.opponent.path


# ----------------------------------------------------------------------


def create_shadow_source(
    shadow_source: str, *, rng_seed: int = 0, refresh_interval: int = 1
) -> DuplicateShadowSourcePort:
    """Factory: map ``duplicate.shadow_source`` string to an adapter."""
    if shadow_source == "previous_iteration":
        return PreviousIterationShadowSource()
    if shadow_source == "trailing":
        return TrailingShadowSource(refresh_interval=refresh_interval)
    if shadow_source == "relative_trailing":
        return RelativeTrailingShadowSource(lag_iterations=refresh_interval)
    if shadow_source == "league_pool":
        return LeaguePoolShadowSource(rng_seed=rng_seed)
    if shadow_source == "best_snapshot":
        return BestSnapshotShadowSource()
    if shadow_source == "weakest_snapshot":
        return WeakestSnapshotShadowSource()
    raise ValueError(
        f"Unknown duplicate.shadow_source={shadow_source!r}; "
        "expected one of 'previous_iteration', 'trailing', 'relative_trailing', "
        "'league_pool', 'best_snapshot', 'weakest_snapshot'."
    )
