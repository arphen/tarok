"""Use case: run a head-to-head duplicate match between two checkpoints.

Arena duplicate mode (see ``docs/double_rl.md`` §8). Orchestrates:

1. Build ``n_boards``-worth of duplicate pods via a
   :class:`DuplicatePairingPort`.
2. Play them deterministically via
   :meth:`SelfPlayPort.run_seeded_pods`, with the challenger in the
   learner seat and the defender in the shadow seat.
3. Aggregate the result into :class:`DuplicateArenaResult` via a
   :class:`DuplicateArenaStatsPort`.

All infrastructure is injected; the use case imports only ports and
entities, keeping it free of numpy / torch / rust.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from training.entities.duplicate_arena_result import DuplicateArenaResult
from training.ports.duplicate_arena_stats_port import DuplicateArenaStatsPort
from training.ports.duplicate_pairing_port import DuplicatePairingPort
from training.ports.selfplay_port import SelfPlayPort

if TYPE_CHECKING:
    from training.entities.league import LeagueConfig


class RunDuplicateArena:
    """Head-to-head duplicate match between challenger and defender."""

    def __init__(
        self,
        selfplay: SelfPlayPort,
        pairing: DuplicatePairingPort,
        stats: DuplicateArenaStatsPort,
    ) -> None:
        self._selfplay = selfplay
        self._pairing = pairing
        self._stats = stats

    def execute(
        self,
        *,
        challenger_path: str,
        defender_path: str,
        n_boards: int,
        rng_seed: int = 0,
        explore_rate: float = 0.0,
        concurrency: int = 1,
        opponents: "LeagueConfig | None" = None,
        score_scale: float = 100.0,
        bootstrap_samples: int = 1000,
    ) -> DuplicateArenaResult:
        """Run the match and return aggregated statistics.

        ``n_boards`` is the target number of paired (active, shadow) games.
        Because the rotation pairing emits multiple games per pod, the
        number of pods is ``max(1, n_boards // games_per_pod)`` — the
        actual ``boards_played`` returned in the result is the exact
        paired count.
        """
        if n_boards <= 0:
            raise ValueError(f"n_boards must be > 0, got {n_boards}")
        if explore_rate < 0.0:
            raise ValueError(f"explore_rate must be >= 0, got {explore_rate}")

        # One pod yields 4 paired games under the default rotation, 4 under
        # rotation_4game, 1 under single_seat_2game. We over-allocate pods
        # and let the stats adapter report the exact count.
        n_pods = max(1, n_boards // 4)

        pods = self._pairing.build_pods(
            pool=opponents,
            learner_seat_token=challenger_path,
            shadow_seat_token=defender_path,
            n_pods=n_pods,
            rng_seed=rng_seed,
        )

        run_result = self._selfplay.run_seeded_pods(
            learner_path=challenger_path,
            shadow_path=defender_path,
            pods=pods,
            explore_rate=explore_rate,
            concurrency=concurrency,
            include_oracle_states=False,
        )

        return self._stats.compute(
            run_result,
            score_scale=score_scale,
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed,
        )
