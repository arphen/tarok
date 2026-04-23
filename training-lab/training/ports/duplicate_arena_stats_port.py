"""Port: aggregate a DuplicateRunResult into head-to-head match statistics.

Split out from the reward computation because the *reward* port maps
active steps to scalar PPO rewards, whereas arena statistics are about
per-board raw score pairs and distributional summaries. Keeping them
separate lets alternative reward models (IMPs, ranking) reuse the same
stats adapter unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.entities.duplicate_arena_result import DuplicateArenaResult
    from training.entities.duplicate_run_result import DuplicateRunResult


class DuplicateArenaStatsPort(ABC):
    """Compute head-to-head match statistics from a seeded-pods run."""

    @abstractmethod
    def compute(
        self,
        result: "DuplicateRunResult",
        *,
        score_scale: float = 100.0,
        bootstrap_samples: int = 1000,
        rng_seed: int = 0,
    ) -> "DuplicateArenaResult":
        """Extract per-board (challenger, defender) pairs and aggregate.

        Parameters
        ----------
        result
            Output of :meth:`SelfPlayPort.run_seeded_pods`.
        score_scale
            Divisor applied to score deltas when computing
            ``imps_per_board``. Defaults to ``100.0`` to match
            ``ShadowScoreRewardAdapter``.
        bootstrap_samples
            Number of bootstrap resamples used for the 95% CI on the mean
            advantage. Pass ``0`` to skip and return ``nan`` bounds.
        rng_seed
            Seed for the bootstrap RNG. Fixed seed keeps the CI
            reproducible across runs with identical inputs.
        """
