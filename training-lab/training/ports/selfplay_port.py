"""Port: self-play engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from training.entities.duplicate_pod import DuplicatePod
    from training.entities.duplicate_run_result import DuplicateRunResult


class SelfPlayPort(ABC):
    @abstractmethod
    def run(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        explore_rate: float,
        concurrency: int,
        include_replay_data: bool = False,
        include_oracle_states: bool = False,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        centaur_handoff_trick: int | None = None,
        centaur_pimc_worlds: int | None = None,
        centaur_endgame_solver: str | None = None,
        centaur_alpha_mu_depth: int | None = None,
        centaur_deterministic_seed: int | None = None,
        variant: int = 0,
    ) -> dict[str, Any]:
        """Run self-play games, return raw experience dict."""

    @abstractmethod
    def compute_run_stats(
        self,
        raw: dict[str, Any],
        seat_labels: list[str],
        session_size: int = 50,
    ) -> tuple[int, tuple[float, float, float, float], dict[int, tuple[int, int, int]]]:
        """Compute ``(n_learner, mean_scores, seat_outcomes)`` from one run.

        ``seat_outcomes`` is keyed by seat index (1..3) and stores
        ``(learner_outplaces, opponent_outplaces, draws)`` where outplacing is
        decided from cumulative session totals.  When more than one seat has
        label ``"nn"`` (``min_nn_per_game > 1``), outcomes are accumulated
        across *all* nn seats vs each opponent seat.
        """

    def run_seeded_pods(
        self,
        learner_path: str,
        shadow_path: str,
        pods: list["DuplicatePod"],
        explore_rate: float,
        concurrency: int,
        include_oracle_states: bool = False,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        centaur_handoff_trick: int | None = None,
        centaur_pimc_worlds: int | None = None,
        centaur_endgame_solver: str | None = None,
        centaur_alpha_mu_depth: int | None = None,
        centaur_deterministic_seed: int | None = None,
        variant: int = 0,
    ) -> "DuplicateRunResult":
        """Run a batch of duplicate pods deterministically.

        Each pod is a seeded deal played at both an active table (learner
        seat rotated through seats 0..3) and a shadow table (same deal, same
        opponents, learner replaced by the frozen snapshot at ``shadow_path``).

        The default implementation raises ``NotImplementedError`` so that
        adapters which do not yet support duplicate RL (e.g., the current
        ``RustSelfPlay``) can inherit it unchanged. Duplicate-capable
        adapters override this method. See ``docs/double_rl.md`` §5.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_seeded_pods; "
            "duplicate RL requires a seeded self-play adapter."
        )

