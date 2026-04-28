"""Port: presenter (output) interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.entities.training_run import TrainingRun

if TYPE_CHECKING:
    from training.entities.duplicate_iteration_stats import DuplicateIterationStats
    from training.entities.league import LeaguePool


class PresenterPort(ABC):
    @abstractmethod
    def on_model_loaded(self, identity: ModelIdentity, save_dir: str) -> None: ...

    @abstractmethod
    def on_device_selected(self, device: str) -> None: ...

    @abstractmethod
    def on_training_plan(self, config: TrainingConfig) -> None: ...

    @abstractmethod
    def on_initial_benchmark(self, placement: float, n_games: int, seats: str, elapsed: float) -> None: ...

    @abstractmethod
    def on_training_loop_start(self, config: TrainingConfig) -> None: ...

    @abstractmethod
    def on_iteration_start(self, iteration: int, total: int, elapsed: float) -> None: ...

    @abstractmethod
    def on_selfplay_start(
        self,
        config: TrainingConfig,
        effective_seats: str | None = None,
        iter_explore_rate: float | None = None,
    ) -> None: ...

    @abstractmethod
    def on_selfplay_done(self, n_total: int, n_learner: int, elapsed: float) -> None: ...

    def on_learner_contract_stats(self, stats: dict[str, dict[str, float | int]]) -> None:
        """Called after self-play with per-contract learner bid/win diagnostics. Optional."""

    @abstractmethod
    def on_ppo_start(
        self,
        config: TrainingConfig,
        iter_lr: float | None = None,
        iter_imitation_coef: float | None = None,
        iter_behavioral_clone_coef: float | None = None,
        iter_entropy_coef: float | None = None,
    ) -> None: ...

    @abstractmethod
    def on_ppo_done(self, metrics: dict[str, float], elapsed: float) -> None: ...

    @abstractmethod
    def on_benchmark_start(self, config: TrainingConfig) -> None: ...

    @abstractmethod
    def on_benchmark_done(self, placement: float, elapsed: float) -> None: ...

    def on_benchmark_skipped(self, iteration: int, config: TrainingConfig) -> None:
        """Called when benchmark is intentionally skipped for an iteration. Optional."""

    @abstractmethod
    def on_iteration_done(self, prev_placement: float, curr_placement: float, elapsed: float) -> None: ...

    @abstractmethod
    def on_training_complete(self, run: TrainingRun) -> None: ...

    def on_league_elo_updated(self, pool: LeaguePool, elo_deltas: dict[str, float] | None = None) -> None:
        """Called after Elo ratings are updated each iteration. Optional."""

    def on_league_snapshot_added(self, iteration: int, path: str) -> None:
        """Called when a checkpoint snapshot is added to the league pool. Optional."""

    def on_duplicate_selfplay_start(
        self,
        n_pods: int,
        n_games_per_pod: int,
        unique_opponents: tuple[str, ...],
        explore_rate: float,
    ) -> None:
        """Announce a duplicate-mode self-play phase.

        Called in place of ``on_selfplay_start`` when duplicate RL is
        active: the learner rotates through all four seats, opponents are
        sampled pod-by-pod from the league pool, so the regular
        ``seats: a,b,c,d`` rendering does not apply.

        Default is a no-op so non-terminal presenters (stubs in tests) are
        unaffected.
        """

    def on_duplicate_iteration_stats(self, stats: "DuplicateIterationStats") -> None:
        """Display per-iteration duplicate stats after self-play finishes.

        Surfaces per-opponent outplace rates and mean duplicate advantage
        (learner − shadow per matched deck). Default is a no-op.
        """

    def on_duplicate_shadow_selected(
        self,
        iteration: int,
        source: str,
        shadow_path: str,
        shadow_iteration: int | None = None,
        refresh_interval: int | None = None,
    ) -> None:
        """Announce which frozen shadow policy is used this duplicate iteration.

        Called once per duplicate iteration after the shadow source resolves a
        path. ``shadow_iteration`` is the training iteration the shadow weights
        came from (when known). ``refresh_interval`` is provided for trailing
        shadows and may be ``None`` for other sources.
        """

    def on_initial_league_calibration_start(self, *args, **kwargs) -> None:
        """Called when initial league Elo calibration begins. Optional."""

    def on_initial_league_calibration_done(self, elapsed: float) -> None:
        """Called when initial league Elo calibration finishes. Optional."""

    def on_initial_league_calibration_mixed_result(self, *args, **kwargs) -> None:
        """Called after each matchup result during initial calibration. Optional."""

    def confirm_league_state_reset(
        self,
        previous_profile: str,
        current_profile: str,
        league_pool_dir: str,
    ) -> bool:
        """Ask whether a config-profile mismatch should reset league state.

        Default is safe: refuse reset unless a presenter explicitly confirms.
        """
        return False
