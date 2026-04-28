"""Use case: run a single training iteration — thin orchestrator.

Delegates each phase to a single-responsibility use case:
  1. CollectExperiences  — self-play + human-data merge + run stats
  2. UpdatePolicy        — PPO hyperparameter overrides + gradient update
  3. ExportModel         — write TorchScript inference artefact
  4. MeasurePlacement    — greedy benchmark (or carry-forward)
  5. SaveCheckpoint      — persist .pt file + assemble IterationResult
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig, variant_int
from training.ports.benchmark_port import BenchmarkPort
from training.ports.duplicate_iteration_stats_port import DuplicateIterationStatsPort
from training.ports.duplicate_pairing_port import DuplicatePairingPort
from training.ports.duplicate_reward_port import DuplicateRewardPort
from training.ports.duplicate_shadow_source_port import DuplicateShadowSourcePort
from training.ports.model_port import ModelPort
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.collect_duplicate_experiences import CollectDuplicateExperiences
from training.use_cases.collect_experiences import CollectExperiences
from training.use_cases.export_model import ExportModel
from training.use_cases.measure_placement import MeasurePlacement
from training.use_cases.save_checkpoint import SaveCheckpoint
from training.use_cases.update_policy import UpdatePolicy

if TYPE_CHECKING:
    from training.entities.league import LeaguePool


class RunIteration:
    def __init__(
        self,
        selfplay: SelfPlayPort,
        ppo: PPOPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
        *,
        duplicate_pairing: DuplicatePairingPort | None = None,
        duplicate_reward: DuplicateRewardPort | None = None,
        duplicate_shadow_source: DuplicateShadowSourcePort | None = None,
        duplicate_iteration_stats: DuplicateIterationStatsPort | None = None,
    ):
        self._presenter = presenter
        self._collect_experiences = CollectExperiences(selfplay, ppo, presenter)
        self._update_policy = UpdatePolicy(ppo, presenter)
        self._export_model = ExportModel(model)
        self._measure_placement = MeasurePlacement(benchmark, presenter)
        self._save_checkpoint = SaveCheckpoint(model)
        # Optional duplicate-RL pipeline. Constructed lazily from the
        # injected ports; left as ``None`` when either port is absent.
        self._collect_duplicate_experiences: CollectDuplicateExperiences | None = (
            CollectDuplicateExperiences(
                selfplay,
                duplicate_pairing,
                duplicate_reward,
                presenter,
                iteration_stats=duplicate_iteration_stats,
            )
            if duplicate_pairing is not None and duplicate_reward is not None
            else None
        )
        # Shadow-source adapter: resolves the frozen policy path for each
        # duplicate iteration. Injected by the composition root (see
        # ``training.container._default_iteration_runner``) via
        # ``training.adapters.duplicate.shadow_sources.create_shadow_source``.
        # Left as ``None`` outside duplicate mode; the duplicate branch
        # below asserts presence before dereferencing.
        self._shadow_source = duplicate_shadow_source

    def execute(
        self,
        iteration: int,
        config: TrainingConfig,
        identity: ModelIdentity,
        ts_path: str,
        save_dir: Path,
        prev_placement: float,
        iter_lr: float | None = None,
        iter_imitation_coef: float | None = None,
        iter_behavioral_clone_coef: float | None = None,
        iter_entropy_coef: float | None = None,
        iter_explore_rate: float | None = None,
        seats_override: str | None = None,
        run_benchmark: bool = True,
        pool: "LeaguePool | None" = None,
    ) -> tuple[IterationResult, dict]:
        duplicate_cfg = getattr(config, "duplicate", None)
        use_duplicate = (
            duplicate_cfg is not None
            and duplicate_cfg.enabled
            and self._collect_duplicate_experiences is not None
        )
        if use_duplicate:
            if self._shadow_source is None:
                raise RuntimeError(
                    "duplicate.enabled=True but no DuplicateShadowSourcePort was "
                    "injected into RunIteration. The composition root must provide "
                    "one via create_shadow_source(...)."
                )
            shadow_path = self._shadow_source.resolve(
                iteration=iteration, learner_ts_path=ts_path, pool=pool,
            )
            # Heuristic-bot shadow sources expose a ``seat_token`` attribute
            # (e.g. ``"bot_v3"``). When present, pairing renders the shadow
            # seat as that token so the engine plays a heuristic bot there
            # instead of an NN — ``shadow_path`` is then only a placeholder.
            shadow_seat_token: str | None = getattr(
                self._shadow_source, "seat_token", None
            )
            shadow_iteration: int | None = None
            if duplicate_cfg.shadow_source == "previous_iteration":
                shadow_iteration = max(iteration - 1, 0)
            elif duplicate_cfg.shadow_source == "trailing":
                shadow_iteration = getattr(self._shadow_source, "last_refresh_iteration", None)
            elif duplicate_cfg.shadow_source == "relative_trailing":
                shadow_iteration = getattr(self._shadow_source, "last_target_iteration", None)
            elif duplicate_cfg.shadow_source in {"league_pool", "best_snapshot", "weakest_snapshot"}:
                # League snapshots use paths like .../iter_005.pt in this repo.
                # If the path follows that convention, surface it.
                m = re.search(r"iter[_-](\d+)", shadow_path)
                if m is not None:
                    shadow_iteration = int(m.group(1))
            refresh_interval = (
                duplicate_cfg.shadow_refresh_interval
                if duplicate_cfg.shadow_source in {"trailing", "relative_trailing"}
                else None
            )
            self._presenter.on_duplicate_shadow_selected(
                iteration=iteration,
                source=duplicate_cfg.shadow_source,
                shadow_path=shadow_path,
                shadow_iteration=shadow_iteration,
                refresh_interval=refresh_interval,
            )
            bundle = self._collect_duplicate_experiences.execute(
                duplicate_config=duplicate_cfg,
                concurrency=config.concurrency,
                explore_rate=config.explore_rate,
                learner_path=ts_path,
                shadow_path=shadow_path,
                pool=pool,
                outplace_session_size=config.outplace_session_size,
                include_oracle_states=False,
                iter_explore_rate=iter_explore_rate,
                seats_label_for_stats=seats_override,
                centaur_handoff_trick=config.centaur_handoff_trick,
                centaur_pimc_worlds=config.centaur_pimc_worlds,
                centaur_endgame_solver=config.centaur_endgame_solver,
                centaur_alpha_mu_depth=config.centaur_alpha_mu_depth,
                centaur_deterministic_seed=config.centaur_deterministic_seed,
                shadow_seat_token=shadow_seat_token,
                variant=variant_int(config.variant),
            )
        else:
            bundle = self._collect_experiences.execute(
                config, identity, ts_path, seats_override, iter_imitation_coef, iter_behavioral_clone_coef,
                iter_explore_rate=iter_explore_rate,
            )

        update = self._update_policy.execute(
            bundle.raw,
            bundle.nn_seats,
            config,
            iter_lr,
            iter_imitation_coef,
            iter_behavioral_clone_coef,
            iter_entropy_coef,
        )

        # Release the heavy Rust self-play payload before I/O phases.
        del bundle.raw
        _release_python_and_device_memory()

        self._export_model.execute(update.new_weights, identity, ts_path)

        placement, bench_time = self._measure_placement.execute(
            iteration, config, ts_path, prev_placement, run_benchmark
        )

        result = self._save_checkpoint.execute(
            iteration, bundle, update, identity, save_dir, placement, bench_time
        )
        return result, update.new_weights


def _release_python_and_device_memory() -> None:
    """Best-effort cleanup between large phases to limit memory growth."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
