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
from pathlib import Path

import torch

from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.ports.benchmark_port import BenchmarkPort
from training.ports.model_port import ModelPort
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.collect_experiences import CollectExperiences
from training.use_cases.export_model import ExportModel
from training.use_cases.measure_placement import MeasurePlacement
from training.use_cases.save_checkpoint import SaveCheckpoint
from training.use_cases.update_policy import UpdatePolicy


class RunIteration:
    def __init__(
        self,
        selfplay: SelfPlayPort,
        ppo: PPOPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
    ):
        self._collect_experiences = CollectExperiences(selfplay, ppo, presenter)
        self._update_policy = UpdatePolicy(ppo, presenter)
        self._export_model = ExportModel(model)
        self._measure_placement = MeasurePlacement(benchmark, presenter)
        self._save_checkpoint = SaveCheckpoint(model)

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
        seats_override: str | None = None,
        run_benchmark: bool = True,
    ) -> tuple[IterationResult, dict]:
        bundle = self._collect_experiences.execute(
            config, identity, ts_path, seats_override, iter_imitation_coef, iter_behavioral_clone_coef
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
