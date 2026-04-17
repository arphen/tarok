"""Use case: run a single training iteration."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

from training.adapters.ppo import load_human_experiences, merge_experiences
from training.entities.iteration_result import IterationResult
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig
from training.ports.benchmark_port import BenchmarkPort
from training.ports.model_port import ModelPort
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort


class RunIteration:
    def __init__(
        self,
        selfplay: SelfPlayPort,
        ppo: PPOPort,
        benchmark: BenchmarkPort,
        model: ModelPort,
        presenter: PresenterPort,
    ):
        self._selfplay = selfplay
        self._ppo = ppo
        self._benchmark = benchmark
        self._model = model
        self._presenter = presenter

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
        seats_override: str | None = None,
        run_benchmark: bool = True,
    ) -> tuple[IterationResult, dict]:
        # Self-play
        effective_seats = seats_override if seats_override is not None else config.seats
        self._presenter.on_selfplay_start(config, effective_seats=effective_seats)
        t0 = time.time()
        include_oracle_states = bool(
            identity.oracle_critic
            and (iter_imitation_coef if iter_imitation_coef is not None else config.imitation_coef) > 0.0
        )
        raw = self._selfplay.run(
            ts_path,
            config.games,
            effective_seats,
            config.explore_rate,
            config.concurrency,
            include_oracle_states=include_oracle_states,
        )
        # Only learn from seats labelled "nn" (the learner).  Frozen snapshots
        # (path-based .pt seats) and heuristic bots are excluded — their log_prob
        # and value come from a different model.
        seat_labels = [s.strip() for s in effective_seats.split(",")]
        nn_seats = [i for i, s in enumerate(seat_labels) if s == "nn"]
        players_np = np.asarray(raw["players"])
        n_total = len(players_np)
        n_learner = int(sum(np.sum(players_np == s) for s in nn_seats))
        sp_time = time.time() - t0
        self._presenter.on_selfplay_done(n_total, n_learner, sp_time)

        # Compute per-seat mean scores as early as possible, then allow
        # large self-play tensors to be released right after PPO.
        scores_arr = raw.get("scores")  # shape (n_games, 4) or None
        if scores_arr is not None and len(scores_arr) > 0:
            scores_np = np.asarray(scores_arr)
            ms = np.mean(scores_np, axis=0)
            mean_scores: tuple[float, float, float, float] = (
                float(ms[0]), float(ms[1]), float(ms[2]), float(ms[3])
            )
            # Per-game outcomes: compare learner (seat 0) vs each opponent seat
            seat_outcomes: dict[int, tuple[int, int, int]] = {}
            learner_scores = scores_np[:, 0]
            for si in range(1, 4):
                if seat_labels[si] == "nn":
                    continue  # learner vs learner — skip
                opp_scores = scores_np[:, si]
                wins = int(np.sum(learner_scores > opp_scores))
                losses = int(np.sum(learner_scores < opp_scores))
                draws = int(np.sum(learner_scores == opp_scores))
                seat_outcomes[si] = (wins, losses, draws)
        else:
            mean_scores = (0.0, 0.0, 0.0, 0.0)
            seat_outcomes = {}

        # Merge human experiences (replay forever — every iteration)
        if config.human_data_dir:
            human_raw = load_human_experiences(config.human_data_dir)
            if human_raw is not None:
                raw = merge_experiences(raw, human_raw)

        # PPO update
        if iter_lr is not None:
            self._ppo.set_lr(iter_lr)
        if iter_imitation_coef is not None:
            self._ppo.set_imitation_coef(iter_imitation_coef)
        self._presenter.on_ppo_start(config, iter_lr=iter_lr, iter_imitation_coef=iter_imitation_coef)
        t0 = time.time()
        metrics, new_weights = self._ppo.update(raw, nn_seats)
        ppo_time = time.time() - t0
        self._presenter.on_ppo_done(metrics, ppo_time)

        # Release the heavy Rust self-play payload before benchmark/checkpoint I/O.
        del raw
        _release_python_and_device_memory()

        # Export updated model
        self._model.export_for_inference(
            new_weights, identity.hidden_size, identity.oracle_critic, identity.model_arch, ts_path,
        )

        # Benchmark (optional, controlled by config checkpoints)
        if run_benchmark:
            self._presenter.on_benchmark_start(config)
            t0 = time.time()
            placement = self._benchmark.measure_placement(
                ts_path, config.bench_games, config.effective_bench_seats,
                config.concurrency, session_size=50,
            )
            bench_time = time.time() - t0
            self._presenter.on_benchmark_done(placement, bench_time)
        else:
            placement = prev_placement
            bench_time = 0.0
            self._presenter.on_benchmark_skipped(iteration, config)

        # Save checkpoint
        ckpt_path = str(save_dir / f"iter_{iteration:03d}.pt")
        self._model.save_checkpoint(
            new_weights, identity.hidden_size, identity.oracle_critic, identity.model_arch,
            iteration, metrics["total_loss"], placement, ckpt_path,
        )

        result = IterationResult(
            iteration=iteration,
            placement=placement,
            loss=metrics["total_loss"],
            policy_loss=metrics["policy_loss"],
            value_loss=metrics["value_loss"],
            entropy=metrics["entropy"],
            n_experiences=n_total,
            selfplay_time=sp_time,
            ppo_time=ppo_time,
            bench_time=bench_time,
            seat_config_used=effective_seats,
            mean_scores=mean_scores,
            seat_outcomes=seat_outcomes,
        )
        return result, new_weights


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
