"""Adapter: compute DuplicateArenaResult from a DuplicateRunResult.

Pure numpy; no Torch, no Rust. Mirrors the per-board pairing logic of
:class:`training.adapters.duplicate.shadow_score_reward.ShadowScoreRewardAdapter`
but returns raw score pairs instead of scaled PPO rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from training.entities.duplicate_arena_result import DuplicateArenaResult
from training.ports.duplicate_arena_stats_port import DuplicateArenaStatsPort

if TYPE_CHECKING:
    from training.entities.duplicate_run_result import DuplicateRunResult


class NumpyDuplicateArenaStats(DuplicateArenaStatsPort):
    """Default aggregator: per-board pair extraction + bootstrap CI."""

    def compute(
        self,
        result: "DuplicateRunResult",
        *,
        score_scale: float = 100.0,
        bootstrap_samples: int = 1000,
        rng_seed: int = 0,
    ) -> DuplicateArenaResult:
        if score_scale <= 0:
            raise ValueError(f"score_scale must be > 0, got {score_scale}")
        if bootstrap_samples < 0:
            raise ValueError(
                f"bootstrap_samples must be >= 0, got {bootstrap_samples}"
            )

        active = result.active
        shadow_scores = np.asarray(result.shadow_scores, dtype=np.int32)
        learner_positions = np.asarray(result.learner_positions, dtype=np.int64)
        active_game_ids = np.asarray(result.active_game_ids, dtype=np.int64)

        if shadow_scores.ndim != 3 or shadow_scores.shape[2] != 4:
            raise ValueError(
                "result.shadow_scores must be shape (n_pods, games_per_group, 4); "
                f"got {shadow_scores.shape}"
            )
        if learner_positions.shape != active_game_ids.shape:
            raise ValueError(
                "learner_positions and active_game_ids must align; "
                f"got {learner_positions.shape} vs {active_game_ids.shape}"
            )

        n_pods, games_per_group = active_game_ids.shape
        n_boards = int(n_pods * games_per_group)

        if n_boards == 0:
            return _empty_result()

        # Per-board challenger score: scores[active_game_id, learner_seat].
        scores_np = np.asarray(active["scores"], dtype=np.int32)
        n_games = int(scores_np.shape[0])
        flat_game_ids = active_game_ids.reshape(-1) % n_games
        flat_learner_pos = learner_positions.reshape(-1)
        challenger_raw = scores_np[flat_game_ids, flat_learner_pos].astype(np.float64)

        # Per-board defender score: shadow_scores[pod, game_idx, learner_seat]
        # at the same seat as the challenger played in the active game.
        pod_idx_grid = np.repeat(np.arange(n_pods, dtype=np.int64), games_per_group)
        game_idx_grid = np.tile(np.arange(games_per_group, dtype=np.int64), n_pods)
        defender_raw = shadow_scores[
            pod_idx_grid, game_idx_grid, flat_learner_pos
        ].astype(np.float64)

        advantage = challenger_raw - defender_raw

        mean_adv = float(advantage.mean())
        std_adv = float(advantage.std(ddof=1)) if n_boards > 1 else 0.0
        challenger_mean = float(challenger_raw.mean())
        defender_mean = float(defender_raw.mean())
        imps = mean_adv / float(score_scale)

        ci_low, ci_high = _bootstrap_ci(
            advantage, n_resamples=bootstrap_samples, rng_seed=rng_seed,
        )

        return DuplicateArenaResult(
            boards_played=n_boards,
            challenger_mean_score=challenger_mean,
            defender_mean_score=defender_mean,
            mean_duplicate_advantage=mean_adv,
            duplicate_advantage_std=std_adv,
            ci_low_95=ci_low,
            ci_high_95=ci_high,
            imps_per_board=imps,
        )


def _bootstrap_ci(
    advantage: np.ndarray, *, n_resamples: int, rng_seed: int,
) -> tuple[float, float]:
    if n_resamples <= 0 or advantage.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    n = advantage.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = advantage[idx].mean(axis=1)
    lo = float(np.quantile(resample_means, 0.025))
    hi = float(np.quantile(resample_means, 0.975))
    return (lo, hi)


def _empty_result() -> DuplicateArenaResult:
    return DuplicateArenaResult(
        boards_played=0,
        challenger_mean_score=0.0,
        defender_mean_score=0.0,
        mean_duplicate_advantage=0.0,
        duplicate_advantage_std=0.0,
        ci_low_95=float("nan"),
        ci_high_95=float("nan"),
        imps_per_board=0.0,
    )
