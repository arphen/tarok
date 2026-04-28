"""Numpy implementation of :class:`DuplicateIterationStatsPort`.

Aggregates per-iteration duplicate statistics from the raw outputs of
``SeededSelfPlayAdapter.run_seeded_pods``. Sits under
``training.adapters`` so that numpy can live alongside the computation —
use cases consume the ``DuplicateIterationStats`` entity only.

Counts two orthogonal aggregations:

* **Per-opponent outcomes**: for every active game, compare the learner's
  score with the score of each of the three opponent seats at that same
  table. Bucket the pairwise comparison by the *opponent seat's label*
  (derived from ``pod.opponents`` via the cyclic offset explained below).
  Yields one comparison per (active game, opponent seat) pair — i.e.
  ``3 * n_active_games`` comparisons total, spread across however many
  distinct opponent tokens the league sampled into pods this iteration.

* **Duplicate advantage**: for every active game, the difference between
  the learner's score at the active table and the shadow's score at the
  matched shadow table (same deck, same learner seat). Divided by 100 so
  the units match the PPO reward, which is the same scale the training
  loss optimises. Mean and sample std reported.

Opponent-label recovery
-----------------------

Each pod holds ``opponents: tuple[str, str, str]`` — three tokens filled
into the non-learner seats in ascending seat order (see
``training.adapters.duplicate.rotation_pairing.RotationPairingAdapter._build_seatings``).
At active-game seat ``j`` with the learner at seat ``learner_pos``, the
opponent is

    pod.opponents[j if j < learner_pos else j - 1]

(i.e. the fill skips the learner's seat; no cyclic wrap).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from training.entities.duplicate_iteration_stats import DuplicateIterationStats
from training.entities.duplicate_pod import DuplicatePod
from training.ports.duplicate_iteration_stats_port import DuplicateIterationStatsPort


class NumpyDuplicateIterationStats(DuplicateIterationStatsPort):
    """Default implementation — pure numpy; no torch, no I/O."""

    def __init__(self, score_scale: float = 100.0) -> None:
        if score_scale <= 0:
            raise ValueError(f"score_scale must be > 0, got {score_scale}")
        self._score_scale = float(score_scale)

    def compute(
        self,
        *,
        active_raw: dict[str, Any],
        shadow_scores: Any,
        pods: Sequence[DuplicatePod],
        pod_ids: Any,  # unused — stats derived from pod/variant grid
        learner_positions: Any,
        active_game_ids: Any,
    ) -> DuplicateIterationStats:
        del pod_ids  # per-step pod_ids not needed for per-game aggregation

        if len(pods) == 0:
            return DuplicateIterationStats()

        active_scores = np.asarray(active_raw["scores"], dtype=np.int32)
        shadow_arr = np.asarray(shadow_scores, dtype=np.int32)
        gids = np.asarray(active_game_ids, dtype=np.int64)
        learner_pos_arr = np.asarray(learner_positions, dtype=np.int64)

        n_seats = pods[0].n_seats
        n_pods, n_gpg = gids.shape
        if shadow_arr.shape != (n_pods, n_gpg, n_seats):
            raise ValueError(
                f"shadow_scores shape {shadow_arr.shape} does not match "
                f"(n_pods={n_pods}, n_games_per_group={n_gpg}, {n_seats})"
            )
        if learner_pos_arr.shape != (n_pods, n_gpg):
            raise ValueError(
                f"learner_positions shape {learner_pos_arr.shape} does not "
                f"match (n_pods={n_pods}, n_games_per_group={n_gpg})"
            )

        # Per-game learner and per-seat active scores. scores[game_id] gives
        # the 4-vector of seat scores; use take_along to pick the learner's
        # score at the learner seat.
        # shape: (n_pods, n_gpg, 4)
        per_game_scores = active_scores[gids]
        learner_scores = np.take_along_axis(
            per_game_scores, learner_pos_arr[..., None], axis=-1
        ).squeeze(-1)  # (n_pods, n_gpg)

        # ── Duplicate advantage ──────────────────────────────────────────
        shadow_at_learner_seat = np.take_along_axis(
            shadow_arr, learner_pos_arr[..., None], axis=-1
        ).squeeze(-1)  # (n_pods, n_gpg)
        advantage = (learner_scores - shadow_at_learner_seat).astype(np.float32) / self._score_scale
        n_active_games = int(advantage.size)
        mean_adv = float(advantage.mean()) if n_active_games > 0 else 0.0
        std_adv = float(advantage.std(ddof=1)) if n_active_games > 1 else 0.0

        # ── Per-opponent outplace outcomes ───────────────────────────────
        opponent_outcomes: dict[str, list[int]] = {}  # token -> [lo, oo, d]
        # Iterate pods — typical n_pods is O(100); negligible cost.
        for pod_idx, pod in enumerate(pods):
            for variant_idx in range(n_gpg):
                learner_pos = int(learner_pos_arr[pod_idx, variant_idx])
                l_score = int(learner_scores[pod_idx, variant_idx])
                seats = per_game_scores[pod_idx, variant_idx]
                for j in range(n_seats):
                    if j == learner_pos:
                        continue
                    opp_idx = j if j < learner_pos else j - 1
                    token = pod.opponents[opp_idx]
                    opp_score = int(seats[j])
                    bucket = opponent_outcomes.setdefault(token, [0, 0, 0])
                    if l_score > opp_score:
                        bucket[0] += 1
                    elif l_score < opp_score:
                        bucket[1] += 1
                    else:
                        bucket[2] += 1

        outcomes_out: dict[str, tuple[int, int, int]] = {
            tok: (lo, oo, d) for tok, (lo, oo, d) in opponent_outcomes.items()
        }
        games_out: dict[str, int] = {
            tok: lo + oo + d for tok, (lo, oo, d) in opponent_outcomes.items()
        }

        return DuplicateIterationStats(
            opponent_outcomes=outcomes_out,
            opponent_games=games_out,
            mean_advantage=mean_adv,
            advantage_std=std_adv,
            n_active_games=n_active_games,
        )
