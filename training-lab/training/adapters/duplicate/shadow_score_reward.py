"""Shadow-score-difference duplicate reward adapter.

Computes per-step rewards as::

    reward[step] = (R_learner[game(step)] − R_shadow[matched_game]) / 100

for every *terminal* step, and ``0.0`` for every non-terminal step (the
downstream ``ppo_batch_preparation`` already zeros non-terminals before GAE,
but we stay consistent here so the array is interpretable on its own).

This adapter is the default implementation of
:class:`training.ports.duplicate_reward_port.DuplicateRewardPort`. Alternative
adapters (IMPs, ranking) will share this skeleton and differ only in how
``R_learner − R_shadow`` is mapped to the reward scalar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from training.ports.duplicate_reward_port import DuplicateRewardPort

if TYPE_CHECKING:
    from training.entities.duplicate_pod import DuplicatePod


class ShadowScoreRewardAdapter(DuplicateRewardPort):
    """Default duplicate reward: scaled score difference between learner and shadow."""

    def __init__(self, score_scale: float = 100.0) -> None:
        if score_scale <= 0:
            raise ValueError(f"score_scale must be > 0, got {score_scale}")
        self._score_scale = float(score_scale)

    def compute_rewards(
        self,
        active_raw: Any,
        shadow_scores: Any,
        pods: list["DuplicatePod"],
    ) -> Any:
        game_ids_np = np.asarray(active_raw["game_ids"])
        players_np = np.asarray(active_raw["players"])
        scores_np = np.asarray(active_raw["scores"], dtype=np.int32)
        pod_ids_np = np.asarray(active_raw["pod_ids"])
        learner_positions_np = np.asarray(active_raw.get("learner_positions_flat"))

        n_total = int(len(game_ids_np))
        if n_total == 0:
            return np.zeros(0, dtype=np.float32)

        n_games = int(scores_np.shape[0])
        if n_games == 0:
            return np.zeros(n_total, dtype=np.float32)

        # Map (active) global game_id -> (pod_idx, game_idx_within_pod) via the
        # active_raw-provided ``pod_ids`` and the pod's ``active_game_ids``.
        # The adapter receives ``active_raw`` with ``pod_ids`` already aligned
        # per-step, so we can index shadow_scores directly.
        shadow_arr = np.asarray(shadow_scores, dtype=np.int32)
        if shadow_arr.ndim != 3 or shadow_arr.shape[2] != 4:
            raise ValueError(
                f"shadow_scores must be shape (n_pods, games_per_group, 4); "
                f"got {shadow_arr.shape}"
            )

        # For every active step, look up:
        #   learner score = active_raw.scores[game_id, player]
        #   shadow score  = shadow_scores[pod_id, game_idx_within_pod, learner_seat]
        # The game_idx_within_pod is derivable from the per-pod sequencing;
        # the Rust binding is responsible for emitting it alongside pod_ids.
        within_pod_idx_np = np.asarray(active_raw.get("game_idx_within_pod"))
        if within_pod_idx_np.shape != game_ids_np.shape:
            raise ValueError(
                "active_raw['game_idx_within_pod'] must align with game_ids; "
                f"got {within_pod_idx_np.shape} vs {game_ids_np.shape}"
            )

        gids = game_ids_np % n_games
        learner_scores = scores_np[gids, players_np].astype(np.float32)
        shadow_scores_flat = shadow_arr[
            pod_ids_np.astype(np.int64),
            within_pod_idx_np.astype(np.int64),
            players_np.astype(np.int64),
        ].astype(np.float32)

        diff = (learner_scores - shadow_scores_flat) / self._score_scale

        # Zero non-terminal steps: a step is terminal iff it is the last in its
        # (game_id, player) trajectory. We follow the same convention that
        # ppo_batch_preparation uses (lexsorted keys, run-length boundaries).
        traj_keys = game_ids_np.astype(np.int64) * 4 + players_np.astype(np.int64)
        sort_idx = np.lexsort((np.arange(n_total), traj_keys))
        sorted_keys = traj_keys[sort_idx]
        is_terminal_sorted = np.ones(n_total, dtype=bool)
        if n_total > 1:
            is_terminal_sorted[:-1] = sorted_keys[:-1] != sorted_keys[1:]
        is_terminal = np.empty(n_total, dtype=bool)
        is_terminal[sort_idx] = is_terminal_sorted

        rewards = np.where(is_terminal, diff, 0.0).astype(np.float32)

        # ``learner_positions_np`` is carried through purely for validation in
        # tests; the reward computation itself does not need it because the
        # per-step ``players`` array already tells us which seat produced each
        # experience. We assert consistency when provided.
        if learner_positions_np is not None and learner_positions_np.size > 0:
            if int(learner_positions_np.max()) > 3 or int(learner_positions_np.min()) < 0:
                raise ValueError("learner_positions_flat must contain seat indices in 0..3")

        return rewards
