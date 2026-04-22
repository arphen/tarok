"""Adapter: duplicate seeded-pods self-play on top of the Rust engine.

Phase 1 strategy — **strictly additive, no Rust runner refactor required**:

For each *unique* seating that appears in the pod batch we invoke
``tarok_engine.run_self_play`` once, passing the per-game ``deck_seeds`` list
in pod-order. Since the deal is driven solely by ``deck_seed`` and is
independent of the seating, game ``g`` of run ``(seating)`` uses the same
deck as game ``g`` of run ``(other_seating)`` — which is exactly the
duplicate invariant.

Active runs emit experiences (learner seats only). Shadow runs are scored
only — no experiences retained. Results are assembled into a
``DuplicateRunResult`` that the reward adapter and use case can consume.

This keeps Phase 1 simple at the cost of running N independent Rust
invocations (N = number of unique seatings). For ``rotation_8game`` that's
8 invocations; for ``single_seat_2game`` that's 2. Each invocation fully
batches across all pods (``concurrency`` still applies), so the loss vs. a
single fused run is primarily per-invocation setup/teardown.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tarok_engine as te

from training.entities.duplicate_pod import DuplicatePod
from training.entities.duplicate_run_result import DuplicateRunResult
from training.entities.training_config import LEARNER_SEAT_LABELS
from training.ports import SelfPlayPort


log = logging.getLogger(__name__)


_LEARNER_TOKEN = "nn"


def _render_seat_config(
    seating: tuple[str, str, str, str],
    learner_pos: int,
    actor_path: str,
) -> tuple[str, str]:
    """Convert a seating tuple to a comma-separated seat_config and the
    path-or-token used in the learner position.

    The learner position is always rendered as ``"nn"`` (the Rust engine's
    NN slot) so that ``run_self_play`` loads ``actor_path`` exactly once via
    ``model_path``. Opponents are rendered as-is (bot labels or checkpoint
    paths); the engine already dedups them.
    """
    rendered = list(seating)
    if rendered[learner_pos] not in LEARNER_SEAT_LABELS:
        raise ValueError(
            f"seating[{learner_pos}] = {rendered[learner_pos]!r} is not a learner token; "
            f"learner_positions disagree with seating."
        )
    rendered[learner_pos] = _LEARNER_TOKEN
    return ",".join(rendered), actor_path


def _learner_mask_for_seating(seating: tuple[str, str, str, str]) -> np.ndarray:
    """Seats that will emit experiences in an active run."""
    return np.asarray(
        [s in LEARNER_SEAT_LABELS for s in seating],
        dtype=bool,
    )


class SeededSelfPlayAdapter(SelfPlayPort):
    """Duplicate-aware self-play adapter.

    Implements both the legacy ``run`` / ``compute_run_stats`` API (by
    delegating to an inner ``SelfPlayPort`` — the default being
    ``RustSelfPlay``) and the new ``run_seeded_pods`` method.
    """

    def __init__(self, inner: SelfPlayPort) -> None:
        self._inner = inner

    # ---- Legacy port surface: delegate ------------------------------------

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
    ) -> dict[str, Any]:
        return self._inner.run(
            model_path=model_path,
            n_games=n_games,
            seat_config=seat_config,
            explore_rate=explore_rate,
            concurrency=concurrency,
            include_replay_data=include_replay_data,
            include_oracle_states=include_oracle_states,
            lapajne_mc_worlds=lapajne_mc_worlds,
            lapajne_mc_sims=lapajne_mc_sims,
            centaur_handoff_trick=centaur_handoff_trick,
            centaur_pimc_worlds=centaur_pimc_worlds,
            centaur_endgame_solver=centaur_endgame_solver,
            centaur_alpha_mu_depth=centaur_alpha_mu_depth,
        )

    def compute_run_stats(
        self,
        raw: dict[str, Any],
        seat_labels: list[str],
        session_size: int = 50,
    ) -> tuple[int, tuple[float, float, float, float], dict[int, tuple[int, int, int]]]:
        return self._inner.compute_run_stats(raw, seat_labels, session_size)

    # ---- Duplicate surface -------------------------------------------------

    def run_seeded_pods(
        self,
        learner_path: str,
        shadow_path: str,
        pods: list[DuplicatePod],
        explore_rate: float,
        concurrency: int,
        include_oracle_states: bool = False,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        centaur_handoff_trick: int | None = None,
        centaur_pimc_worlds: int | None = None,
        centaur_endgame_solver: str | None = None,
        centaur_alpha_mu_depth: int | None = None,
    ) -> DuplicateRunResult:
        if not pods:
            raise ValueError("run_seeded_pods requires at least one pod")

        n_pods = len(pods)
        n_games_per_group = pods[0].n_games_per_group
        for p in pods:
            if p.n_games_per_group != n_games_per_group:
                raise ValueError(
                    "all pods must share the same n_games_per_group; got "
                    f"{p.n_games_per_group} and {n_games_per_group}"
                )

        deck_seeds = [int(p.deck_seed) for p in pods]

        # One invocation per (variant_index, active/shadow) pair. All pods in
        # a given invocation share the seating, so deck_seeds alone drives
        # dealing differences.
        active_runs: list[dict[str, Any]] = []
        shadow_scores = np.zeros((n_pods, n_games_per_group, 4), dtype=np.int32)

        learner_positions = np.zeros((n_pods, n_games_per_group), dtype=np.int8)
        for pod_idx, p in enumerate(pods):
            for g, lp in enumerate(p.learner_positions):
                learner_positions[pod_idx, g] = int(lp)

        for variant_idx in range(n_games_per_group):
            active_seating = pods[0].active_seatings[variant_idx]
            shadow_seating = pods[0].shadow_seatings[variant_idx]
            active_lp = pods[0].learner_positions[variant_idx]

            # Sanity: every pod must agree on the seat shape at this variant
            # (learner position). Opponent *tokens* may differ pod-to-pod
            # because opponents are sampled per pod, so we issue one Rust
            # call per (variant_idx, opponent_tuple) group.
            groups: dict[
                tuple[tuple[str, str, str, str], tuple[str, str, str, str], int],
                list[int],
            ] = {}
            for pod_idx, p in enumerate(pods):
                if p.learner_positions[variant_idx] != active_lp:
                    raise ValueError(
                        "learner_positions at variant "
                        f"{variant_idx} differ across pods — pairing adapter bug?"
                    )
                key = (
                    p.active_seatings[variant_idx],
                    p.shadow_seatings[variant_idx],
                    active_lp,
                )
                groups.setdefault(key, []).append(pod_idx)

            del active_seating, shadow_seating  # unused: per-group keys drive

            for (act_seat, sh_seat, lp), group_pod_ids in groups.items():
                group_seeds = [deck_seeds[pi] for pi in group_pod_ids]

                active_cfg, _ = _render_seat_config(act_seat, lp, learner_path)
                shadow_cfg, _ = _render_seat_config(sh_seat, lp, shadow_path)

                # Active run: emit experiences.
                active_raw = te.run_self_play(
                    n_games=len(group_seeds),
                    concurrency=min(concurrency, len(group_seeds)),
                    model_path=learner_path,
                    explore_rate=explore_rate,
                    seat_config=active_cfg,
                    include_replay_data=False,
                    include_oracle_states=include_oracle_states,
                    lapajne_mc_worlds=lapajne_mc_worlds,
                    lapajne_mc_sims=lapajne_mc_sims,
                    centaur_handoff_trick=centaur_handoff_trick,
                    centaur_pimc_worlds=centaur_pimc_worlds,
                    centaur_endgame_solver=centaur_endgame_solver,
                    centaur_alpha_mu_depth=centaur_alpha_mu_depth,
                    deck_seeds=group_seeds,
                )
                # Tag each experience row with (pod_idx, variant_idx) so we
                # can stitch them back into global arrays later.
                active_runs.append(
                    {
                        "raw": active_raw,
                        "group_pod_ids": group_pod_ids,
                        "variant_idx": variant_idx,
                        "learner_pos": lp,
                    }
                )

                # Shadow run: collect scores only.
                shadow_raw = te.run_self_play(
                    n_games=len(group_seeds),
                    concurrency=min(concurrency, len(group_seeds)),
                    model_path=shadow_path,
                    explore_rate=0.0,  # shadow plays greedily
                    seat_config=shadow_cfg,
                    include_replay_data=False,
                    include_oracle_states=False,
                    lapajne_mc_worlds=lapajne_mc_worlds,
                    lapajne_mc_sims=lapajne_mc_sims,
                    centaur_handoff_trick=centaur_handoff_trick,
                    centaur_pimc_worlds=centaur_pimc_worlds,
                    centaur_endgame_solver=centaur_endgame_solver,
                    centaur_alpha_mu_depth=centaur_alpha_mu_depth,
                    deck_seeds=group_seeds,
                )
                sh_scores = np.asarray(shadow_raw["scores"], dtype=np.int32)
                # scores from run_self_play are indexed by game_id which the
                # Rust side assigns 0..n_games-1 in group order.
                for local_g, pod_idx in enumerate(group_pod_ids):
                    shadow_scores[pod_idx, variant_idx, :] = sh_scores[local_g, :]

        # Merge all active runs into one experience dict + per-step pod_ids.
        active_merged, pod_ids_flat, active_game_ids = _merge_active_runs(
            active_runs, n_pods, n_games_per_group
        )

        return DuplicateRunResult(
            active=active_merged,
            shadow_scores=shadow_scores,
            pod_ids=pod_ids_flat,
            learner_positions=learner_positions,
            active_game_ids=active_game_ids,
        )


def _merge_active_runs(
    active_runs: list[dict[str, Any]],
    n_pods: int,
    n_games_per_group: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Stitch per-group active ``run_self_play`` outputs into a single dict.

    Returns
    -------
    merged
        Active experience dict with the same schema as ``run_self_play`` —
        same key names, concatenated along the experience axis. ``game_ids``
        is rewritten to a globally unique id = ``pod_idx * n_games_per_group
        + variant_idx`` so that reward lookups can index a 2D
        ``(n_pods, n_games_per_group)`` array with ``gid // n_gpg`` and
        ``gid % n_gpg``.
    pod_ids_flat
        ``(n_total_steps,)`` pod index for every merged experience row.
    active_game_ids
        ``(n_pods, n_games_per_group)`` global game id — i.e. the same value
        present in ``merged["game_ids"]`` for that ``(pod, variant)``.
    """
    # Keys we carry through. Absent keys (e.g. oracle_states when not
    # requested) simply won't appear in merged.
    carry_keys_1d = [
        "actions",
        "log_probs",
        "values",
        "decision_types",
        "game_modes",
        "game_ids",
        "players",
    ]
    carry_keys_2d = ["states", "legal_masks", "oracle_states"]
    per_game_keys = ["scores"]  # per-game, not per-step

    active_game_ids = np.zeros((n_pods, n_games_per_group), dtype=np.int64)

    # Rewrite each run's game_ids to global ids and collect row slabs.
    row_blocks: dict[str, list[np.ndarray]] = {k: [] for k in carry_keys_1d + carry_keys_2d}
    pod_id_blocks: list[np.ndarray] = []
    scores_out = np.zeros((n_pods, n_games_per_group, 4), dtype=np.int32)

    for run in active_runs:
        raw = run["raw"]
        variant_idx = run["variant_idx"]
        group_pod_ids = run["group_pod_ids"]

        # Map local game_id (0..len(group)-1) → global id & pod index.
        local_game_ids = np.asarray(raw["game_ids"], dtype=np.int64)
        global_ids = np.empty_like(local_game_ids)
        pod_ids_for_rows = np.empty_like(local_game_ids)
        for local_g, pod_idx in enumerate(group_pod_ids):
            mask = local_game_ids == local_g
            gid = pod_idx * n_games_per_group + variant_idx
            global_ids[mask] = gid
            pod_ids_for_rows[mask] = pod_idx
            active_game_ids[pod_idx, variant_idx] = gid

            # scores: shape (n_games, 4) in group-local ordering
            scores_out[pod_idx, variant_idx, :] = np.asarray(
                raw["scores"], dtype=np.int32
            )[local_g, :]

        pod_id_blocks.append(pod_ids_for_rows)

        for k in carry_keys_1d:
            if k not in raw:
                continue
            if k == "game_ids":
                row_blocks[k].append(global_ids.astype(np.uint32, copy=False))
            else:
                row_blocks[k].append(np.asarray(raw[k]))
        for k in carry_keys_2d:
            if k in raw:
                row_blocks[k].append(np.asarray(raw[k]))

    merged: dict[str, Any] = {}
    for k, blocks in row_blocks.items():
        if blocks:
            merged[k] = np.concatenate(blocks, axis=0)
    merged["scores"] = scores_out.reshape(n_pods * n_games_per_group, 4)
    pod_ids_flat = np.concatenate(pod_id_blocks, axis=0) if pod_id_blocks else np.zeros(
        0, dtype=np.int64
    )

    _ = per_game_keys  # keep name referenced for readability
    return merged, pod_ids_flat, active_game_ids
