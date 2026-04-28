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

    def __init__(
        self,
        score_scale: float = 100.0,
        negative_reward_multiplier: float = 2.0,
        berac_bid_penalty: float = 0.0,
    ) -> None:
        if score_scale <= 0:
            raise ValueError(f"score_scale must be > 0, got {score_scale}")
        if negative_reward_multiplier <= 0:
            raise ValueError(
                "negative_reward_multiplier must be > 0, got "
                f"{negative_reward_multiplier}"
            )
        self._score_scale = float(score_scale)
        self._negative_reward_multiplier = float(negative_reward_multiplier)
        self._berac_bid_penalty = float(berac_bid_penalty)

    def compute_rewards(
        self,
        active_raw: Any,
        shadow_scores: Any,
        pods: list["DuplicatePod"],
        shadow_contracts: Any = None,
    ) -> Any:
        game_ids_np = np.asarray(active_raw["game_ids"])
        players_np = np.asarray(active_raw["players"])
        # Use the per-seat training reward signal (``reward_scores``) rather
        # than the leaderboard ``scores`` field. For defender seats the
        # leaderboard value is always 0, which would make every defender
        # trajectory yield a trivially-zero duplicate advantage regardless of
        # play quality. ``reward_scores`` distributes a non-zero signal to
        # defenders so they actually learn. See engine-rs/src/scoring.rs.
        scores_np = np.asarray(
            active_raw.get("reward_scores", active_raw["scores"]), dtype=np.int32
        )
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
        if shadow_arr.ndim != 3 or shadow_arr.shape[2] not in (3, 4):
            raise ValueError(
                f"shadow_scores must be shape (n_pods, games_per_group, n_seats); "
                f"got {shadow_arr.shape}"
            )
        n_seats = int(shadow_arr.shape[2])

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

        # Amplify losses so low-win-rate contracts (e.g. Berač) get a stronger
        # "avoid this" signal during policy optimization.
        neg_mask = diff < 0.0
        if bool(neg_mask.any()) and self._negative_reward_multiplier != 1.0:
            diff = diff.copy()
            diff[neg_mask] *= self._negative_reward_multiplier

        # Zero non-terminal steps: a step is terminal iff it is the last in its
        # (game_id, player) trajectory. We follow the same convention that
        # ppo_batch_preparation uses (lexsorted keys, run-length boundaries).
        traj_keys = game_ids_np.astype(np.int64) * n_seats + players_np.astype(np.int64)
        sort_idx = np.lexsort((np.arange(n_total), traj_keys))
        sorted_keys = traj_keys[sort_idx]
        is_terminal_sorted = np.ones(n_total, dtype=bool)
        if n_total > 1:
            is_terminal_sorted[:-1] = sorted_keys[:-1] != sorted_keys[1:]
        is_terminal = np.empty(n_total, dtype=bool)
        is_terminal[sort_idx] = is_terminal_sorted

        rewards = np.where(is_terminal, diff, 0.0).astype(np.float32)

        # Bid-divergence credit assignment.
        #
        # When the learner's pod and the shadow's pod played different
        # contracts (e.g. learner bid Berač, shadow bid SoloOne on the
        # same hand), the score difference is overwhelmingly attributable
        # to the *bid* choice rather than to how cards were played: card
        # play could even have been optimal under both contracts and
        # still yield wildly different rewards. Letting the duplicate
        # advantage flow back through the cardplay head poisons that
        # head's gradient with bid-driven noise.
        #
        # For divergent-contract trajectories we move the terminal
        # reward from the last cardplay step to the last *non-cardplay*
        # decision step (last bid / king / talon-pick), and zero the
        # cardplay terminal. ``ppo_batch_preparation`` propagates this
        # mid-trajectory reward through GAE so the bid head receives the
        # gradient while cardplay-step advantages contain only the
        # value-function residual (no future reward → ~0 at convergence).
        decision_types_raw = active_raw.get("decision_types")
        if (
            shadow_contracts is not None
            and decision_types_raw is not None
            and "contracts" in active_raw
        ):
            shadow_contracts_arr = np.asarray(shadow_contracts, dtype=np.int16)
            active_contracts = np.asarray(active_raw["contracts"], dtype=np.int16)
            decision_types_np = np.asarray(decision_types_raw, dtype=np.int8)

            # Per-step learner / shadow contract.
            learner_contract_step = active_contracts[gids]
            shadow_contract_step = shadow_contracts_arr[
                pod_ids_np.astype(np.int64),
                within_pod_idx_np.astype(np.int64),
            ]
            mismatch = (learner_contract_step != shadow_contract_step) & (
                shadow_contract_step != 255  # 255 = sentinel "no shadow contract captured"
            )

            # CARD_PLAY decision_type discriminant value (engine-rs/src/player.rs).
            DT_CARD_PLAY = 3

            if bool(mismatch.any()):
                # Walk each trajectory once. For mismatch trajectories,
                # locate the last non-cardplay step (highest sorted index
                # in the trajectory whose decision_type != CardPlay) and
                # move the terminal reward there. Everything stays on
                # ``rewards`` (precomputed reward stream); the consumer
                # ``prepare_batched`` is responsible for not zeroing
                # non-terminal rewards in the precomputed-rewards path.
                sorted_decision_types = decision_types_np[sort_idx]
                sorted_mismatch = mismatch[sort_idx]
                sorted_rewards = rewards[sort_idx]

                # Trajectory boundaries: terminal step indexes (sorted).
                term_pos = np.flatnonzero(is_terminal_sorted)
                seg_start_pos = np.concatenate([[0], term_pos[:-1] + 1])

                for seg_idx, (start, end) in enumerate(zip(seg_start_pos, term_pos)):
                    if not sorted_mismatch[end]:
                        continue
                    seg_dt = sorted_decision_types[start : end + 1]
                    non_card_local = np.flatnonzero(seg_dt != DT_CARD_PLAY)
                    if non_card_local.size == 0:
                        # No bidding-phase steps recorded for this
                        # trajectory (shouldn't normally happen). Drop
                        # the reward entirely rather than leaving it on
                        # the cardplay terminal where it would mis-train
                        # the card head.
                        sorted_rewards[end] = 0.0
                        continue
                    last_non_card = start + int(non_card_local[-1])
                    sorted_rewards[last_non_card] = sorted_rewards[end]
                    sorted_rewards[end] = 0.0

                rewards = np.empty(n_total, dtype=np.float32)
                rewards[sort_idx] = sorted_rewards

        # Contract-specific bid shaping: discourage Berač at the bid step.
        # This targets a common failure mode where the policy keeps over-bidding
        # Berač despite persistently negative returns.
        if (
            self._berac_bid_penalty != 0.0
            and decision_types_raw is not None
            and "bid_contracts" in active_raw
            and learner_positions_np is not None
            and learner_positions_np.shape == game_ids_np.shape
        ):
            decision_types_np = np.asarray(decision_types_raw, dtype=np.int8)
            bid_contracts_np = np.asarray(active_raw["bid_contracts"], dtype=np.int16)
            sorted_decision_types = decision_types_np[sort_idx]
            sorted_rewards = rewards[sort_idx]
            sorted_gids = gids[sort_idx]
            sorted_learner_pos = learner_positions_np[sort_idx].astype(np.int64)

            # CARD_PLAY decision_type discriminant value (engine-rs/src/player.rs).
            DT_CARD_PLAY = 3
            BERAC_CONTRACT_ID = 8

            term_pos = np.flatnonzero(is_terminal_sorted)
            seg_start_pos = np.concatenate([[0], term_pos[:-1] + 1])

            n_bid_seats = bid_contracts_np.shape[1] if bid_contracts_np.ndim >= 2 else 0
            for start, end in zip(seg_start_pos, term_pos):
                learner_seat = int(sorted_learner_pos[end])
                if learner_seat < 0 or learner_seat >= n_bid_seats:
                    continue
                gid = int(sorted_gids[end])
                if int(bid_contracts_np[gid, learner_seat]) != BERAC_CONTRACT_ID:
                    continue
                seg_dt = sorted_decision_types[start : end + 1]
                non_card_local = np.flatnonzero(seg_dt != DT_CARD_PLAY)
                target = end if non_card_local.size == 0 else start + int(non_card_local[-1])
                sorted_rewards[target] += self._berac_bid_penalty

            rewards = np.empty(n_total, dtype=np.float32)
            rewards[sort_idx] = sorted_rewards

        # ``learner_positions_np`` is carried through purely for validation in
        # tests; the reward computation itself does not need it because the
        # per-step ``players`` array already tells us which seat produced each
        # experience. We assert consistency when provided.
        if learner_positions_np is not None and learner_positions_np.size > 0:
            if int(learner_positions_np.max()) >= n_seats or int(learner_positions_np.min()) < 0:
                raise ValueError(
                    f"learner_positions_flat must contain seat indices in 0..{n_seats - 1}"
                )

        return rewards
