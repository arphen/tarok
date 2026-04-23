"""PPO batch preparation helpers."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import tarok_engine as te
import torch


PAGAT_CARD_IDX = 0
MOND_CARD_IDX = 20
SKIS_CARD_IDX = 21

# Dominance inside a trick for shaped rewards: Pagat > Skis > Mond.
_TRULA_SHAPED_PRIORITY = (PAGAT_CARD_IDX, SKIS_CARD_IDX, MOND_CARD_IDX)

TRULA_DOMINANCE_REWARD = 0.80
TRULA_DOMINANCE_PENALTY = -0.80

PAGAT_LOSS_TO_NON_PARTNER_PENALTY = -1.50
MOND_LOSS_TO_NON_PARTNER_PENALTY = -2.00
MOND_CAPTURE_OPPONENT_REWARD = 2.00
SKIS_LOSS_TO_NON_PARTNER_PENALTY = -1.80
SKIS_CAPTURE_OPPONENT_REWARD = 1.80


def _card_suit_idx(card_idx: int) -> int | None:
    if 0 <= card_idx <= 21:
        return None
    if 22 <= card_idx <= 53:
        return (card_idx - 22) // 8
    return None


def _is_same_team(player_a: int, player_b: int, declarer: int, partner: int) -> bool:
    if player_a == player_b:
        return True
    if declarer < 0:
        return False
    declarer_team = {declarer}
    if partner >= 0:
        declarer_team.add(partner)
    return (player_a in declarer_team) == (player_b in declarer_team)


def _compute_special_shaped_bonus_by_game(raw: dict[str, Any], n_games: int) -> np.ndarray:
    bonuses = np.zeros((n_games, 4), dtype=np.float32)
    traces = raw.get("traces")
    if traces is None:
        return bonuses

    declarers_np = np.asarray(raw.get("declarers", np.full(n_games, -1, dtype=np.int8)), dtype=np.int16)
    partners_np = np.asarray(raw.get("partners", np.full(n_games, -1, dtype=np.int8)), dtype=np.int16)

    n_traces = min(n_games, len(traces))
    for gid in range(n_traces):
        trace = traces[gid]
        if not isinstance(trace, dict):
            continue
        cards_played = trace.get("cards_played")
        if not cards_played:
            continue

        declarer = int(declarers_np[gid]) if gid < len(declarers_np) else -1
        partner = int(partners_np[gid]) if gid < len(partners_np) else -1

        # cards_played is a flat sequence of (player, card_idx) in trick order.
        n_full = (len(cards_played) // 4) * 4
        for base in range(0, n_full, 4):
            trick_slice = cards_played[base : base + 4]
            trick: list[tuple[int, int]] = []
            for entry in trick_slice:
                try:
                    player = int(entry[0])
                    card_idx = int(entry[1])
                except Exception:
                    trick = []
                    break
                trick.append((player, card_idx))
            if len(trick) != 4:
                continue

            lead_suit = _card_suit_idx(trick[0][1])
            winning_player, winning_card = trick[0]
            for player, card_idx in trick[1:]:
                if te.RustGameState.card_beats(card_idx, winning_card, lead_suit):
                    winning_player = player
                    winning_card = card_idx

            owners = {card_idx: player for player, card_idx in trick}

            # Dominance shaping in the same trick: Pagat > Skis > Mond.
            present_specials = [c for c in _TRULA_SHAPED_PRIORITY if c in owners]
            special_winner = winning_player
            if len(present_specials) >= 2:
                dominant = present_specials[0]
                dominant_owner = owners[dominant]
                special_winner = dominant_owner
                bonuses[gid, dominant_owner] += TRULA_DOMINANCE_REWARD
                for losing_card in present_specials[1:]:
                    losing_owner = owners[losing_card]
                    bonuses[gid, losing_owner] += TRULA_DOMINANCE_PENALTY

            pagat_owner = owners.get(PAGAT_CARD_IDX)
            if pagat_owner is not None and special_winner != pagat_owner:
                if not _is_same_team(pagat_owner, special_winner, declarer, partner):
                    bonuses[gid, pagat_owner] += PAGAT_LOSS_TO_NON_PARTNER_PENALTY

            mond_owner = owners.get(MOND_CARD_IDX)
            if mond_owner is not None and special_winner != mond_owner:
                if not _is_same_team(mond_owner, special_winner, declarer, partner):
                    bonuses[gid, mond_owner] += MOND_LOSS_TO_NON_PARTNER_PENALTY
                    bonuses[gid, special_winner] += MOND_CAPTURE_OPPONENT_REWARD

            skis_owner = owners.get(SKIS_CARD_IDX)
            if skis_owner is not None and special_winner != skis_owner:
                if not _is_same_team(skis_owner, special_winner, declarer, partner):
                    bonuses[gid, skis_owner] += SKIS_LOSS_TO_NON_PARTNER_PENALTY
                    bonuses[gid, special_winner] += SKIS_CAPTURE_OPPONENT_REWARD

    return bonuses


def _broadcast_terminal_advantage(
    terminal_adv: np.ndarray,
    game_ids: np.ndarray,
    players: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Broadcast one terminal advantage per trajectory to every step.

    For a trajectory of T steps ending with advantage A, step t (0-indexed
    within the trajectory) gets ``gamma ** (T - 1 - t) * A``. The terminal
    step itself gets exactly ``A``. See docs/double_rl.md §4.2.

    The input ``terminal_adv`` carries the per-step reward dict already
    constructed by the duplicate-RL reward adapter: nonzero at the terminal
    step of each (game_id, player) trajectory, zero elsewhere. We pick up
    the single nonzero entry per trajectory (or fall back to the last
    entry's value if the adapter chose to broadcast) and discount it back.

    Pure numpy — no torch, no Rust dependency.
    """
    n = terminal_adv.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    gids = np.asarray(game_ids, dtype=np.int64)
    plrs = np.asarray(players, dtype=np.int64)
    traj_keys = gids * 4 + plrs
    # Stable sort by (traj_key, original_index) so within each trajectory the
    # rows stay in the original engine-emitted order (= chronological).
    sort_idx = np.lexsort((np.arange(n), traj_keys))
    sorted_keys = traj_keys[sort_idx]
    sorted_adv = np.asarray(terminal_adv, dtype=np.float32)[sort_idx]

    out_sorted = np.zeros(n, dtype=np.float32)
    # Find trajectory boundaries.
    # ``is_terminal`` is true at the last step of each (game, player) run.
    is_terminal = np.ones(n, dtype=bool)
    is_terminal[:-1] = sorted_keys[:-1] != sorted_keys[1:]
    # Per-trajectory terminal advantage = value at the terminal row.
    # (The reward adapter sets zero for non-terminal rows.)
    terminal_values = sorted_adv[is_terminal]

    # Trajectory lengths and segment ids.
    term_pos = np.flatnonzero(is_terminal)
    # Segment id for each row: how many terminals have occurred at or before
    # this row? Equivalently, cumulative terminals minus the one at this row.
    seg_id = np.cumsum(is_terminal) - is_terminal.astype(np.int64)
    # Clip: all rows belong to seg 0..n_segments-1.
    n_segments = term_pos.shape[0]
    if n_segments == 0:
        return np.zeros(n, dtype=np.float32)
    seg_id = np.clip(seg_id, 0, n_segments - 1)

    # Within-trajectory step index (0 at first step, T-1 at terminal).
    seg_start = np.concatenate([[0], term_pos[:-1] + 1])
    step_idx = np.arange(n) - seg_start[seg_id]
    # Distance to terminal = (T - 1 - step_idx).
    traj_len = (term_pos - seg_start + 1)[seg_id]
    dist_to_terminal = traj_len - 1 - step_idx
    discount = np.power(np.float32(gamma), dist_to_terminal.astype(np.float32))
    out_sorted = terminal_values[seg_id] * discount

    out = np.empty(n, dtype=np.float32)
    out[sort_idx] = out_sorted
    return out


def prepare_batched(raw: dict[str, Any], gamma: float = 0.99, gae_lambda: float = 0.95) -> dict[str, Any]:
    """Convert raw Rust arrays into batched tensors and vectorized GAE inputs."""

    states_np = np.asarray(raw["states"])
    actions_np = np.asarray(raw["actions"])
    log_probs_np = np.asarray(raw["log_probs"])
    values_np = np.asarray(raw["values"])
    decision_types_np = np.asarray(raw["decision_types"])
    game_ids_np = np.asarray(raw["game_ids"])
    players_np = np.asarray(raw["players"])
    scores_np = np.asarray(raw["scores"])
    legal_masks_np = np.asarray(raw["legal_masks"])
    behavioral_clone_raw = raw.get("behavioral_clone_mask")
    behavioral_clone_mask_np = (
        np.asarray(behavioral_clone_raw, dtype=bool)
        if behavioral_clone_raw is not None
        else np.zeros(len(actions_np), dtype=bool)
    )
    oracle_states_raw = raw.get("oracle_states")
    oracle_states_np = np.asarray(oracle_states_raw, dtype=np.float32) if oracle_states_raw is not None else None
    oracle_valid_raw = raw.get("oracle_valid_mask")
    if oracle_states_np is None:
        oracle_valid_mask_np = None
    elif oracle_valid_raw is not None:
        oracle_valid_mask_np = np.asarray(oracle_valid_raw, dtype=bool)
    else:
        oracle_valid_mask_np = np.ones(len(actions_np), dtype=bool)

    game_modes_np = np.asarray(raw["game_modes"], dtype=np.int8)

    n_total = len(actions_np)
    # `scores_np` is expected to be [num_games, 4] aligned with game_ids. A stray
    # game_id outside that range would silently wrap around via modulo and attribute
    # rewards to the wrong game, so verify the invariant up front.
    if n_total > 0 and scores_np.shape[0] > 0:
        max_gid = int(game_ids_np.max())
        if max_gid >= scores_np.shape[0]:
            raise ValueError(
                "game_ids contain id "
                f"{max_gid} >= scores.shape[0]={scores_np.shape[0]}; "
                "Rust self-play returned inconsistent batch alignment."
            )
        if int(game_ids_np.min()) < 0:
            raise ValueError("game_ids must be non-negative")
    gids = game_ids_np % scores_np.shape[0] if scores_np.shape[0] > 0 else game_ids_np
    precomputed_rewards = raw.get("precomputed_rewards")
    actor_only = bool(raw.get("actor_only", False))
    if actor_only:
        # Actor-only mode (docs/double_rl.md §4.2): GAE + critic are gone.
        # Advantage for every step in a trajectory is the discounted terminal
        # advantage; returns equal advantages (no critic bootstrap).
        if precomputed_rewards is None:
            raise ValueError(
                "actor_only=True requires precomputed_rewards in raw "
                "(duplicate-RL reward adapter must be wired)."
            )
        terminal_adv_np = np.asarray(precomputed_rewards, dtype=np.float32)
        if terminal_adv_np.shape != (n_total,):
            raise ValueError(
                f"precomputed_rewards shape {terminal_adv_np.shape} does not match "
                f"experience count {n_total}"
            )
        advantages_np = _broadcast_terminal_advantage(
            terminal_adv_np, game_ids_np, players_np, gamma=gamma
        )
        if advantages_np.size > 1:
            adv_mean = float(advantages_np.mean())
            adv_std = float(advantages_np.std())
            advantages_np = (advantages_np - adv_mean) / (adv_std + 1e-8)
        returns_np = advantages_np.copy()
        # No critic ⇒ values column is zero; keep (N,3) layout for the
        # downstream PPO loss which indexes column 0 for old_values.
        vad_np = np.stack(
            [np.zeros(n_total, dtype=np.float32), advantages_np, returns_np], axis=1
        )
        return {
            "states": torch.from_numpy(states_np),
            "actions": torch.from_numpy(actions_np.astype(np.int64)),
            "log_probs": torch.from_numpy(log_probs_np),
            "vad": torch.from_numpy(vad_np),
            "decision_types": decision_types_np,
            "legal_masks": torch.from_numpy(legal_masks_np),
            "oracle_states": (
                torch.from_numpy(oracle_states_np) if oracle_states_np is not None else None
            ),
            "oracle_valid_mask": (
                torch.from_numpy(oracle_valid_mask_np)
                if oracle_valid_mask_np is not None
                else None
            ),
            "game_modes": game_modes_np,
            "behavioral_clone_mask": torch.from_numpy(behavioral_clone_mask_np),
            "actor_only": True,
        }

    if precomputed_rewards is not None:
        # Duplicate-RL conservative mode: the reward source has been replaced by
        # an adapter that supplies the per-step terminal reward directly. Skip
        # the score-based extraction and any additional shaping — the adapter
        # is responsible for emitting exactly the reward the PPO update should
        # receive. See docs/double_rl.md §4.1.
        rewards_np = np.asarray(precomputed_rewards, dtype=np.float32)
        if rewards_np.shape != (n_total,):
            raise ValueError(
                f"precomputed_rewards shape {rewards_np.shape} does not match "
                f"experience count {n_total}"
            )
    else:
        rewards_np = scores_np[gids, players_np].astype(np.float32) / 100.0
        if scores_np.shape[0] > 0:
            shaped_bonus_by_game = _compute_special_shaped_bonus_by_game(raw, int(scores_np.shape[0]))
            rewards_np = rewards_np + shaped_bonus_by_game[gids, players_np]

    traj_keys = game_ids_np.astype(np.int64) * 4 + players_np.astype(np.int64)
    sort_idx = np.lexsort((np.arange(n_total), traj_keys))

    sorted_keys = np.asarray(traj_keys[sort_idx], dtype=np.int64)
    sorted_values = np.asarray(values_np[sort_idx], dtype=np.float32)
    sorted_rewards = np.asarray(rewards_np[sort_idx], dtype=np.float32)

    # Zero out non-terminal rewards so the critic learns position value,
    # not a countdown timer. A step is terminal if it's the last element
    # or if the next step belongs to a different (game, player) trajectory.
    is_terminal = np.ones(n_total, dtype=bool)
    if n_total > 0:
        is_terminal[:-1] = sorted_keys[:-1] != sorted_keys[1:]
    sorted_rewards[~is_terminal] = 0.0

    advantages_sorted, returns_sorted = te.compute_gae(
        sorted_values,
        sorted_rewards,
        sorted_keys,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages_sorted = np.asarray(advantages_sorted, dtype=np.float32)
    returns_sorted = np.asarray(returns_sorted, dtype=np.float32)

    advantages_np = np.empty(n_total, dtype=np.float32)
    returns_np = np.empty(n_total, dtype=np.float32)
    advantages_np[sort_idx] = advantages_sorted
    returns_np[sort_idx] = returns_sorted

    # Normalize advantages once globally to avoid over-scaling rare
    # decision-type / game-mode subgroups during PPO updates.
    if advantages_np.size > 1:
        adv_mean = float(advantages_np.mean())
        adv_std = float(advantages_np.std())
        advantages_np = (advantages_np - adv_mean) / (adv_std + 1e-8)

    # Stack values / advantages / returns into a single (N, 3) matrix so that
    # one advanced-index copy on the CPU and one PCIe transfer replace three.
    # Column layout: 0=old_values, 1=advantages (normalised), 2=returns.
    vad_np = np.stack([values_np.astype(np.float32), advantages_np, returns_np], axis=1)

    return {
        "states": torch.from_numpy(states_np),
        "actions": torch.from_numpy(actions_np.astype(np.int64)),
        "log_probs": torch.from_numpy(log_probs_np),
        "vad": torch.from_numpy(vad_np),
        "decision_types": decision_types_np,
        "legal_masks": torch.from_numpy(legal_masks_np),
        "oracle_states": torch.from_numpy(oracle_states_np) if oracle_states_np is not None else None,
        "oracle_valid_mask": (
            torch.from_numpy(oracle_valid_mask_np) if oracle_valid_mask_np is not None else None
        ),
        "game_modes": game_modes_np,
        "behavioral_clone_mask": torch.from_numpy(behavioral_clone_mask_np),
        "actor_only": False,
    }


def release_allocator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass