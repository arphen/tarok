"""Experience data structures for PPO training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from training_lab.entities.encoding import DecisionType


@dataclass
class Experience:
    """A single step of experience for PPO training."""
    state: torch.Tensor
    action: int
    log_prob: torch.Tensor
    value: torch.Tensor
    decision_type: DecisionType = DecisionType.CARD_PLAY
    reward: float = 0.0
    done: bool = False
    oracle_state: torch.Tensor | None = None
    legal_mask: torch.Tensor | None = None
    game_id: int = 0
    step_in_game: int = 0


@dataclass
class TaggedExperience:
    """Experience annotated with policy version for staleness tracking."""
    experience: Experience
    policy_version: int
    collection_time: float


@dataclass
class ExperienceBatch:
    """A materialized tensor batch of N experiences, grouped by DecisionType.

    Ready for GPU transfer and PPO update.
    """
    states: torch.Tensor          # (N, STATE_SIZE)
    actions: torch.Tensor         # (N,) int64
    old_log_probs: torch.Tensor   # (N,)
    values: torch.Tensor          # (N,)
    returns: torch.Tensor         # (N,) — GAE-computed
    advantages: torch.Tensor      # (N,) — GAE-computed
    legal_masks: torch.Tensor     # (N, action_size)
    decision_types: list[DecisionType]
    oracle_states: torch.Tensor | None = None  # (N, ORACLE_STATE_SIZE) or None
