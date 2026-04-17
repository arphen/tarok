"""Experience type for PPO training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tarok_model.encoding import DecisionType, GameMode


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
    player: int = 0
    game_mode: GameMode | None = None
