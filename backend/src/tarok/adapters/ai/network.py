"""Neural network for the RL agent — multi-head actor-critic architecture.

Supports four decision types with separate action heads:
  - Bidding (8 actions: pass + 7 contracts)
  - King calling (4 actions: one per suit)
  - Talon group selection (6 actions)
  - Card play (54 actions: one per card)

Supports an Oracle Critic mode (Perfect Training, Imperfect Execution):
  - Actor heads see only imperfect information (player's own observation)
  - Critic sees perfect information (all hands) during training
  - At deployment, the oracle critic is discarded
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tarok.adapters.ai.encoding import (
    STATE_SIZE,
    ORACLE_STATE_SIZE,
    BID_ACTION_SIZE,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
    CARD_ACTION_SIZE,
    ANNOUNCE_ACTION_SIZE,
    DecisionType,
)


class TarokNet(nn.Module):
    """Multi-head Actor-Critic network for all Tarok decisions.

    When oracle_critic=True, the critic uses a separate backbone that
    takes the full perfect-information state (all hands visible).
    """

    def __init__(self, hidden_size: int = 256, oracle_critic: bool = False):
        super().__init__()
        self.oracle_critic_enabled = oracle_critic

        # Actor backbone (imperfect info — only sees own hand)
        self.shared = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        half = hidden_size // 2

        # Actor heads — one per decision type
        self.bid_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, BID_ACTION_SIZE),
        )
        self.king_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, KING_ACTION_SIZE),
        )
        self.talon_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, TALON_ACTION_SIZE),
        )
        self.card_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, CARD_ACTION_SIZE),
        )
        self.announce_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, ANNOUNCE_ACTION_SIZE),
        )

        # Oracle critic backbone (perfect info — sees all hands during training)
        if oracle_critic:
            self.critic_backbone = nn.Sequential(
                nn.Linear(ORACLE_STATE_SIZE, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )

        # Critic head — shared value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )

        self._heads = {
            DecisionType.BID: self.bid_head,
            DecisionType.KING_CALL: self.king_head,
            DecisionType.TALON_PICK: self.talon_head,
            DecisionType.CARD_PLAY: self.card_head,
            DecisionType.ANNOUNCE: self.announce_head,
        }

    def forward(
        self,
        state: torch.Tensor,
        decision_type: DecisionType = DecisionType.CARD_PLAY,
        oracle_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(state)
        logits = self._heads[decision_type](shared)

        # Critic: use oracle backbone if available and oracle_state provided
        if self.oracle_critic_enabled and oracle_state is not None:
            critic_features = self.critic_backbone(oracle_state)
        else:
            critic_features = shared

        value = self.critic(critic_features)
        return logits, value

    def get_action(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType = DecisionType.CARD_PLAY,
        oracle_state: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select an action from legal moves. Returns (action_idx, log_prob, value)."""
        logits, value = self(state, decision_type, oracle_state)
        # Mask illegal actions with -inf
        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = float("-inf")

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action).squeeze(), value.squeeze(-1).squeeze()

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType = DecisionType.CARD_PLAY,
        oracle_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a batch of actions. Returns (log_probs, values, entropy)."""
        logits, values = self(state, decision_type, oracle_state)
        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = float("-inf")

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy
