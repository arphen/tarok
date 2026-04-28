"""Neural network for the 3-player Tarok agent.

This is a clean-slate, simpler network than [`TarokNet`](network.py) — it
matches the structure of the 3p game:

  - input dim = `STATE_SIZE_3P = 513` (v2 layout: belief planes, void
    inference, centaur trick context — mirrors v10 4p features sans
    partner / king-call; fed by the Rust `encode_state_3p` encoder)
  - oracle dim = `ORACLE_STATE_SIZE_3P = 621` (adds 2 × 54 perfect opp-hand planes)
  - heads:
      * BID:      `BID_ACTION_SIZE_3P = 8`  (pass + 7 contracts)
      * TALON:    `TALON_ACTION_SIZE_3P = 6`
      * CARD:     `CARD_ACTION_SIZE_3P = 54`
      * ANNOUNCE: `ANNOUNCE_ACTION_SIZE_3P = 10`
      * **No** KING head — 3-player Tarok never calls a king.

The 3p network does **not** load 4p checkpoints. The two architectures are
intentionally incompatible: drop in a freshly-initialized TarokNet3 for any
3p training run.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn

# Sizes are sourced from the Rust engine to keep encoder/network in lockstep.
# We import lazily inside `_get_dims()` so unit tests can run without a built
# extension when sizes are passed explicitly.

DEFAULT_STATE_SIZE_3P = 513
DEFAULT_ORACLE_STATE_SIZE_3P = 621
DEFAULT_BID_ACTION_SIZE_3P = 8
DEFAULT_TALON_ACTION_SIZE_3P = 6
DEFAULT_CARD_ACTION_SIZE_3P = 54
DEFAULT_ANNOUNCE_ACTION_SIZE_3P = 10


class DecisionType3P(Enum):
    """3p decision types. King-calling is intentionally absent."""

    BID = 0
    TALON_PICK = 2  # gap at 1 keeps integer codes aligned with the 4p enum
    CARD_PLAY = 3
    ANNOUNCE = 4


def _resolve_dims(
    state_size: int | None,
    oracle_state_size: int | None,
) -> tuple[int, int]:
    """Resolve input dims, preferring values from the Rust engine if available."""
    if state_size is not None and oracle_state_size is not None:
        return state_size, oracle_state_size
    try:
        import tarok_engine as _te  # type: ignore[import-not-found]

        return (
            state_size if state_size is not None else _te.STATE_SIZE_3P,
            oracle_state_size if oracle_state_size is not None else _te.ORACLE_STATE_SIZE_3P,
        )
    except Exception:
        return (
            state_size if state_size is not None else DEFAULT_STATE_SIZE_3P,
            oracle_state_size if oracle_state_size is not None else DEFAULT_ORACLE_STATE_SIZE_3P,
        )


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class TarokNet3(nn.Module):
    """Multi-head actor-critic for 3-player Tarok.

    Architecture:
      - 2-layer MLP trunk (Linear+LN+ReLU) → 2 residual blocks
      - Per-head 2-layer MLPs for BID / TALON / CARD / ANNOUNCE
      - Optional oracle-critic backbone over `ORACLE_STATE_SIZE_3P`
    """

    def __init__(
        self,
        hidden_size: int = 256,
        oracle_critic: bool = False,
        state_size: int | None = None,
        oracle_state_size: int | None = None,
        bid_action_size: int = DEFAULT_BID_ACTION_SIZE_3P,
        talon_action_size: int = DEFAULT_TALON_ACTION_SIZE_3P,
        card_action_size: int = DEFAULT_CARD_ACTION_SIZE_3P,
        announce_action_size: int = DEFAULT_ANNOUNCE_ACTION_SIZE_3P,
    ):
        super().__init__()
        self.oracle_critic_enabled = oracle_critic
        self._hidden_size = hidden_size
        self._state_size, self._oracle_state_size = _resolve_dims(
            state_size, oracle_state_size
        )
        self._bid_action_size = bid_action_size
        self._talon_action_size = talon_action_size
        self._card_action_size = card_action_size
        self._announce_action_size = announce_action_size

        # Actor backbone (imperfect info).
        self.shared = nn.Sequential(
            nn.Linear(self._state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            _ResidualBlock(hidden_size),
            _ResidualBlock(hidden_size),
        )

        half = hidden_size // 2

        self.bid_head = nn.Sequential(
            nn.Linear(hidden_size, half), nn.ReLU(),
            nn.Linear(half, bid_action_size),
        )
        self.talon_head = nn.Sequential(
            nn.Linear(hidden_size, half), nn.ReLU(),
            nn.Linear(half, talon_action_size),
        )
        self.card_head = nn.Sequential(
            nn.Linear(hidden_size, half), nn.ReLU(),
            nn.Linear(half, card_action_size),
        )
        self.announce_head = nn.Sequential(
            nn.Linear(hidden_size, half), nn.ReLU(),
            nn.Linear(half, announce_action_size),
        )

        # Optional oracle critic (perfect info).
        if oracle_critic:
            self.critic_backbone = nn.Sequential(
                nn.Linear(self._oracle_state_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            )
            self.critic_res_blocks = nn.Sequential(_ResidualBlock(hidden_size))

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, half), nn.ReLU(),
            nn.Linear(half, 1),
        )

        self._heads = {
            DecisionType3P.BID: self.bid_head,
            DecisionType3P.TALON_PICK: self.talon_head,
            DecisionType3P.CARD_PLAY: self.card_head,
            DecisionType3P.ANNOUNCE: self.announce_head,
        }

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def oracle_state_size(self) -> int:
        return self._oracle_state_size

    def forward(
        self,
        state: torch.Tensor,
        decision_type: DecisionType3P = DecisionType3P.CARD_PLAY,
        oracle_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value).

        - `state`: (B, STATE_SIZE_3P) imperfect-info observation.
        - `decision_type`: which actor head to use.
        - `oracle_state`: (B, ORACLE_STATE_SIZE_3P), required if
          `oracle_critic=True` was set at construction; ignored otherwise.
        """
        if decision_type not in self._heads:
            raise ValueError(
                f"TarokNet3 has no head for decision_type {decision_type!r} "
                "(king-calling does not exist in 3-player Tarok)"
            )

        actor_features = self.res_blocks(self.shared(state))
        logits = self._heads[decision_type](actor_features)

        if self.oracle_critic_enabled and oracle_state is not None:
            critic_features = self.critic_backbone(oracle_state)
            critic_features = self.critic_res_blocks(critic_features)
        else:
            critic_features = actor_features

        value = self.critic(critic_features).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------
    # PPO-trainer-compatible API
    # ------------------------------------------------------------------
    #
    # The 3p network is wired into the same PPO update loop as the 4p
    # network. The trainer talks to the network through three concrete
    # operations:
    #
    #   evaluate_action(state, action, legal_mask, dt, oracle_state, gm)
    #     -> (log_probs, values, entropy)
    #
    # The trainer uses the **4-player** `DecisionType` enum (from
    # `tarok_model.encoding`) and the **4-player** legal-mask widths
    # (BID=9, TALON=6, CARD=54, ANNOUNCE=10). For 3p the bid head only
    # has 8 logits, so we right-pad to 9 with a large-negative constant
    # (the 9th slot is always illegal in the Rust 3p engine, so masking
    # would set it to -inf anyway).

    _BID_PAD_NEG: float = -1e9

    def _decision_type_3p(self, decision_type) -> "DecisionType3P":
        # Accept either the v4 `DecisionType` (from tarok_model.encoding)
        # or the native `DecisionType3P`. We compare on the integer value
        # to avoid importing the v4 enum here.
        if isinstance(decision_type, DecisionType3P):
            return decision_type
        v = getattr(decision_type, "value", decision_type)
        if v == 1:
            raise ValueError(
                "TarokNet3 received decision_type=KING_CALL, but king-calling "
                "does not exist in 3-player Tarok."
            )
        return DecisionType3P(v)

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type,
        oracle_state: torch.Tensor | None = None,
        game_mode=None,  # 3p has no mode-conditional card head
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt3 = self._decision_type_3p(decision_type)
        logits, values = self.forward(state, dt3, oracle_state)

        # Pad logits up to the legal-mask width if needed (used for BID:
        # 8 -> 9 to match the 4p superset width the trainer slices to).
        target_width = legal_mask.shape[-1]
        if logits.shape[-1] < target_width:
            pad_cols = target_width - logits.shape[-1]
            pad = torch.full(
                (*logits.shape[:-1], pad_cols),
                self._BID_PAD_NEG,
                dtype=logits.dtype,
                device=logits.device,
            )
            logits = torch.cat([logits, pad], dim=-1)
        elif logits.shape[-1] > target_width:
            logits = logits[..., :target_width]

        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = float("-inf")

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return log_probs, values, entropy

    def get_action(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type,
        oracle_state: torch.Tensor | None = None,
        game_mode=None,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Inference entry point used by `NeuralPlayer`.

        Mirrors `TarokNetV4.get_action`. Accepts the v4 `DecisionType` enum
        (or `DecisionType3P`) and pads/truncates logits to match the legal
        mask width, so the caller can pass either 3p- or 4p-shaped masks.
        """
        del game_mode
        dt3 = self._decision_type_3p(decision_type)
        logits, value = self.forward(state, dt3, oracle_state)

        target_width = legal_mask.shape[-1]
        if logits.shape[-1] < target_width:
            pad_cols = target_width - logits.shape[-1]
            pad = torch.full(
                (*logits.shape[:-1], pad_cols),
                self._BID_PAD_NEG,
                dtype=logits.dtype,
                device=logits.device,
            )
            logits = torch.cat([logits, pad], dim=-1)
        elif logits.shape[-1] > target_width:
            logits = logits[..., :target_width]

        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = float("-inf")

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(), value.squeeze(-1).squeeze()

