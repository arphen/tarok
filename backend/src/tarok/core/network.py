"""Neural network for the RL agent — multi-head actor-critic architecture.

v2 Architecture upgrades:
  - Residual blocks with LayerNorm for deeper, more stable training
  - Multi-head self-attention over card embeddings for relational reasoning
  - Oracle guiding: actor latent space aligned with oracle critic via distillation
  - Expanded state encoding with belief tracking (450 dims)

Supports five decision types with separate action heads:
  - Bidding (9 actions: pass + 8 contracts)
  - King calling (4 actions: one per suit)
  - Talon group selection (6 actions)
  - Card play (54 actions: one per card)
  - Announcements (10 actions: pass + 4 announcements + 5 kontras)

Supports an Oracle Critic mode (Perfect Training, Imperfect Execution):
  - Actor heads see only imperfect information (player's own observation)
  - Critic sees perfect information (all hands) during training
  - At deployment, the oracle critic is discarded
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tarok.core.encoding import (
    STATE_SIZE,
    ORACLE_STATE_SIZE,
    BID_ACTION_SIZE,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
    CARD_ACTION_SIZE,
    ANNOUNCE_ACTION_SIZE,
    DecisionType,
    GameMode,
)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → ReLU → Linear → Add."""

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


class CardAttention(nn.Module):
    """Multi-head self-attention over card-level features.

    Splits the 54-card belief vectors into per-card tokens, applies
    self-attention to capture card-card relationships (e.g., suit
    correlations, void inference), then projects back.
    """

    def __init__(self, num_cards: int = 54, card_dim: int = 4, num_heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.num_cards = num_cards
        self.card_dim = card_dim
        # Project each card's features into attention space
        self.card_proj = nn.Linear(card_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, card_features: torch.Tensor) -> torch.Tensor:
        """card_features: (batch, num_cards, card_dim) → (batch, hidden_dim)"""
        tokens = self.card_proj(card_features)  # (B, 54, hidden_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = self.norm(attn_out)
        # Global average pooling over card tokens
        pooled = attn_out.mean(dim=1)  # (B, hidden_dim)
        return self.out_proj(pooled)


class TarokNet(nn.Module):
    """Multi-head Actor-Critic network for all Tarok decisions.

    v2 architecture features:
    - Residual backbone with 2 residual blocks
    - Card-level multi-head self-attention over belief vectors
    - Oracle guiding support (shared latent space distillation)

    When oracle_critic=True, the critic uses a separate backbone that
    takes the full perfect-information state (all hands visible).
    """

    def __init__(self, hidden_size: int = 256, oracle_critic: bool = False):
        super().__init__()
        self.oracle_critic_enabled = oracle_critic
        self._hidden_size = hidden_size

        # Actor backbone (imperfect info — only sees own hand)
        # Input projection + 2 residual blocks
        self.shared = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
        )

        # Card-level attention over belief vectors
        # Each card has: own_hand(1) + played(1) + current_trick(1) +
        #                belief_opp1(1) + belief_opp2(1) + belief_opp3(1) = 6 features
        self.card_attention = CardAttention(
            num_cards=54, card_dim=6, num_heads=4, hidden_dim=hidden_size // 4,
        )
        # Fuse attention output with backbone
        attn_hidden = hidden_size // 4
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size + attn_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
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
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            )
            self.critic_res_blocks = nn.Sequential(
                ResidualBlock(hidden_size),
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

    def _extract_card_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract per-card feature matrix from the flat state tensor.

        Returns (batch, 54, 6) tensor:
          channel 0: own hand
          channel 1: played cards
          channel 2: current trick cards
          channel 3-5: belief probabilities for opponents 1-3
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]

        hand = state[:, 0:54]           # own hand
        played = state[:, 54:108]       # played cards
        trick = state[:, 108:162]       # current trick
        # Belief vectors start at offset 270 (after v1 features + role)
        # v1 features = 270, beliefs at 270:270+162
        belief_start = 270  # _OLD_STATE_SIZE_V1
        if state.shape[1] > belief_start + 162:
            b1 = state[:, belief_start:belief_start + 54]
            b2 = state[:, belief_start + 54:belief_start + 108]
            b3 = state[:, belief_start + 108:belief_start + 162]
        else:
            # Old-format state without beliefs — use zeros
            b1 = torch.zeros(B, 54, device=state.device)
            b2 = torch.zeros(B, 54, device=state.device)
            b3 = torch.zeros(B, 54, device=state.device)

        # Stack: (B, 54, 6)
        return torch.stack([hand, played, trick, b1, b2, b3], dim=-1)

    # ------------------------------------------------------------------
    # Auto-migration: pad old checkpoints to new dimensions
    # Supports: 267→270→450 state size, no-LN→LN, no-residual→residual
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict=True, assign=False):
        # --- Migrate old 2-layer (no LayerNorm) checkpoints to new layout ---
        _LAYER_REMAP = {
            "shared.2.weight": "shared.3.weight",
            "shared.2.bias": "shared.3.bias",
        }
        _CRITIC_REMAP = {
            "critic_backbone.2.weight": "critic_backbone.3.weight",
            "critic_backbone.2.bias": "critic_backbone.3.bias",
        }
        needs_ln_migration = "shared.0.weight" in state_dict and "shared.1.weight" not in state_dict
        if needs_ln_migration:
            new_sd = {}
            for key, val in state_dict.items():
                new_key = _LAYER_REMAP.get(key, _CRITIC_REMAP.get(key, key))
                new_sd[new_key] = val
            state_dict = new_sd
            strict = False

        # --- Pad input dimension for expanded state encoding ---
        for key in ["shared.0.weight", "critic_backbone.0.weight"]:
            if key not in state_dict:
                continue
            w = state_dict[key]
            expected = ORACLE_STATE_SIZE if "critic" in key else STATE_SIZE
            if w.shape[1] < expected:
                pad = torch.zeros(w.shape[0], expected - w.shape[1], dtype=w.dtype, device=w.device)
                state_dict[key] = torch.cat([w, pad], dim=1)

        # --- Allow missing keys for new v2 modules (res_blocks, card_attention, fuse, critic_res_blocks) ---
        has_new_modules = any(k.startswith("res_blocks.") for k in state_dict)
        if not has_new_modules:
            strict = False

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        state: torch.Tensor,
        decision_type: DecisionType = DecisionType.CARD_PLAY,
        oracle_state: torch.Tensor | None = None,
        game_mode: GameMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Backbone + residual blocks
        shared = self.shared(state)
        shared = self.res_blocks(shared)

        # Card-level attention over belief vectors
        card_feats = self._extract_card_features(state)
        attn_out = self.card_attention(card_feats)

        # Fuse backbone + attention
        fused = self.fuse(torch.cat([shared, attn_out], dim=-1))

        if decision_type == DecisionType.CARD_PLAY and hasattr(self, "_card_logits_for_mode"):
            logits = self._card_logits_for_mode(fused, state, game_mode)
        else:
            logits = self._heads[decision_type](fused)

        # Critic: use oracle backbone if available and oracle_state provided
        if self.oracle_critic_enabled and oracle_state is not None:
            critic_features = self.critic_backbone(oracle_state)
            if hasattr(self, 'critic_res_blocks'):
                critic_features = self.critic_res_blocks(critic_features)
        else:
            critic_features = fused

        value = self.critic(critic_features)
        return logits, value

    def get_actor_features(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the actor's latent representation (for oracle guiding distillation)."""
        shared = self.shared(state)
        shared = self.res_blocks(shared)
        card_feats = self._extract_card_features(state)
        attn_out = self.card_attention(card_feats)
        return self.fuse(torch.cat([shared, attn_out], dim=-1))

    def get_critic_features(
        self,
        oracle_state: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the oracle critic's latent representation (for oracle guiding distillation)."""
        features = self.critic_backbone(oracle_state)
        if hasattr(self, 'critic_res_blocks'):
            features = self.critic_res_blocks(features)
        return features

    def get_action(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType = DecisionType.CARD_PLAY,
        oracle_state: torch.Tensor | None = None,
        game_mode: GameMode | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select an action from legal moves. Returns (action_idx, log_prob, value)."""
        logits, value = self(state, decision_type, oracle_state, game_mode)
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
        game_mode: GameMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a batch of actions. Returns (log_probs, values, entropy)."""
        logits, values = self(state, decision_type, oracle_state, game_mode)
        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = float("-inf")

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def forward_batch(
        self,
        states: torch.Tensor,
        decision_types: list[DecisionType],
        oracle_states: torch.Tensor | None = None,
        game_modes: list[GameMode] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched forward for mixed decision types.

        Runs the shared backbone once for the entire batch, then routes
        each sample to its decision-type head.  Returns per-sample logits
        padded to CARD_ACTION_SIZE (the max head size) and values.
        """
        B = states.shape[0]
        # Shared backbone (one pass for the whole batch)
        shared = self.shared(states)
        shared = self.res_blocks(shared)
        card_feats = self._extract_card_features(states)
        attn_out = self.card_attention(card_feats)
        fused = self.fuse(torch.cat([shared, attn_out], dim=-1))

        # Critic
        if self.oracle_critic_enabled and oracle_states is not None:
            critic_features = self.critic_backbone(oracle_states)
            if hasattr(self, 'critic_res_blocks'):
                critic_features = self.critic_res_blocks(critic_features)
        else:
            critic_features = fused
        values = self.critic(critic_features).squeeze(-1)  # (B,)

        # Route to decision heads by grouping samples of the same type
        max_action_size = CARD_ACTION_SIZE  # 54 — largest head
        all_logits = torch.full(
            (B, max_action_size), float("-inf"),
            device=states.device, dtype=states.dtype,
        )

        # Build index groups per decision type
        type_indices: dict[DecisionType, list[int]] = {}
        for i, dt in enumerate(decision_types):
            type_indices.setdefault(dt, []).append(i)

        for dt, idxs in type_indices.items():
            idx_t = torch.tensor(idxs, device=states.device)
            head_input = fused[idx_t]
            if dt == DecisionType.CARD_PLAY and hasattr(self, "_card_logits_for_mode"):
                mode_subset = [game_modes[i] for i in idxs] if game_modes is not None else None
                head_logits = self._card_logits_for_mode(head_input, states[idx_t], mode_subset)
            else:
                head_logits = self._heads[dt](head_input)  # (n, head_action_size)
            # Write into the correct rows, left-aligned
            all_logits[idx_t, :head_logits.shape[1]] = head_logits

        return all_logits, values

    def card_logits_for_export(self, fused: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Card logits used by TorchScript export wrappers."""
        return self.card_head(fused)


class TarokNetV3(TarokNet):
    """v3 actor-critic: card-play head split by game mode family.

    Heads:
      - solo
      - klop_berac
      - partner_play
      - color_valat
    """

    _CONTRACT_FEATURE_OFFSET = 220
    _KLOP_IDX = 0
    _THREE_IDX = 1
    _TWO_IDX = 2
    _ONE_IDX = 3
    _SOLO_THREE_IDX = 4
    _SOLO_TWO_IDX = 5
    _SOLO_ONE_IDX = 6
    _SOLO_IDX = 7
    _BERAC_IDX = 8
    _BARVNI_VALAT_IDX = 9

    _SOLO_CONTRACT_INDICES = {
        _SOLO_THREE_IDX,
        _SOLO_TWO_IDX,
        _SOLO_ONE_IDX,
        _SOLO_IDX,
    }
    _KLOP_BERAC_CONTRACT_INDICES = {
        _KLOP_IDX,
        _BERAC_IDX,
    }

    def __init__(self, hidden_size: int = 256, oracle_critic: bool = False):
        super().__init__(hidden_size=hidden_size, oracle_critic=oracle_critic)
        half = hidden_size // 2
        self.card_heads = nn.ModuleDict({
            GameMode.SOLO.value: nn.Sequential(
                nn.Linear(hidden_size, half),
                nn.ReLU(),
                nn.Linear(half, CARD_ACTION_SIZE),
            ),
            GameMode.KLOP_BERAC.value: nn.Sequential(
                nn.Linear(hidden_size, half),
                nn.ReLU(),
                nn.Linear(half, CARD_ACTION_SIZE),
            ),
            GameMode.PARTNER_PLAY.value: nn.Sequential(
                nn.Linear(hidden_size, half),
                nn.ReLU(),
                nn.Linear(half, CARD_ACTION_SIZE),
            ),
            GameMode.COLOR_VALAT.value: nn.Sequential(
                nn.Linear(hidden_size, half),
                nn.ReLU(),
                nn.Linear(half, CARD_ACTION_SIZE),
            ),
        })

    def load_state_dict(self, state_dict, strict=True, assign=False):
        has_mode_heads = any(k.startswith("card_heads.") for k in state_dict)
        if not has_mode_heads and "card_head.0.weight" in state_dict:
            state_dict = dict(state_dict)
            for mode_key in self.card_heads.keys():
                for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
                    src = f"card_head.{suffix}"
                    dst = f"card_heads.{mode_key}.{suffix}"
                    if src in state_dict and dst not in state_dict:
                        state_dict[dst] = state_dict[src].clone()
            strict = False
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _mode_from_contract_idx(contract_idx: int) -> GameMode:
        if contract_idx in TarokNetV3._SOLO_CONTRACT_INDICES:
            return GameMode.SOLO
        if contract_idx in TarokNetV3._KLOP_BERAC_CONTRACT_INDICES:
            return GameMode.KLOP_BERAC
        if contract_idx == TarokNetV3._BARVNI_VALAT_IDX:
            return GameMode.COLOR_VALAT
        return GameMode.PARTNER_PLAY

    def _infer_modes_from_state(self, state: torch.Tensor) -> list[GameMode]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        c0 = self._CONTRACT_FEATURE_OFFSET
        c1 = c0 + 10
        if state.shape[1] < c1:
            return [GameMode.PARTNER_PLAY for _ in range(state.shape[0])]

        contract_slice = state[:, c0:c1]
        max_vals, max_idx = torch.max(contract_slice, dim=1)
        modes: list[GameMode] = []
        for i in range(state.shape[0]):
            if float(max_vals[i].item()) <= 0.0:
                modes.append(GameMode.PARTNER_PLAY)
                continue
            idx = int(max_idx[i].item())
            modes.append(self._mode_from_contract_idx(idx))
        return modes

    def _resolve_modes(
        self,
        state: torch.Tensor,
        game_mode: GameMode | list[GameMode] | tuple[GameMode, ...] | None,
    ) -> list[GameMode]:
        if game_mode is None:
            return self._infer_modes_from_state(state)
        if isinstance(game_mode, GameMode):
            n = state.shape[0] if state.dim() > 1 else 1
            return [game_mode for _ in range(n)]
        return list(game_mode)

    def _card_logits_for_mode(
        self,
        fused: torch.Tensor,
        state: torch.Tensor,
        game_mode: GameMode | list[GameMode] | tuple[GameMode, ...] | None,
    ) -> torch.Tensor:
        if fused.dim() == 1:
            fused = fused.unsqueeze(0)
        modes = self._resolve_modes(state, game_mode)
        logits = torch.empty((fused.shape[0], CARD_ACTION_SIZE), dtype=fused.dtype, device=fused.device)

        groups: dict[GameMode, list[int]] = {}
        for i, gm in enumerate(modes):
            groups.setdefault(gm, []).append(i)

        for gm, idxs in groups.items():
            idx_t = torch.tensor(idxs, device=fused.device)
            logits[idx_t] = self.card_heads[gm.value](fused[idx_t])
        return logits

    def card_logits_for_export(self, fused: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Trace-safe path for TorchScript export: tensor-only mode routing.
        if state.dim() == 1:
            state = state.unsqueeze(0)
        c0 = self._CONTRACT_FEATURE_OFFSET
        c1 = c0 + 10
        contract_slice = state[:, c0:c1]
        max_vals, max_idx = torch.max(contract_slice, dim=1)

        # mode ids: 0=solo, 1=klop_berac, 2=partner_play, 3=color_valat
        mode_ids = torch.full_like(max_idx, 2)
        solo_mask = (
            (max_idx == self._SOLO_THREE_IDX)
            | (max_idx == self._SOLO_TWO_IDX)
            | (max_idx == self._SOLO_ONE_IDX)
            | (max_idx == self._SOLO_IDX)
        )
        klop_mask = (max_idx == self._KLOP_IDX) | (max_idx == self._BERAC_IDX)
        color_mask = max_idx == self._BARVNI_VALAT_IDX

        mode_ids = torch.where(klop_mask, torch.ones_like(mode_ids), mode_ids)
        mode_ids = torch.where(color_mask, torch.full_like(mode_ids, 3), mode_ids)
        mode_ids = torch.where(solo_mask, torch.zeros_like(mode_ids), mode_ids)
        mode_ids = torch.where(max_vals <= 0.0, torch.full_like(mode_ids, 2), mode_ids)

        solo_logits = self.card_heads[GameMode.SOLO.value](fused)
        klop_logits = self.card_heads[GameMode.KLOP_BERAC.value](fused)
        partner_logits = self.card_heads[GameMode.PARTNER_PLAY.value](fused)
        color_logits = self.card_heads[GameMode.COLOR_VALAT.value](fused)

        stacked = torch.stack([solo_logits, klop_logits, partner_logits, color_logits], dim=1)
        gather_idx = mode_ids.view(-1, 1, 1).expand(-1, 1, CARD_ACTION_SIZE)
        return stacked.gather(1, gather_idx).squeeze(1)
