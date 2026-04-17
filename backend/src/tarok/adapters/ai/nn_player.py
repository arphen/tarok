"""RL Agent adapter — inference-only neural network player.

All decisions (bid, king call, talon pick, card play) go through the network.
Discard remains heuristic (combinatorial action space too large for a single head).

Training (PPO, experience collection, compute backends) lives in training-lab/.
"""

from __future__ import annotations

import torch

from tarok.entities import Card, CardType, Suit, Announcement, Contract, GameState, KontraLevel
from tarok_model.network import TarokNetV4
from tarok_model.encoding import (
    DecisionType,
    GameMode,
    contract_to_game_mode,
    encode_state,
    encode_legal_mask,
    encode_bid_mask,
    encode_king_mask,
    encode_talon_mask,
    card_idx_to_card,
    CARD_TO_IDX,
    BID_ACTIONS,
    BID_TO_IDX,
    KING_ACTIONS,
    SUIT_TO_IDX,
    ANNOUNCE_PASS,
    ANNOUNCE_IDX_TO_ANN,
    KONTRA_IDX_TO_KEY,
)


class RLAgent:
    """Inference-only neural network player for Tarok.

    Loads a trained checkpoint and plays using greedy policy (argmax over
    legal actions).  All game decisions (bidding, king calling, talon
    selection, card play) go through the network.
    """

    def __init__(
        self,
        name: str = "RL-Agent",
        hidden_size: int = 256,
        device: str = "cpu",
        oracle_critic: bool = False,
        mode_heads: bool = True,
    ):
        self._name = name
        self.device = torch.device(device)
        del mode_heads
        self.network = TarokNetV4(hidden_size, oracle_critic=oracle_critic).to(self.device)
        self.network.eval()

    @property
    def name(self) -> str:
        return self._name

    def set_training(self, training: bool) -> None:
        """No-op kept for backward compatibility with server code."""
        self.network.train(training)

    @staticmethod
    def from_checkpoint(
        path: str | "Path",
        name: str = "RL-Agent",
        device: str = "cpu",
        oracle_critic: bool = False,
    ) -> "RLAgent":
        """Create an RLAgent with hidden_size inferred from checkpoint weights."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model_arch = ckpt.get("model_arch")
        if model_arch != "v4":
            raise ValueError(
                f"Unsupported checkpoint architecture '{model_arch}'. Only 'v4' checkpoints are supported."
            )
        state_dict = ckpt["model_state_dict"]
        # Infer hidden_size from the first layer of the shared backbone
        hidden_size = state_dict["shared.0.weight"].shape[0]
        # Detect oracle critic from checkpoint keys
        has_oracle = any(k.startswith("critic_backbone.") for k in state_dict)
        agent = RLAgent(
            name=name,
            hidden_size=hidden_size,
            device=device,
            oracle_critic=has_oracle or oracle_critic,
            mode_heads=True,
        )
        agent.network.load_state_dict(state_dict)
        return agent

    # ------------------------------------------------------------------
    # Generic decision helper
    # ------------------------------------------------------------------

    def _decide_from_tensors(
        self,
        state_tensor: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType,
        oracle_tensor: torch.Tensor | None = None,
        game_mode: GameMode | None = None,
    ) -> int:
        """Greedy decision from pre-encoded tensors (for Rust engine)."""
        state_tensor = state_tensor.to(self.device)
        mask = legal_mask.to(self.device)

        with torch.no_grad():
            action_idx, _log_prob, _value = self.network.get_action(
                state_tensor.unsqueeze(0), mask.unsqueeze(0), decision_type,
                game_mode=game_mode,
            )
        return action_idx

    def _decide(
        self,
        state: GameState,
        player_idx: int,
        legal_mask: torch.Tensor,
        decision_type: DecisionType,
    ) -> int:
        """Run the network for *decision_type*, return action index."""
        state_tensor = encode_state(state, player_idx, decision_type).to(self.device)
        mask = legal_mask.to(self.device)

        # Derive game mode from contract for mode-specific card heads
        game_mode = contract_to_game_mode(state.contract) if decision_type == DecisionType.CARD_PLAY else None

        with torch.no_grad():
            action_idx, _log_prob, _value = self.network.get_action(
                state_tensor.unsqueeze(0), mask.unsqueeze(0), decision_type,
                game_mode=game_mode,
            )
        return action_idx

    # ------------------------------------------------------------------
    # PlayerPort implementation
    # ------------------------------------------------------------------

    async def choose_bid(
        self, state: GameState, player_idx: int, legal_bids: list[Contract | None]
    ) -> Contract | None:
        mask = encode_bid_mask(legal_bids)
        action_idx = self._decide(state, player_idx, mask, DecisionType.BID)
        bid = BID_ACTIONS[action_idx]
        # Safety: make sure the decoded bid is actually legal
        if bid not in legal_bids:
            bid = None  # fall back to pass
        return bid

    async def choose_king(
        self, state: GameState, player_idx: int, callable_kings: list[Card]
    ) -> Card:
        mask = encode_king_mask(callable_kings)
        action_idx = self._decide(state, player_idx, mask, DecisionType.KING_CALL)
        chosen_suit = KING_ACTIONS[action_idx]
        # Find the king (or queen) matching that suit
        for king in callable_kings:
            if king.suit == chosen_suit:
                return king
        return callable_kings[0]  # safety fallback

    async def choose_talon_group(
        self, state: GameState, player_idx: int, talon_groups: list[list[Card]]
    ) -> int:
        mask = encode_talon_mask(len(talon_groups))
        action_idx = self._decide(state, player_idx, mask, DecisionType.TALON_PICK)
        if action_idx >= len(talon_groups):
            action_idx = 0  # safety fallback
        return action_idx

    async def choose_discard(
        self, state: GameState, player_idx: int, must_discard: int
    ) -> list[Card]:
        """Heuristic discard — kept out of RL due to combinatorial action space."""
        hand = state.hands[player_idx]
        # Prefer discarding non-king suit cards (sorted cheapest first)
        suit_cards = [
            c for c in hand if not c.is_king and c.card_type != CardType.TAROK
        ]
        suit_cards.sort(key=lambda c: c.points)
        if len(suit_cards) >= must_discard:
            return suit_cards[:must_discard]
        # Not enough suit cards — take all suit cards first, then fill with taroks
        # (taroks are only legal to discard when no suit cards remain in hand after)
        taroks = [c for c in hand if c.card_type == CardType.TAROK and not c.is_king]
        taroks.sort(key=lambda c: c.points)
        result = suit_cards + taroks
        return result[:must_discard]

    async def choose_announcements(
        self, state: GameState, player_idx: int
    ) -> list[Announcement]:
        return []

    async def choose_announce_action(
        self, state: GameState, player_idx: int
    ) -> int:
        """Choose a single announcement action (0=pass, 1-4=announce, 5-9=kontra).

        Called repeatedly by the game loop until the player passes.
        Always pass — the ANNOUNCE head is not trained during self-play so
        its weights are essentially random.  Using it in spectate / play
        causes absurd valat and pagat ultimo calls every game.
        """
        return ANNOUNCE_PASS

    async def choose_card(
        self, state: GameState, player_idx: int, legal_plays: list[Card]
    ) -> Card:
        mask = encode_legal_mask(legal_plays)
        action_idx = self._decide(state, player_idx, mask, DecisionType.CARD_PLAY)
        card = card_idx_to_card(action_idx)
        if card not in legal_plays:
            card = legal_plays[0]
        return card
