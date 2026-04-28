"""Neural network PlayerPort implementation (inference-only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tarok.entities import Announcement, Card, CardType, Contract, GameState
from tarok_model.encoding import (
    ANNOUNCE_PASS,
    BID_ACTIONS,
    DecisionType,
    GameMode,
    KING_ACTIONS,
    card_idx_to_card,
    contract_to_game_mode,
    encode_bid_mask,
    encode_king_mask,
    encode_legal_mask,
    encode_state,
    encode_talon_mask,
)
from tarok_model.network import TarokNetV4
from tarok_model.network_3p import TarokNet3


class NeuralPlayer:
    """Inference-only neural network player for Tarok.

    Supports both 4-player (`model_arch == "v4"` → `TarokNetV4`) and
    3-player (`model_arch == "v3p"` → `TarokNet3`) checkpoints. The `variant`
    flag tells the game loop which Rust state encoder to feed in via the fast
    tensor path; PlayerPort callers go through Python-state helpers that are
    variant-agnostic.
    """

    def __init__(
        self,
        name: str = "NN-Player",
        hidden_size: int = 256,
        device: str = "cpu",
        oracle_critic: bool = False,
        mode_heads: bool = True,
        variant: str = "four_player",
    ):
        self._name = name
        self.device = torch.device(device)
        del mode_heads
        self.variant = variant
        if variant == "three_player":
            self.network = TarokNet3(hidden_size=hidden_size, oracle_critic=oracle_critic).to(
                self.device
            )
        else:
            self.network = TarokNetV4(hidden_size, oracle_critic=oracle_critic).to(self.device)
        self.network.eval()

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_three_player(self) -> bool:
        return self.variant == "three_player"

    def set_training(self, training: bool) -> None:
        self.network.train(training)

    @staticmethod
    def from_checkpoint(
        path: str | Path,
        name: str = "NN-Player",
        device: str = "cpu",
        oracle_critic: bool = False,
    ) -> "NeuralPlayer":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model_arch = ckpt.get("model_arch")
        state_dict = ckpt["model_state_dict"]
        if model_arch == "v3p":
            hidden_size = state_dict["shared.0.weight"].shape[0]
            has_oracle = any(k.startswith("critic_backbone.") for k in state_dict)
            player = NeuralPlayer(
                name=name,
                hidden_size=hidden_size,
                device=device,
                oracle_critic=has_oracle or oracle_critic,
                variant="three_player",
            )
            player.network.load_state_dict(state_dict)
            return player
        if model_arch != "v4":
            raise ValueError(
                f"Unsupported checkpoint architecture '{model_arch}'. "
                "Expected 'v4' (4-player) or 'v3p' (3-player)."
            )
        hidden_size = state_dict["shared.0.weight"].shape[0]
        has_oracle = any(k.startswith("critic_backbone.") for k in state_dict)
        player = NeuralPlayer(
            name=name,
            hidden_size=hidden_size,
            device=device,
            oracle_critic=has_oracle or oracle_critic,
            mode_heads=True,
            variant="four_player",
        )
        player.network.load_state_dict(state_dict)
        return player

    def _decide_from_tensors(
        self,
        state_tensor: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType,
        oracle_tensor: torch.Tensor | None = None,
        game_mode: GameMode | None = None,
    ) -> int:
        del oracle_tensor
        state_tensor = state_tensor.to(self.device)
        mask = legal_mask.to(self.device)
        with torch.no_grad():
            action_idx, _log_prob, _value = self.network.get_action(
                state_tensor.unsqueeze(0), mask.unsqueeze(0), decision_type, game_mode=game_mode
            )
        return action_idx

    def _decide(
        self,
        state: GameState,
        player_idx: int,
        legal_mask: torch.Tensor,
        decision_type: DecisionType,
    ) -> int:
        if self.is_three_player:
            # Use the Rust 3p encoder when running through PlayerPort calls.
            rust_gs = getattr(state, "_rust_gs", None)
            if rust_gs is None:
                raise RuntimeError(
                    "3-player NeuralPlayer requires a Rust-backed GameState (state._rust_gs)."
                )
            arr = rust_gs.encode_state_3p(player_idx, decision_type.value)
            state_tensor = torch.from_numpy(np.asarray(arr)).float().to(self.device)
        else:
            state_tensor = encode_state(state, player_idx, decision_type).to(self.device)
        mask = legal_mask.to(self.device)
        game_mode = (
            contract_to_game_mode(state.contract)
            if decision_type == DecisionType.CARD_PLAY
            else None
        )
        with torch.no_grad():
            action_idx, _log_prob, _value = self.network.get_action(
                state_tensor.unsqueeze(0), mask.unsqueeze(0), decision_type, game_mode=game_mode
            )
        return action_idx

    async def choose_bid(
        self, state: GameState, player_idx: int, legal_bids: list[Contract | None]
    ) -> Contract | None:
        mask = encode_bid_mask(legal_bids)
        action_idx = self._decide(state, player_idx, mask, DecisionType.BID)
        bid = BID_ACTIONS[action_idx]
        if bid not in legal_bids:
            bid = None
        return bid

    async def choose_king(
        self, state: GameState, player_idx: int, callable_kings: list[Card]
    ) -> Card:
        mask = encode_king_mask(callable_kings)
        action_idx = self._decide(state, player_idx, mask, DecisionType.KING_CALL)
        chosen_suit = KING_ACTIONS[action_idx]
        for king in callable_kings:
            if king.suit == chosen_suit:
                return king
        return callable_kings[0]

    async def choose_talon_group(
        self, state: GameState, player_idx: int, talon_groups: list[list[Card]]
    ) -> int:
        mask = encode_talon_mask(len(talon_groups))
        action_idx = self._decide(state, player_idx, mask, DecisionType.TALON_PICK)
        if action_idx >= len(talon_groups):
            action_idx = 0
        return action_idx

    async def choose_discard(
        self, state: GameState, player_idx: int, must_discard: int
    ) -> list[Card]:
        hand = state.hands[player_idx]
        suit_cards = [c for c in hand if not c.is_king and c.card_type != CardType.TAROK]
        suit_cards.sort(key=lambda c: c.points)
        if len(suit_cards) >= must_discard:
            return suit_cards[:must_discard]
        taroks = [c for c in hand if c.card_type == CardType.TAROK and not c.is_king]
        taroks.sort(key=lambda c: c.points)
        return (suit_cards + taroks)[:must_discard]

    async def choose_announcements(self, state: GameState, player_idx: int) -> list[Announcement]:
        return []

    async def choose_announce_action(self, state: GameState, player_idx: int) -> int:
        return ANNOUNCE_PASS

    async def choose_card(self, state: GameState, player_idx: int, legal_plays: list[Card]) -> Card:
        mask = encode_legal_mask(legal_plays)
        action_idx = self._decide(state, player_idx, mask, DecisionType.CARD_PLAY)
        card = card_idx_to_card(action_idx)
        if card not in legal_plays:
            card = legal_plays[0]
        return card
