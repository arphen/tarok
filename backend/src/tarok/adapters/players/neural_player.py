"""Neural network PlayerPort implementation (inference-only)."""

from __future__ import annotations

from pathlib import Path

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


class NeuralPlayer:
    """Inference-only neural network player for Tarok."""

    def __init__(
        self,
        name: str = "NN-Player",
        hidden_size: int = 256,
        device: str = "cpu",
        oracle_critic: bool = False,
        mode_heads: bool = True,
    ):
        self._name = name
        self.device = torch.device(device)
        del mode_heads
        self._jit: torch.jit.ScriptModule | None = None
        self.network = TarokNetV4(hidden_size, oracle_critic=oracle_critic).to(self.device)
        self.network.eval()

    @property
    def name(self) -> str:
        return self._name

    def set_training(self, training: bool) -> None:
        if self._jit is not None:
            self._jit.train(training)
            return
        self.network.train(training)

    @staticmethod
    def _from_torchscript(script: torch.jit.ScriptModule, name: str, device: str) -> "NeuralPlayer":
        """Load from arena-style traced TorchScript (forward → bid/king/talon/card logits + value)."""
        self = NeuralPlayer.__new__(NeuralPlayer)
        self._name = name
        self.device = torch.device(device)
        self._jit = script.to(self.device)
        self._jit.eval()
        self.network = None  # type: ignore[assignment]
        return self

    def _jit_pick_action(
        self,
        state_tensor: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType,
        game_mode: GameMode | None = None,
    ) -> int:
        """Masked softmax sample over logits; matches TarokNetV4.get_action for traced _AllHeads exports."""
        del game_mode  # traced arena models use a single card head (no per-mode routing)
        if decision_type == DecisionType.ANNOUNCE:
            return ANNOUNCE_PASS

        with torch.no_grad():
            x = state_tensor.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            assert self._jit is not None
            out = self._jit(x)
            if not isinstance(out, (tuple, list)) or len(out) != 5:
                raise ValueError(
                    "TorchScript model must return (bid, king, talon, card, value) like arena export."
                )
            bid_l, king_l, talon_l, card_l, _value = out
            if decision_type == DecisionType.BID:
                logits = bid_l
            elif decision_type == DecisionType.KING_CALL:
                logits = king_l
            elif decision_type == DecisionType.TALON_PICK:
                logits = talon_l
            elif decision_type == DecisionType.CARD_PLAY:
                logits = card_l
            else:
                logits = bid_l

            logits = logits.squeeze(0)
            mask = legal_mask.to(logits.device)
            if mask.dim() > logits.dim():
                mask = mask.squeeze(0)
            masked = logits.clone()
            masked[mask == 0] = float("-inf")
            probs = torch.softmax(masked, dim=-1)
            dist = torch.distributions.Categorical(probs)
            return int(dist.sample().item())

    @staticmethod
    def from_checkpoint(
        path: str | Path,
        name: str = "NN-Player",
        device: str = "cpu",
        oracle_critic: bool = False,
    ) -> "NeuralPlayer":
        # Trusted local checkpoints; weights_only=True raises on TorchScript-style zips (PyTorch 2.6+).
        loaded = torch.load(path, map_location=device, weights_only=False)
        if isinstance(loaded, torch.jit.ScriptModule):
            return NeuralPlayer._from_torchscript(loaded, name, device)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Unsupported checkpoint type {type(loaded).__name__}; "
                "expected a v4 training dict or TorchScript module."
            )
        model_arch = loaded.get("model_arch")
        if model_arch != "v4":
            raise ValueError(
                f"Unsupported checkpoint architecture '{model_arch}'. Only 'v4' checkpoints are supported."
            )
        state_dict = loaded["model_state_dict"]
        hidden_size = state_dict["shared.0.weight"].shape[0]
        has_oracle = any(k.startswith("critic_backbone.") for k in state_dict)
        player = NeuralPlayer(
            name=name,
            hidden_size=hidden_size,
            device=device,
            oracle_critic=has_oracle or oracle_critic,
            mode_heads=True,
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
        if self._jit is not None:
            return self._jit_pick_action(
                state_tensor, legal_mask, decision_type, game_mode=game_mode
            )
        assert self.network is not None
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
        state_tensor = encode_state(state, player_idx, decision_type).to(self.device)
        mask = legal_mask.to(self.device)
        game_mode = (
            contract_to_game_mode(state.contract)
            if decision_type == DecisionType.CARD_PLAY
            else None
        )
        if self._jit is not None:
            return self._jit_pick_action(state_tensor, mask, decision_type, game_mode=game_mode)
        assert self.network is not None
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
