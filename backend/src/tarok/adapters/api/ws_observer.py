"""WebSocket observer — broadcasts game events to connected clients."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket

from tarok.entities.card import Card
from tarok.entities.game_state import Contract, GameState, Trick


def _card_to_dict(card: Card) -> dict:
    return {
        "card_type": card.card_type.value,
        "value": card.value,
        "suit": card.suit.value if card.suit else None,
        "label": card.label,
        "points": card.points,
    }


def _state_for_player(state: GameState, player_idx: int, player_names: list[str]) -> dict:
    from tarok.entities.game_state import Phase

    is_current = state.current_player == player_idx

    # Phase-appropriate legal actions
    legal_plays: list[dict] = []
    legal_bids: list[int | None] | None = None
    callable_kings: list[dict] | None = None
    must_discard: int = 0

    if state.phase == Phase.BIDDING and is_current:
        legal_bids = [
            b.value if b is not None else None
            for b in state.legal_bids(player_idx)
        ]
    elif state.phase == Phase.KING_CALLING and is_current:
        callable_kings = [_card_to_dict(k) for k in state.callable_kings()]
    elif state.phase == Phase.TALON_EXCHANGE and is_current:
        if state.contract and state.contract.talon_cards > 0:
            expected_hand_size = 12 + state.contract.talon_cards
            if len(state.hands[player_idx]) > 12:
                must_discard = len(state.hands[player_idx]) - 12
    elif state.phase == Phase.TRICK_PLAY and is_current:
        legal_plays = [_card_to_dict(c) for c in state.legal_plays(player_idx)]

    return {
        "phase": state.phase.value,
        "hand": [_card_to_dict(c) for c in state.hands[player_idx]],
        "hand_sizes": [len(h) for h in state.hands],
        "talon_groups": (
            [[_card_to_dict(c) for c in g] for g in state.talon_revealed]
            if state.talon_revealed
            else None
        ),
        "bids": [
            {"player": b.player, "contract": b.contract.value if b.contract else None}
            for b in state.bids
        ],
        "contract": state.contract.value if state.contract else None,
        "declarer": state.declarer,
        "called_king": _card_to_dict(state.called_king) if state.called_king else None,
        "partner_revealed": state.is_partner_revealed,
        "partner": state.partner if state.is_partner_revealed else None,
        "current_trick": (
            [(p, _card_to_dict(c)) for p, c in state.current_trick.cards]
            if state.current_trick
            else []
        ),
        "tricks_played": state.tricks_played,
        "current_player": state.current_player,
        "scores": state.scores if state.scores else None,
        "legal_plays": legal_plays,
        "legal_bids": legal_bids,
        "callable_kings": callable_kings,
        "must_discard": must_discard,
        "player_names": player_names,
    }


class WebSocketObserver:
    """Broadcasts game events to a connected WebSocket client."""

    def __init__(self, ws: WebSocket, player_idx: int, player_names: list[str], ai_delay: float = 1.0):
        self._ws = ws
        self._player_idx = player_idx
        self._player_names = player_names
        self.ai_delay = ai_delay

    async def _send(self, event: str, data: Any, state: GameState) -> None:
        msg = {
            "event": event,
            "data": data,
            "state": _state_for_player(state, self._player_idx, self._player_names),
        }
        await self._ws.send_json(msg)

    async def on_game_start(self, state: GameState) -> None:
        await self._send("game_start", {}, state)

    async def on_deal(self, state: GameState) -> None:
        await self._send("deal", {}, state)

    async def on_bid(self, player: int, bid: Contract | None, state: GameState) -> None:
        await self._send("bid", {
            "player": player,
            "contract": bid.value if bid else None,
        }, state)
        if player != self._player_idx and self.ai_delay > 0:
            await asyncio.sleep(self.ai_delay)

    async def on_contract_won(self, player: int, contract: Contract, state: GameState) -> None:
        await self._send("contract_won", {
            "player": player,
            "contract": contract.value,
        }, state)

    async def on_king_called(self, player: int, king: Card, state: GameState) -> None:
        await self._send("king_called", {
            "player": player,
            "king": _card_to_dict(king),
        }, state)

    async def on_talon_revealed(self, groups: list[list[Card]], state: GameState) -> None:
        await self._send("talon_revealed", {
            "groups": [[_card_to_dict(c) for c in g] for g in groups],
        }, state)

    async def on_talon_exchanged(self, state: GameState) -> None:
        await self._send("talon_exchanged", {}, state)

    async def on_card_played(self, player: int, card: Card, state: GameState) -> None:
        await self._send("card_played", {
            "player": player,
            "card": _card_to_dict(card),
        }, state)
        if player != self._player_idx and self.ai_delay > 0:
            await asyncio.sleep(self.ai_delay)

    async def on_rule_verified(self, player: int, rule: str, state: GameState) -> None:
        await self._send("rule_verified", {
            "player": player,
            "rule": rule,
        }, state)

    async def on_trick_won(self, trick: Trick, winner: int, state: GameState) -> None:
        await self._send("trick_won", {
            "winner": winner,
            "cards": [(p, _card_to_dict(c)) for p, c in trick.cards],
        }, state)

    async def on_game_end(self, scores: dict[int, int], state: GameState) -> None:
        await self._send("game_end", {
            "scores": {str(k): v for k, v in scores.items()},
        }, state)
