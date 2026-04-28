"""WebSocket observer — broadcasts game events to connected clients."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket

from tarok.entities import Card, CardType, DECK, Contract, GameState, Phase


def _card_to_dict(card: Card) -> dict:
    return {
        "card_type": card.card_type.value,
        "value": card.value,
        "suit": card.suit.value if card.suit else None,
        "label": card.label,
        "points": card.points,
    }


def _build_card_tracker(state: GameState) -> dict:
    """Build a card tracker for the human player (P0).

    Tracks which cards have been played, which are remaining,
    which players are void in certain suits, and tarok ranges per player.
    """
    # All cards in the deck
    all_cards = set(DECK)

    # Cards played so far (from completed tricks + current trick)
    played_cards: list[tuple[int, Card]] = []
    for trick in state.tricks:
        for player, card in trick.cards:
            played_cards.append((player, card))
    current_trick = getattr(state, "current_trick", None)
    if current_trick and hasattr(current_trick, "cards"):
        for player, card in current_trick.cards:
            played_cards.append((player, card))

    played_set = {c for _, c in played_cards}

    # Cards known to be out of play: human hand + played + put_down
    human_hand = set(state.hands[0]) if state.hands else set()
    put_down = set(state.put_down) if state.put_down else set()
    known_out = played_set | human_hand | put_down

    # Remaining cards (not in our hand, not played, not put down)
    remaining = all_cards - known_out

    # Group remaining by suit/tarok
    remaining_by_group: dict[str, list[dict]] = {
        "taroks": [],
        "hearts": [],
        "diamonds": [],
        "clubs": [],
        "spades": [],
    }
    for c in sorted(remaining, key=lambda x: x.sort_key):
        d = _card_to_dict(c)
        if c.card_type == CardType.TAROK:
            remaining_by_group["taroks"].append(d)
        elif c.suit:
            remaining_by_group[c.suit.value].append(d)

    # Track voids and tarok range per opponent
    player_info: dict[int, dict] = {}
    n_players = len(state.hands) if state.hands else 4
    for p in range(n_players):
        if p == 0:
            continue  # skip human

        void_suits: list[str] = []
        highest_tarok: int | None = None
        lowest_tarok: int | None = None
        taroks_played: list[int] = []

        for trick in state.tricks:
            # Check for voids: if a player didn't follow suit, they're void
            if len(trick.cards) >= 2:
                lead_card = trick.cards[0][1]
                lead_suit = lead_card.suit if lead_card.card_type == CardType.SUIT else None

                for tp, tc in trick.cards:
                    if (
                        tp == p
                        and lead_suit
                        and tc.card_type == CardType.SUIT
                        and tc.suit != lead_suit
                    ):
                        if lead_suit.value not in void_suits:
                            void_suits.append(lead_suit.value)
                    if tp == p and lead_suit and tc.card_type == CardType.TAROK:
                        if lead_suit.value not in void_suits:
                            void_suits.append(lead_suit.value)

            # Track taroks played by this player
            for tp, tc in trick.cards:
                if tp == p and tc.card_type == CardType.TAROK:
                    taroks_played.append(tc.value)

        if taroks_played:
            highest_tarok = max(taroks_played)
            lowest_tarok = min(taroks_played)

        player_info[p] = {
            "void_suits": void_suits,
            "highest_tarok": highest_tarok,
            "lowest_tarok": lowest_tarok,
            "taroks_played_count": len(taroks_played),
        }

    return {
        "remaining_by_group": remaining_by_group,
        "remaining_count": len(remaining),
        "player_info": {str(k): v for k, v in player_info.items()},
    }


def _normalize_legal_bids(values: list[int | None]) -> list[int | None]:
    """Return legal bids unchanged.

    Bidding now uses Rust contract ids directly across the websocket path.
    """
    return values


def _state_for_player(
    state: Any,
    player_idx: int,
    player_names: list[str],
    match_info: dict | None = None,
    reveal_hands: bool = False,
    card_tracker: dict | None = None,
) -> dict:

    raw_current_player = state.current_player
    has_bidding_current = hasattr(state, "current_bidder")
    bidding_current = getattr(state, "current_bidder", None)
    current_player = raw_current_player
    if state.phase == Phase.BIDDING:
        # During the terminal bid event we deliberately set current_bidder=None
        # so clients don't render another clickable pass for the winner.
        if has_bidding_current and bidding_current is not None:
            current_player = bidding_current
            is_current = current_player == player_idx
        elif has_bidding_current and bidding_current is None:
            is_current = False
        else:
            is_current = current_player == player_idx
    else:
        is_current = current_player == player_idx

    # Phase-appropriate legal actions
    # Note: legal_bids/legal_plays/callable_kings are game logic that now
    # lives exclusively in the Rust engine.  The game loop communicates
    # these to players directly; the state snapshot may not have them.
    legal_plays: list[dict] = []
    legal_bids: list[int | None] | None = None
    callable_kings: list[dict] | None = None
    must_discard: int = 0

    if state.phase == Phase.BIDDING and is_current:
        if hasattr(state, "legal_bids") and callable(getattr(state, "legal_bids", None)):
            legal_bids = _normalize_legal_bids(list(state.legal_bids(player_idx)))
            if legal_bids == [None]:
                active_contracts: list[int] = []
                has_active_solo = False
                for bid in state.bids:
                    contract = getattr(bid, "contract", None)
                    if contract is None:
                        continue
                    value = contract.value if hasattr(contract, "value") else int(contract)
                    active_contracts.append(value)
                    # Accept both representations:
                    # - Python Contract.SOLO.value == 0
                    # - Rust bid id SOLO == 7
                    if value in (Contract.SOLO.value, 7):
                        has_active_solo = True
                if not has_active_solo:
                    raise RuntimeError(
                        "Invalid bidding snapshot: pass-only legal_bids without active solo bid "
                        f"(player_idx={player_idx}, current_player={current_player}, "
                        f"current_bidder={bidding_current if has_bidding_current else 'missing'}, "
                        f"active_contracts={active_contracts}, legal_bids={legal_bids})"
                    )
    elif state.phase == Phase.KING_CALLING and is_current:
        if hasattr(state, "callable_kings") and callable(getattr(state, "callable_kings", None)):
            callable_kings = [_card_to_dict(k) for k in state.callable_kings()]
    elif state.phase == Phase.TALON_EXCHANGE and is_current:
        if state.contract and state.contract.talon_cards > 0:
            n_players = len(state.hands) if state.hands else 4
            base_hand_size = 16 if n_players == 3 else 12
            if len(state.hands[player_idx]) > base_hand_size:
                must_discard = len(state.hands[player_idx]) - base_hand_size
    elif state.phase == Phase.TRICK_PLAY and is_current:
        if hasattr(state, "legal_plays") and callable(getattr(state, "legal_plays", None)):
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
        "contract": state.contract.value if state.contract is not None else None,
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
        "current_player": current_player,
        "scores": state.scores if state.scores else None,
        "legal_plays": legal_plays,
        "legal_bids": legal_bids,
        "callable_kings": callable_kings,
        "must_discard": must_discard,
        "player_names": player_names,
        "card_tracker": card_tracker
        if card_tracker is not None
        else (
            _build_card_tracker(state)
            if state.phase in (Phase.TRICK_PLAY, Phase.SCORING, Phase.FINISHED)
            else None
        ),
        "match_info": match_info,
        "hands": (
            {str(i): [_card_to_dict(c) for c in state.hands[i]] for i in range(len(state.hands))}
            if reveal_hands
            else None
        ),
    }


class WebSocketObserver:
    """Broadcasts game events to a connected WebSocket client."""

    def __init__(
        self, ws: WebSocket, player_idx: int, player_names: list[str], ai_delay: float = 1.0
    ):
        self._ws = ws
        self._player_idx = player_idx
        self._player_names = player_names
        self.ai_delay = ai_delay
        self._match_info: dict | None = None
        self.reveal_hands: bool = False
        self._last_tracker_tricks: int = -1
        self._cached_tracker: dict | None = None

    def set_match_info(
        self,
        round_num: int,
        total_rounds: int,
        cumulative_scores: dict,
        caller_counts: dict,
        called_counts: dict,
        round_history: list[dict],
    ) -> None:
        self._match_info = {
            "round_num": round_num,
            "total_rounds": total_rounds,
            "cumulative_scores": {str(k): v for k, v in cumulative_scores.items()},
            "caller_counts": {str(k): v for k, v in caller_counts.items()},
            "called_counts": {str(k): v for k, v in called_counts.items()},
            "round_history": round_history,
        }

    async def _send(self, event: str, data: Any, state: Any) -> None:
        # Cache card tracker — only recompute when trick count changes
        tracker = None
        if state.phase in (Phase.TRICK_PLAY, Phase.SCORING, Phase.FINISHED):
            trick_count = len(state.current_trick.cards) if state.current_trick else 0
            trick_count += state.tricks_played
            if trick_count != self._last_tracker_tricks:
                self._cached_tracker = _build_card_tracker(state)
                self._last_tracker_tricks = trick_count
            tracker = self._cached_tracker

        msg = {
            "event": event,
            "data": data,
            "state": _state_for_player(
                state,
                self._player_idx,
                self._player_names,
                self._match_info,
                self.reveal_hands,
                tracker,
            ),
        }
        await self._ws.send_json(msg)

    async def on_game_start(self, state: Any) -> None:
        await self._send("game_start", {}, state)

    async def on_deal(self, state: Any) -> None:
        await self._send("deal", {}, state)

    async def on_bid(self, player: int, bid: Contract | None, state: Any) -> None:
        await self._send(
            "bid",
            {
                "player": player,
                "contract": bid.value if bid else None,
            },
            state,
        )
        if player != self._player_idx and self.ai_delay > 0:
            await asyncio.sleep(self.ai_delay)

    async def on_contract_won(self, player: int, contract: Contract | None, state: Any) -> None:
        await self._send(
            "contract_won",
            {
                "player": player,
                "contract": contract.value if contract is not None else None,
            },
            state,
        )

    async def on_king_called(self, player: int, king: Card, state: Any) -> None:
        await self._send(
            "king_called",
            {
                "player": player,
                "king": _card_to_dict(king),
            },
            state,
        )

    async def on_talon_revealed(self, groups: list[list[Card]], state: Any) -> None:
        await self._send(
            "talon_revealed",
            {
                "groups": [[_card_to_dict(c) for c in g] for g in groups],
            },
            state,
        )

    async def on_talon_exchanged(self, state: Any, picked=None, discarded=None) -> None:
        await self._send("talon_exchanged", {}, state)

    async def on_trick_start(self, state: Any) -> None:
        await self._send("trick_start", {}, state)

    async def on_talon_group_picked(self, state: Any) -> None:
        await self._send("talon_group_picked", {}, state)

    async def on_card_played(self, player: int, card: Card, state: Any) -> None:
        await self._send(
            "card_played",
            {
                "player": player,
                "card": _card_to_dict(card),
            },
            state,
        )
        if player != self._player_idx and self.ai_delay > 0:
            await asyncio.sleep(self.ai_delay)

    async def on_rule_verified(self, player: int, rule: str, state: Any) -> None:
        await self._send(
            "rule_verified",
            {
                "player": player,
                "rule": rule,
            },
            state,
        )

    async def on_trick_won(self, trick: Any, winner: int, state: Any) -> None:
        await self._send(
            "trick_won",
            {
                "winner": winner,
                "cards": [(p, _card_to_dict(c)) for p, c in trick.cards],
            },
            state,
        )

    async def on_game_end(
        self, scores: dict[int, int], state: Any, breakdown: dict | None = None
    ) -> None:
        await self._send(
            "game_end",
            {
                "scores": {str(k): v for k, v in scores.items()},
            },
            state,
        )

    async def send_match_update(
        self,
        cumulative_scores: dict,
        caller_counts: dict,
        called_counts: dict,
        round_history: list[dict],
        round_num: int,
        total_rounds: int,
        state: Any,
    ) -> None:
        """Send match progress between rounds."""
        await self._send(
            "match_update",
            {
                "cumulative_scores": {str(k): v for k, v in cumulative_scores.items()},
                "caller_counts": {str(k): v for k, v in caller_counts.items()},
                "called_counts": {str(k): v for k, v in called_counts.items()},
                "round_history": round_history,
                "round_num": round_num,
                "total_rounds": total_rounds,
            },
            state,
        )

    async def send_match_end(
        self,
        cumulative_scores: dict,
        caller_counts: dict,
        called_counts: dict,
        round_history: list[dict],
        total_rounds: int,
        state: GameState,
    ) -> None:
        """Send final match results."""
        await self._send(
            "match_end",
            {
                "cumulative_scores": {str(k): v for k, v in cumulative_scores.items()},
                "caller_counts": {str(k): v for k, v in caller_counts.items()},
                "called_counts": {str(k): v for k, v in called_counts.items()},
                "round_history": round_history,
                "total_rounds": total_rounds,
            },
            state,
        )
