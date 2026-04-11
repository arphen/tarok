"""StockSkis v5 backed by Rust heuristics.

This adapter keeps the Python PlayerPort interface while delegating decision
logic to the real Rust StockSkis v5 implementation exposed through
``tarok_engine.RustGameState`` V5 methods.
"""

from __future__ import annotations

from collections.abc import Iterable

from tarok.entities.card import Card, DECK
from tarok.entities.game_state import Announcement, Contract, GameState, Phase

import tarok_engine as te


_CARD_TO_IDX: dict[Card, int] = {card: idx for idx, card in enumerate(DECK)}

_PY_TO_RUST_CONTRACT: dict[Contract, int] = {
    Contract.KLOP: 0,
    Contract.THREE: 1,
    Contract.TWO: 2,
    Contract.ONE: 3,
    Contract.SOLO_THREE: 4,
    Contract.SOLO_TWO: 5,
    Contract.SOLO_ONE: 6,
    Contract.SOLO: 7,
    Contract.BERAC: 8,
    Contract.BARVNI_VALAT: 9,
}

_RUST_TO_PY_CONTRACT: dict[int, Contract] = {
    value: key for key, value in _PY_TO_RUST_CONTRACT.items()
}

_PHASE_TO_RUST: dict[Phase, int] = {
    Phase.DEALING: 0,
    Phase.BIDDING: 1,
    Phase.KING_CALLING: 2,
    Phase.TALON_EXCHANGE: 3,
    Phase.ANNOUNCEMENTS: 4,
    Phase.TRICK_PLAY: 5,
    Phase.SCORING: 6,
    Phase.FINISHED: 7,
}


def _card_to_idx(card: Card) -> int:
    return _CARD_TO_IDX[card]


def _cards_to_indices(cards: Iterable[Card]) -> list[int]:
    return [_card_to_idx(card) for card in cards]


def _contract_to_rust(contract: Contract | None) -> int | None:
    if contract is None:
        return None
    return _PY_TO_RUST_CONTRACT.get(contract)


def _contract_to_python(contract_u8: int | None) -> Contract | None:
    if contract_u8 is None:
        return None
    return _RUST_TO_PY_CONTRACT.get(contract_u8)


def _sync_python_state_to_rust(state: GameState):
    assert te is not None

    gs = te.RustGameState(state.dealer)

    # Set cards as visible in the Python state.
    hands = [_cards_to_indices(hand) for hand in state.hands]
    talon = _cards_to_indices(state.talon)
    gs.deal_hands(hands, talon)

    # Sync bidding, contract and roles.
    for bid in state.bids:
        gs.add_bid(bid.player, _contract_to_rust(bid.contract))

    gs.declarer = state.declarer
    gs.partner = state.partner
    gs.contract = _contract_to_rust(state.contract)

    if state.called_king is not None:
        gs.set_called_king(_card_to_idx(state.called_king))

    if state.talon_revealed:
        gs.set_talon_revealed([_cards_to_indices(group) for group in state.talon_revealed])

    for card in state.put_down:
        gs.add_put_down(_card_to_idx(card))

    # Rebuild trick history so Rust has played-card history for V5 tracking.
    for trick in state.tricks:
        if not trick.cards:
            continue
        gs.start_trick(trick.lead_player)
        for player, card in trick.cards:
            gs.play_card(player, _card_to_idx(card))
        if len(trick.cards) == state.num_players:
            gs.finish_trick()

    # Current trick (if any) is partially played.
    if state.current_trick is not None:
        gs.start_trick(state.current_trick.lead_player)
        for player, card in state.current_trick.cards:
            gs.play_card(player, _card_to_idx(card))

    gs.current_player = state.current_player
    gs.phase = _PHASE_TO_RUST.get(state.phase, 7)
    return gs


class StockSkisPlayerV5:
    """Heuristic bot v5 powered by Rust StockSkis logic."""

    def __init__(
        self,
        name: str = "StockŠkis-v5",
        strength: float = 1.0,
        seed: int | None = None,
    ):
        self._name = name
        self.strength = strength
        self.seed = seed

    @property
    def name(self) -> str:
        return self._name

    async def choose_bid(
        self,
        state: GameState,
        player_idx: int,
        legal_bids: list[Contract | None],
    ) -> Contract | None:
        gs = _sync_python_state_to_rust(state)
        chosen = _contract_to_python(gs.v5_choose_bid(player_idx))
        if chosen is None or chosen not in legal_bids:
            # V5 passed, or its preferred bid is blocked by rules
            # (e.g. THREE is forehand-only). Pass — still pure V5.
            return None
        return chosen

    async def choose_king(
        self,
        state: GameState,
        player_idx: int,
        callable_kings: list[Card],
    ) -> Card:
        gs = _sync_python_state_to_rust(state)
        chosen_idx = gs.v5_choose_king(player_idx)
        assert chosen_idx is not None, "V5 Rust returned None for king call"
        chosen = DECK[chosen_idx]
        assert chosen in callable_kings, (
            f"V5 Rust chose {chosen} but callable kings are {callable_kings}"
        )
        return chosen

    async def choose_talon_group(
        self,
        state: GameState,
        player_idx: int,
        talon_groups: list[list[Card]],
    ) -> int:
        gs = _sync_python_state_to_rust(state)
        groups_idx = [_cards_to_indices(group) for group in talon_groups]
        chosen_idx = int(gs.v5_choose_talon_group(player_idx, groups_idx))
        assert 0 <= chosen_idx < len(talon_groups), (
            f"V5 Rust chose talon group {chosen_idx} but only {len(talon_groups)} groups exist"
        )
        return chosen_idx

    async def choose_discard(
        self,
        state: GameState,
        player_idx: int,
        must_discard: int,
    ) -> list[Card]:
        gs = _sync_python_state_to_rust(state)
        chosen_idx: list[int] = list(gs.v5_choose_discards(player_idx, must_discard))
        hand_set = set(state.hands[player_idx])
        chosen_cards: list[Card] = []
        for idx in chosen_idx:
            card = DECK[idx]
            if card in hand_set and card not in chosen_cards:
                chosen_cards.append(card)
            if len(chosen_cards) == must_discard:
                break

        assert len(chosen_cards) == must_discard, (
            f"V5 Rust returned {len(chosen_cards)} discards but need {must_discard}; "
            f"indices={chosen_idx}, hand={state.hands[player_idx]}"
        )
        return chosen_cards

    async def choose_announcements(
        self,
        state: GameState,
        player_idx: int,
    ) -> list[Announcement]:
        return []

    async def choose_announce_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> int:
        # Rust V5 has no announcement policy — always pass.
        return 0

    async def choose_card(
        self,
        state: GameState,
        player_idx: int,
        legal_plays: list[Card],
    ) -> Card:
        if len(legal_plays) == 1:
            return legal_plays[0]

        gs = _sync_python_state_to_rust(state)
        chosen_idx = int(gs.v5_choose_card(player_idx))
        chosen = DECK[chosen_idx]
        assert chosen in legal_plays, (
            f"V5 Rust chose {chosen} but legal plays are {legal_plays}"
        )
        return chosen

