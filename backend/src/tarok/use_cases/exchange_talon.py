"""Talon exchange use case — uses talon strategy plugins."""

from __future__ import annotations

from tarok.entities.card import Card, CardType, SuitRank
from tarok.entities.game_state import GameState, Phase


def reveal_talon(state: GameState) -> list[list[Card]]:
    """Split the talon into groups based on contract."""
    assert state.contract is not None
    n = state.contract.talon_cards
    if n == 0:
        return []

    group_size = 6 // (6 // n) if n in (1, 2, 3) else n
    # Three: 2 groups of 3
    # Two: 3 groups of 2
    # One: 6 groups of 1
    groups: list[list[Card]] = []
    for i in range(0, 6, group_size):
        groups.append(state.talon[i : i + group_size])

    state.talon_revealed = groups
    return groups


def pick_talon_group(state: GameState, group_idx: int) -> GameState:
    """Declarer picks up a talon group."""
    assert state.phase == Phase.TALON_EXCHANGE
    assert state.declarer is not None
    assert 0 <= group_idx < len(state.talon_revealed)

    picked = state.talon_revealed[group_idx]
    state.hands[state.declarer].extend(picked)
    return state


def discard_cards(state: GameState, cards: list[Card]) -> GameState:
    """Declarer discards cards back (cannot discard kings or taroks, with exceptions)."""
    assert state.phase == Phase.TALON_EXCHANGE
    assert state.declarer is not None
    assert state.contract is not None
    assert len(cards) == state.contract.talon_cards

    # Collect non-king suit cards that will remain after discarding
    discarding_suits = [
        c for c in cards
        if c.card_type != CardType.TAROK and not c.is_king
    ]
    for card in cards:
        # Cannot discard kings
        assert not card.is_king, f"Cannot discard a king: {card}"
        # Cannot discard taroks unless all remaining suit cards are also being discarded
        if card.card_type == CardType.TAROK:
            hand_suits = [
                c for c in state.hands[state.declarer]
                if c.card_type != CardType.TAROK and not c.is_king
            ]
            remaining_suits = len(hand_suits) - len(discarding_suits)
            assert remaining_suits <= 0, "Cannot discard taroks if you have suit cards"

    hand = state.hands[state.declarer]
    for card in cards:
        hand.remove(card)
    state.hands[state.declarer] = sorted(hand, key=lambda c: c.sort_key)
    state.put_down = cards

    state.phase = Phase.ANNOUNCEMENTS
    return state
