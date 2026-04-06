"""King calling use case — declarer names a king to find a partner."""

from __future__ import annotations

from tarok.entities.card import Card
from tarok.entities.game_state import GameState, Phase, PlayerRole


def call_king(state: GameState, king: Card) -> GameState:
    assert state.phase == Phase.KING_CALLING
    assert state.declarer is not None
    assert king in state.callable_kings()

    state.called_king = king

    # Find partner (secret until king is played)
    for p in range(state.num_players):
        if p == state.declarer:
            continue
        if king in state.hands[p]:
            state.partner = p
            state.roles[p] = PlayerRole.PARTNER
            break

    # If called king is in talon, declarer plays alone
    if state.partner is None:
        for p in range(state.num_players):
            if p != state.declarer:
                state.roles[p] = PlayerRole.OPPONENT

    # Set remaining players as opponents
    for p in range(state.num_players):
        if p not in state.roles:
            state.roles[p] = PlayerRole.OPPONENT

    state.phase = Phase.TALON_EXCHANGE
    state.current_player = state.declarer
    return state
