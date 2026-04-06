"""Trick play use case — individual card plays and trick resolution."""

from __future__ import annotations

from tarok.entities.card import Card
from tarok.entities.game_state import GameState, Phase, Trick


def start_trick(state: GameState) -> GameState:
    """Begin a new trick led by current_player."""
    assert state.phase == Phase.TRICK_PLAY
    state.current_trick = Trick(lead_player=state.current_player)
    return state


def play_card(state: GameState, player: int, card: Card) -> GameState:
    """Play a card into the current trick."""
    assert state.phase == Phase.TRICK_PLAY
    assert state.current_trick is not None
    assert player == state.current_player
    assert card in state.legal_plays(player), (
        f"Illegal play: {card} not in {state.legal_plays(player)}"
    )

    state.hands[player].remove(card)
    state.current_trick.cards.append((player, card))

    if state.current_trick.is_complete:
        _resolve_trick(state)
    else:
        state.current_player = (player + 1) % state.num_players

    return state


def _resolve_trick(state: GameState) -> None:
    """Resolve completed trick — determine winner and start next or end game."""
    assert state.current_trick is not None
    assert state.current_trick.is_complete

    winner = state.current_trick.winner()
    state.tricks.append(state.current_trick)
    state.current_trick = None
    state.current_player = winner

    if state.tricks_played == 12:
        state.phase = Phase.SCORING
