"""BDD steps for tricks.feature — rewritten to avoid duplicate step names.

pytest-bdd matches steps globally by text, so we use unique Given texts
and a single shared When/Then per unique step phrase.
"""

import random

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from tarok.entities import Card, CardType, Suit, SuitRank, tarok, suit_card, GameState, Phase, Trick, Contract, PlayerRole
from tarok.use_cases.play_trick import play_card, start_trick

scenarios("features/tricks.feature")


# ---- helpers ----

def _add_heart_lead(state: GameState) -> GameState:
    state.current_trick = Trick(lead_player=1)
    state.current_trick.cards.append((1, suit_card(Suit.HEARTS, SuitRank.QUEEN)))
    state.current_player = 0
    return state


# ---- Scenario: Must follow lead suit when possible ----

@given("a player has hearts and clubs in hand", target_fixture="state")
def player_hearts_clubs():
    state = GameState(phase=Phase.TRICK_PLAY)
    state.hands[0] = [
        suit_card(Suit.HEARTS, SuitRank.KING),
        suit_card(Suit.HEARTS, SuitRank.PIP_1),
        suit_card(Suit.CLUBS, SuitRank.KING),
        suit_card(Suit.CLUBS, SuitRank.PIP_1),
    ]
    return state


@when("a heart is led", target_fixture="after_lead")
def lead_heart(state):
    return _add_heart_lead(state)


@then("the player can only play hearts")
def only_hearts(after_lead):
    legal = after_lead.legal_plays(0)
    assert all(c.suit == Suit.HEARTS for c in legal)
    assert len(legal) == 2


# ---- Scenario: Must play tarok if cannot follow suit ----

@given("a player has only clubs and taroks in hand", target_fixture="state")
def player_clubs_taroks():
    state = GameState(phase=Phase.TRICK_PLAY)
    state.hands[0] = [
        suit_card(Suit.CLUBS, SuitRank.KING),
        suit_card(Suit.CLUBS, SuitRank.PIP_1),
        tarok(5),
        tarok(10),
    ]
    return state


@then("the player can only play taroks")
def only_taroks(after_lead):
    legal = after_lead.legal_plays(0)
    assert all(c.card_type == CardType.TAROK for c in legal)
    assert len(legal) == 2


# ---- Scenario: Can play anything if no suit and no tarok ----

@given("a player has only clubs and diamonds in hand", target_fixture="state")
def player_clubs_diamonds():
    state = GameState(phase=Phase.TRICK_PLAY)
    state.hands[0] = [
        suit_card(Suit.CLUBS, SuitRank.KING),
        suit_card(Suit.DIAMONDS, SuitRank.PIP_1),
    ]
    return state


@then("the player can play any card")
def any_card(after_lead):
    legal = after_lead.legal_plays(0)
    assert len(legal) == 2


# ---- Scenario: Must follow tarok with tarok ----

@given("a player has taroks and suit cards in hand", target_fixture="state")
def player_taroks_and_suits():
    state = GameState(phase=Phase.TRICK_PLAY)
    state.hands[0] = [
        tarok(3),
        tarok(8),
        suit_card(Suit.HEARTS, SuitRank.KING),
        suit_card(Suit.CLUBS, SuitRank.PIP_1),
    ]
    return state


@when("a tarok is led", target_fixture="after_lead")
def lead_tarok(state):
    state.current_trick = Trick(lead_player=1)
    state.current_trick.cards.append((1, tarok(15)))
    state.current_player = 0
    return state


# ---- Scenario: Higher tarok beats lower tarok ----

@given("tarok XV is played against tarok X", target_fixture="tarok_trick")
def tarok_15_vs_10():
    trick = Trick(lead_player=0)
    trick.cards = [(0, tarok(10)), (1, tarok(15)), (2, tarok(5)), (3, tarok(3))]
    return trick


@then("tarok XV should win")
def tarok_15_wins(tarok_trick):
    assert tarok_trick.winner() == 1


# ---- Scenario: Any tarok beats any suit card ----

@given("tarok I is played against the king of hearts", target_fixture="tarok_vs_suit")
def tarok_1_vs_king():
    trick = Trick(lead_player=0)
    trick.cards = [
        (0, suit_card(Suit.HEARTS, SuitRank.KING)),
        (1, tarok(1)),
        (2, suit_card(Suit.HEARTS, SuitRank.PIP_1)),
        (3, suit_card(Suit.HEARTS, SuitRank.PIP_2)),
    ]
    return trick


@then("the tarok should win")
def tarok_wins(tarok_vs_suit):
    assert tarok_vs_suit.winner() == 1


# ---- Scenario: Lead suit wins against off-suit ----

@given("king of hearts leads and king of clubs follows", target_fixture="suit_trick")
def heart_king_vs_club_king():
    trick = Trick(lead_player=0)
    trick.cards = [
        (0, suit_card(Suit.HEARTS, SuitRank.KING)),
        (1, suit_card(Suit.CLUBS, SuitRank.KING)),
        (2, suit_card(Suit.HEARTS, SuitRank.PIP_1)),
        (3, suit_card(Suit.HEARTS, SuitRank.PIP_2)),
    ]
    return trick


@then("king of hearts should win")
def heart_king_wins(suit_trick):
    assert suit_trick.winner() == 0


# ---- Scenario: After 12 tricks the phase is scoring ----

@given("a game in trick play phase with 4 random players", target_fixture="trick_play_state")
def trick_play_game():
    from tarok.use_cases.deal import deal
    state = GameState()
    state.phase = Phase.DEALING
    state = deal(state, rng=random.Random(42))
    state.phase = Phase.TRICK_PLAY
    state.contract = Contract.THREE
    state.declarer = 0
    state.roles = {
        0: PlayerRole.DECLARER, 1: PlayerRole.PARTNER,
        2: PlayerRole.OPPONENT, 3: PlayerRole.OPPONENT,
    }
    state.current_player = 1
    return state


@when("all 12 tricks are played", target_fixture="after_tricks")
def play_all_tricks(trick_play_state):
    state = trick_play_state
    for _ in range(12):
        state = start_trick(state)
        for __ in range(4):
            legal = state.legal_plays(state.current_player)
            state = play_card(state, state.current_player, legal[0])
    return state


@then(parsers.parse('the game phase should be "{phase}"'))
def phase_check(after_tricks, phase):
    assert after_tricks.phase.value == phase
