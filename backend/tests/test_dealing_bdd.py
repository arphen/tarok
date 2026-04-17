"""BDD steps for dealing.feature"""

import random

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from tarok.entities import GameState, Phase
from tarok.use_cases.deal import deal

scenarios("features/dealing.feature")


@given("a new game with 4 players", target_fixture="game_state")
def new_game():
    return GameState()


@when("cards are dealt", target_fixture="dealt_state")
def deal_cards(game_state):
    game_state.phase = Phase.DEALING
    return deal(game_state, rng=random.Random(42))


@then("each player should have 12 cards")
def each_player_12(dealt_state):
    for i, hand in enumerate(dealt_state.hands):
        assert len(hand) == 12, f"Player {i} has {len(hand)} cards"


@then("the talon should have 6 cards")
def talon_6(dealt_state):
    assert len(dealt_state.talon) == 6


@then("no cards should be duplicated across hands and talon")
def no_duplicates(dealt_state):
    all_cards = []
    for hand in dealt_state.hands:
        all_cards.extend(hand)
    all_cards.extend(dealt_state.talon)
    assert len(set(all_cards)) == len(all_cards)


@then("the total number of cards in hands and talon should be 54")
def total_54(dealt_state):
    total = sum(len(h) for h in dealt_state.hands) + len(dealt_state.talon)
    assert total == 54


@then(parsers.parse('the game phase should be "{phase}"'))
def phase_is(dealt_state, phase):
    assert dealt_state.phase.value == phase
