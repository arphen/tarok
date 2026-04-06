"""BDD steps for bidding.feature"""

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from tarok.entities.game_state import Bid, Contract, GameState, Phase, PlayerRole
from tarok.use_cases.bid import place_bid

scenarios("features/bidding.feature")

CONTRACT_MAP = {
    "three": Contract.THREE,
    "two": Contract.TWO,
    "one": Contract.ONE,
    "solo_three": Contract.SOLO_THREE,
    "solo_two": Contract.SOLO_TWO,
    "solo_one": Contract.SOLO_ONE,
    "solo": Contract.SOLO,
}


# --- First player can bid anything ---

@given("a game in bidding phase", target_fixture="bidding_state")
def bidding_state():
    state = GameState(phase=Phase.BIDDING)
    state.current_bidder = 0
    state.current_player = 0
    return state


@when("it is the first player's turn", target_fixture="first_player_state")
def first_player_turn(bidding_state):
    return bidding_state


@then("the legal bids should include pass and all contracts")
def all_bids_legal(first_player_state):
    legal = first_player_state.legal_bids(0)
    assert None in legal  # pass
    for c in Contract:
        if c.is_biddable:
            assert c in legal, f"Missing contract: {c}"


# --- Later bids higher than current ---

@given(parsers.parse('a game where "{contract}" has been bid'), target_fixture="after_first_bid")
def after_first_bid(contract):
    state = GameState(phase=Phase.BIDDING)
    state.current_bidder = 0
    state.current_player = 0
    c = CONTRACT_MAP[contract]
    state.bids.append(Bid(player=0, contract=c))
    state.current_bidder = 1
    state.current_player = 1
    return state


@when("the next player considers bidding", target_fixture="next_player_state")
def next_player(after_first_bid):
    return after_first_bid


@then(parsers.parse('only contracts higher than "{contract}" should be available'))
def higher_contracts_only(next_player_state, contract):
    legal = next_player_state.legal_bids(1)
    threshold = CONTRACT_MAP[contract].strength
    for bid in legal:
        if bid is not None:
            assert bid.strength > threshold, f"{bid} is not stronger than {contract}"
    # Also check pass is available
    assert None in legal


# --- All pass triggers re-deal ---

@when("all four players pass", target_fixture="all_passed")
def all_pass(bidding_state):
    state = bidding_state
    for p in range(4):
        if state.phase != Phase.BIDDING:
            break
        state.current_bidder = p
        state.current_player = p
        state = place_bid(state, p, None)
    return state


@then("the game should enter klop mode")
def game_enters_klop(all_passed):
    assert all_passed.phase == Phase.TRICK_PLAY
    assert all_passed.contract == Contract.KLOP


# --- Single bidder wins ---

@given(parsers.parse('a game where 3 players passed and 1 bid "{contract}"'), target_fixture="single_bidder")
def single_bidder(contract):
    state = GameState(phase=Phase.BIDDING)
    state.current_bidder = 0
    state.current_player = 0
    # Player 0 bids, players 1,2,3 pass
    state = place_bid(state, 0, CONTRACT_MAP[contract])
    state = place_bid(state, 1, None)
    state = place_bid(state, 2, None)
    state = place_bid(state, 3, None)
    return state


@then("that player should be the declarer")
def is_declarer(single_bidder):
    assert single_bidder.declarer == 0


@then(parsers.parse('the contract should be "{contract}"'))
def contract_is(single_bidder, contract):
    assert single_bidder.contract == CONTRACT_MAP[contract]
