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
    state = GameState(phase=Phase.BIDDING, dealer=0)
    # Forehand (obvezen) = dealer+1 = 1; first bidder = dealer+2 = 2
    state.current_bidder = 2
    state.current_player = 2
    return state


@when("it is the first player's turn", target_fixture="first_player_state")
def first_player_turn(bidding_state):
    return bidding_state


@then("the legal bids should include pass and all contracts")
def all_bids_legal(first_player_state):
    # Forehand (player 1) gets all bids including THREE
    forehand = first_player_state.forehand  # player 1
    legal_fh = first_player_state.legal_bids(forehand)
    assert None in legal_fh  # pass
    for c in Contract:
        if c.is_biddable:
            assert c in legal_fh, f"Forehand missing contract: {c}"
    # Non-forehand (player 2) should NOT see THREE
    legal_nonfh = first_player_state.legal_bids(2)
    assert None in legal_nonfh
    assert Contract.THREE not in legal_nonfh
    assert Contract.TWO in legal_nonfh  # can bid two or higher


# --- Later bids higher than current ---

@given(parsers.parse('a game where "{contract}" has been bid'), target_fixture="after_first_bid")
def after_first_bid(contract):
    state = GameState(phase=Phase.BIDDING, dealer=0)
    # Player 2 (non-forehand) made the first bid
    state.current_bidder = 2
    c = CONTRACT_MAP[contract]
    state.bids.append(Bid(player=2, contract=c))
    # Next bidder is player 3 (also non-forehand)
    state.current_bidder = 3
    state.current_player = 3
    return state


@when("the next player considers bidding", target_fixture="next_player_state")
def next_player(after_first_bid):
    return after_first_bid


@then(parsers.parse('only contracts higher than "{contract}" should be available'))
def higher_contracts_only(next_player_state, contract):
    # Player 3 is non-forehand, must bid strictly higher
    legal = next_player_state.legal_bids(3)
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
    # Bidding order: 2, 3, 0, 1 (forehand last)
    for p in [2, 3, 0, 1]:
        if state.phase != Phase.BIDDING:
            break
        state = place_bid(state, p, None)
    return state


@then("the game should enter klop mode")
def game_enters_klop(all_passed):
    assert all_passed.phase == Phase.TRICK_PLAY
    assert all_passed.contract == Contract.KLOP


# --- Single bidder wins ---

@given(parsers.parse('a game where 3 players passed and 1 bid "{contract}"'), target_fixture="single_bidder")
def single_bidder(contract):
    state = GameState(phase=Phase.BIDDING, dealer=0)
    # Bidding order: 2, 3, 0, then forehand (1)
    state.current_bidder = 2
    state.current_player = 2
    # Player 2 bids, players 3, 0, 1 (forehand) pass
    state = place_bid(state, 2, CONTRACT_MAP[contract])
    state = place_bid(state, 3, None)
    state = place_bid(state, 0, None)
    state = place_bid(state, 1, None)  # forehand passes
    return state


@then("that player should be the declarer")
def is_declarer(single_bidder):
    assert single_bidder.declarer == 2


@then(parsers.parse('the contract should be "{contract}"'))
def contract_is(single_bidder, contract):
    assert single_bidder.contract == CONTRACT_MAP[contract]
