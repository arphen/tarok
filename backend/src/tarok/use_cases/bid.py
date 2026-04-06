"""Bidding use case — players bid ascending contracts or pass."""

from __future__ import annotations

from tarok.entities.game_state import Bid, Contract, GameState, Phase, PlayerRole


def place_bid(state: GameState, player: int, contract: Contract | None) -> GameState:
    """Player places a bid or passes. Returns updated state."""
    assert state.phase == Phase.BIDDING
    assert player == state.current_bidder

    state.bids.append(Bid(player=player, contract=contract))

    # Check if bidding is over
    if _bidding_complete(state):
        _resolve_bidding(state)
    else:
        state.current_bidder = _next_bidder(state)
        state.current_player = state.current_bidder

    return state


def _next_bidder(state: GameState) -> int:
    """Find next player who hasn't passed."""
    passed = {b.player for b in state.bids if b.contract is None}
    start = (state.current_bidder + 1) % state.num_players
    for i in range(state.num_players):
        candidate = (start + i) % state.num_players
        if candidate not in passed:
            return candidate
    return state.current_bidder  # Shouldn't reach here


def _bidding_complete(state: GameState) -> bool:
    """Bidding ends when 3 players have passed (one winner) or all passed."""
    passed = {b.player for b in state.bids if b.contract is None}
    active_bidders = set(range(state.num_players)) - passed

    # Everyone passed
    if len(passed) == state.num_players:
        return True

    # Only one bidder remains and at least one full round completed
    if len(active_bidders) == 1 and len(state.bids) >= state.num_players:
        return True

    # All 4 have acted at least once, and 3 have passed
    if len(passed) >= state.num_players - 1:
        return True

    return False


def _resolve_bidding(state: GameState) -> None:
    """Determine the winner and contract."""
    bids_with_contract = [b for b in state.bids if b.contract is not None]

    if not bids_with_contract:
        # Everyone passed → klop
        state.contract = Contract.KLOP
        state.phase = Phase.TRICK_PLAY
        # No declarer in klop — everyone plays for themselves
        for p in range(state.num_players):
            state.roles[p] = PlayerRole.OPPONENT
        state.current_player = (state.dealer + 1) % state.num_players
        return

    # Highest bid wins
    winning_bid = max(bids_with_contract, key=lambda b: b.contract.strength)  # type: ignore
    state.declarer = winning_bid.player
    state.contract = winning_bid.contract
    state.current_player = state.declarer

    # Set declarer role
    state.roles[state.declarer] = PlayerRole.DECLARER

    if state.contract.is_berac:
        # Berac: no partner, no talon, no announcements → straight to trick play
        for p in range(state.num_players):
            if p != state.declarer:
                state.roles[p] = PlayerRole.OPPONENT
        state.phase = Phase.TRICK_PLAY
        state.current_player = (state.dealer + 1) % state.num_players
    elif state.contract.is_solo:
        # Solo: no partner, all others are opponents
        for p in range(state.num_players):
            if p != state.declarer:
                state.roles[p] = PlayerRole.OPPONENT
        if state.contract.talon_cards > 0:
            state.phase = Phase.TALON_EXCHANGE
        else:
            state.phase = Phase.ANNOUNCEMENTS
    else:
        state.phase = Phase.KING_CALLING
