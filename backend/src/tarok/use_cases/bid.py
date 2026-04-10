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
    """Move to the next player clockwise (single-round, no skipping)."""
    return (state.current_bidder + 1) % state.num_players


def _bidding_complete(state: GameState) -> bool:
    """Bidding ends after all 4 players have acted (single round).

    Forehand goes last and their decision is final (priority rule).
    """
    players_acted = {b.player for b in state.bids}
    if len(players_acted) >= state.num_players:
        return True

    # Safety: everyone passed before a full round (shouldn't happen)
    passed = {b.player for b in state.bids if b.contract is None}
    if len(passed) == state.num_players:
        return True

    return False


def _resolve_bidding(state: GameState) -> None:
    """Determine the winner and contract.

    When two players bid the same contract, forehand wins (priority).
    """
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

    # Highest bid wins; forehand wins ties
    forehand = (state.dealer + 1) % state.num_players

    def bid_priority(b: Bid) -> tuple[int, int]:
        """(strength, forehand_bonus) — higher is better."""
        return (b.contract.strength, 1 if b.player == forehand else 0)  # type: ignore

    winning_bid = max(bids_with_contract, key=bid_priority)
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
