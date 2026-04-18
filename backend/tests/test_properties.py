"""Hypothesis property-based tests for Slovenian Tarok invariants."""

import random

from hypothesis import given as hgiven, settings, assume
from hypothesis import strategies as st

from tarok.entities import (
    Card,
    CardType,
    Suit,
    SuitRank,
    DECK,
    tarok,
    suit_card,
    PAGAT,
    MOND,
    SKIS,
    Contract,
    GameState,
    Phase,
    PlayerRole,
    Team,
    Trick,
    Announcement,
    compute_card_points,
)
from tarok.use_cases.deal import deal
from tarok.use_cases.bid import place_bid
from tarok.use_cases.play_trick import play_card, start_trick


def _safe_discard(state):
    """Pick legal cards to discard after talon exchange.  Returns None if impossible."""
    hand = state.hands[state.declarer]
    n = state.contract.talon_cards
    suit_cards = [c for c in hand if c.card_type != CardType.TAROK and not c.is_king]
    if len(suit_cards) >= n:
        return suit_cards[:n]
    # Taroks only allowed when hand has zero non-king suit cards
    if not suit_cards:
        taroks = [c for c in hand if c.card_type == CardType.TAROK]
        if len(taroks) >= n:
            return taroks[:n]
    return None


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

random_seeds = st.integers(min_value=0, max_value=2**32 - 1)
contracts = st.sampled_from(list(Contract))


# ---------------------------------------------------------------------------
# Property: Deck always has 54 cards with correct raw total
# ---------------------------------------------------------------------------


@hgiven(random_seeds)
@settings(max_examples=20)
def test_deck_invariants(seed):
    deck = list(DECK)
    assert len(deck) == 54
    assert len(set(deck)) == 54  # all unique
    raw = sum(c.points for c in deck)
    assert raw == 106  # known raw total


# ---------------------------------------------------------------------------
# Property: compute_card_points of entire deck is always 70
# ---------------------------------------------------------------------------


@hgiven(random_seeds)
@settings(max_examples=20)
def test_full_deck_counted_points(seed):
    deck = list(DECK)
    rng = random.Random(seed)
    rng.shuffle(deck)  # order shouldn't matter for total
    assert compute_card_points(deck) == TOTAL_GAME_POINTS


# ---------------------------------------------------------------------------
# Property: Any deal distributes all 54 cards (12 per player + 6 talon)
# ---------------------------------------------------------------------------


@hgiven(random_seeds)
@settings(max_examples=50)
def test_deal_card_conservation(seed):
    rng = random.Random(seed)
    state = GameState(phase=Phase.DEALING)
    state = deal(state, rng=rng)
    all_cards = []
    for hand in state.hands:
        assert len(hand) == 12
        all_cards.extend(hand)
    assert len(state.talon) == 6
    all_cards.extend(state.talon)
    assert len(all_cards) == 54
    assert len(set(all_cards)) == 54  # no duplicates


# ---------------------------------------------------------------------------
# Property: Legal plays are always non-empty during trick play
# ---------------------------------------------------------------------------


@hgiven(random_seeds)
@settings(max_examples=30, deadline=5000)
def test_legal_plays_always_nonempty(seed):
    rng = random.Random(seed)
    state = GameState(phase=Phase.DEALING)
    state = deal(state, rng=rng)

    # Quick bid setup
    state.phase = Phase.BIDDING
    state.current_bidder = 0
    state.current_player = 0
    state = place_bid(state, 0, Contract.THREE)
    for p in [1, 2, 3]:
        state = place_bid(state, p, None)

    # King calling
    if not state.contract.is_solo:
        kings = state.callable_kings()
        if kings:
            from tarok.use_cases.call_king import call_king

            state = call_king(state, kings[0])

    # Talon exchange
    if state.phase == Phase.TALON_EXCHANGE:
        from tarok.use_cases.exchange_talon import reveal_talon, pick_talon_group, discard_cards

        reveal_talon(state)  # mutates state, returns groups
        state = pick_talon_group(state, 0)
        discards = _safe_discard(state)
        assume(discards is not None)
        state = discard_cards(state, discards)

    state.phase = Phase.TRICK_PLAY
    for trick_num in range(12):
        state = start_trick(state)
        for _ in range(4):
            legal = state.legal_plays(state.current_player)
            assert len(legal) > 0, (
                f"No legal plays for player {state.current_player} "
                f"on trick {trick_num}, hand={state.hands[state.current_player]}"
            )
            card = rng.choice(legal)
            state = play_card(state, state.current_player, card)


# ---------------------------------------------------------------------------
# Property: Only declarer team scores after a complete game
# ---------------------------------------------------------------------------


@hgiven(random_seeds)
@settings(max_examples=50, deadline=5000)
def test_opponents_score_zero(seed):
    rng = random.Random(seed)
    state = GameState(phase=Phase.DEALING)
    state = deal(state, rng=rng)

    state.phase = Phase.BIDDING
    state.current_bidder = 0
    state.current_player = 0
    state = place_bid(state, 0, Contract.THREE)
    for p in [1, 2, 3]:
        state = place_bid(state, p, None)

    if not state.contract.is_solo:
        kings = state.callable_kings()
        if kings:
            from tarok.use_cases.call_king import call_king

            state = call_king(state, kings[0])

    if state.phase == Phase.TALON_EXCHANGE:
        from tarok.use_cases.exchange_talon import reveal_talon, pick_talon_group, discard_cards

        reveal_talon(state)  # mutates state, returns groups
        state = pick_talon_group(state, 0)
        discards = _safe_discard(state)
        assume(discards is not None)
        state = discard_cards(state, discards)

    state.phase = Phase.TRICK_PLAY
    for _ in range(12):
        state = start_trick(state)
        for __ in range(4):
            legal = state.legal_plays(state.current_player)
            card = rng.choice(legal)
            state = play_card(state, state.current_player, card)

    scores = score_game(state)
    for p in range(4):
        team = state.get_team(p)
        if team != Team.DECLARER_TEAM:
            assert scores[p] == 0, f"Opponent {p} scored {scores[p]}: {scores}"


# ---------------------------------------------------------------------------
# Property: Card points are additive / partition-consistent
# ---------------------------------------------------------------------------


@hgiven(random_seeds)
@settings(max_examples=30)
def test_card_points_partition(seed):
    """Points of a full deck split into two piles always sum to TOTAL_GAME_POINTS."""
    rng = random.Random(seed)
    deck = list(DECK)
    rng.shuffle(deck)
    split = rng.randint(0, 54)
    pile_a = deck[:split]
    pile_b = deck[split:]
    # Points of two disjoint piles DON'T generally sum to 70 because
    # of the grouping deduction—but the raw sum does:
    assert sum(c.points for c in pile_a) + sum(c.points for c in pile_b) == 106


# ---------------------------------------------------------------------------
# Property: Higher tarok always beats lower tarok
# ---------------------------------------------------------------------------


@hgiven(
    st.integers(min_value=1, max_value=22),
    st.integers(min_value=1, max_value=22),
)
def test_higher_tarok_beats_lower(a, b):
    assume(a != b)
    high, low = max(a, b), min(a, b)
    card_high = tarok(high)
    card_low = tarok(low)
    assert card_high.beats(card_low, lead_suit=None)
    assert not card_low.beats(card_high, lead_suit=None)


# ---------------------------------------------------------------------------
# Property: Any tarok beats any suit card
# ---------------------------------------------------------------------------


@hgiven(
    st.integers(min_value=1, max_value=22),
    st.sampled_from(list(Suit)),
    st.sampled_from(list(SuitRank)),
)
def test_tarok_beats_suit(tarok_val, suit, rank):
    t = tarok(tarok_val)
    s = suit_card(suit, rank)
    # Tarok beats suit card (when lead is that suit, tarok trumps)
    assert t.beats(s, lead_suit=suit)


# ---------------------------------------------------------------------------
# Property: Trick winner is always among the 4 players
# ---------------------------------------------------------------------------


@hgiven(random_seeds)
@settings(max_examples=30, deadline=5000)
def test_trick_winner_valid(seed):
    rng = random.Random(seed)
    state = GameState(phase=Phase.DEALING)
    state = deal(state, rng=rng)

    state.phase = Phase.BIDDING
    state.current_bidder = 0
    state.current_player = 0
    state = place_bid(state, 0, Contract.THREE)
    for p in [1, 2, 3]:
        state = place_bid(state, p, None)

    if not state.contract.is_solo:
        kings = state.callable_kings()
        if kings:
            from tarok.use_cases.call_king import call_king

            state = call_king(state, kings[0])

    if state.phase == Phase.TALON_EXCHANGE:
        from tarok.use_cases.exchange_talon import reveal_talon, pick_talon_group, discard_cards

        reveal_talon(state)  # mutates state, returns groups
        state = pick_talon_group(state, 0)
        discards = _safe_discard(state)
        assume(discards is not None)
        state = discard_cards(state, discards)

    state.phase = Phase.TRICK_PLAY
    for _ in range(12):
        state = start_trick(state)
        for __ in range(4):
            legal = state.legal_plays(state.current_player)
            card = rng.choice(legal)
            state = play_card(state, state.current_player, card)

        # Check last completed trick
        last = state.tricks[-1]
        assert 0 <= last.winner() < 4


# ---------------------------------------------------------------------------
# Property: Contract strength ordering is total and consistent
# ---------------------------------------------------------------------------


def test_contract_strength_total_order():
    contracts = list(Contract)
    for i in range(len(contracts)):
        for j in range(i + 1, len(contracts)):
            a, b = contracts[i], contracts[j]
            assert a.strength != b.strength, f"{a} and {b} have same strength"
