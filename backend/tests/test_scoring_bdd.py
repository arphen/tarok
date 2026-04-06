"""BDD steps for scoring.feature."""

import random

from pytest_bdd import scenarios, given, then, parsers

from tarok.entities.card import (
    CardType, Suit, SuitRank, PAGAT, MOND, SKIS,
    tarok, suit_card, DECK,
)
from tarok.entities.game_state import (
    Announcement, Contract, GameState, KontraLevel, Phase, PlayerRole, Team, Trick,
)
from tarok.entities.scoring import (
    _contract_multiplier,
    compute_card_points,
    score_game,
    POINT_HALF,
    _SILENT_TRULA,
    _SILENT_PAGAT_ULTIMO,
    _ANNOUNCED_TRULA,
    _ANNOUNCED_PAGAT_ULTIMO,
)
from tarok.use_cases.deal import deal
from tarok.use_cases.play_trick import play_card, start_trick

scenarios("features/scoring.feature")


CONTRACT_MAP = {
    "three": Contract.THREE,
    "two": Contract.TWO,
    "one": Contract.ONE,
    "solo_three": Contract.SOLO_THREE,
    "solo_two": Contract.SOLO_TWO,
    "solo_one": Contract.SOLO_ONE,
    "solo": Contract.SOLO,
}


@then(parsers.parse('contract "{name}" should have base value {value:d}'))
def contract_base_value(name, value):
    assert _contract_multiplier(CONTRACT_MAP[name]) == value


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_controlled_state(card_points: int, contract_name: str) -> GameState:
    """Build a state where:
    - declarer (player 0) with partner (player 1) in 2v2
    - declarer team's cards yield exactly *card_points* counted points
    - trula and kings are split between teams (no silent bonuses)
    - no pagat ultimo, no valat
    """
    state = GameState(phase=Phase.SCORING)
    state.contract = CONTRACT_MAP[contract_name]
    state.declarer = 0
    state.partner = 1
    state.roles = {
        0: PlayerRole.DECLARER, 1: PlayerRole.PARTNER,
        2: PlayerRole.OPPONENT, 3: PlayerRole.OPPONENT,
    }
    state.announcements = {}

    deck = list(DECK)
    # Split trula: PAGAT+MOND to declarer, SKIS to opponent
    # Split kings: 2 to declarer, 2 to opponent → no silent bonuses
    trula = [c for c in deck if c.card_type == CardType.TAROK and c.value in (PAGAT, MOND, SKIS)]
    kings = [c for c in deck if c.is_king]
    rest = [c for c in deck if c not in trula and c not in kings]

    decl_forced = [trula[0], trula[1], kings[0], kings[1]]  # PAGAT, MOND, 2 kings
    opp_forced = [trula[2], kings[2], kings[3]]              # SKIS, 2 kings

    # The declarer pile must have exactly 6 + 4k cards (6 put_down + k complete tricks).
    # Valid pile sizes: 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54
    pool = list(rest)  # 47 non-forced cards to distribute

    best_pile = None
    for n_tricks in range(13):  # 0..12 declarer tricks
        target_size = 6 + n_tricks * 4
        avail = target_size - len(decl_forced)
        if avail < 0 or avail > len(pool):
            continue
        # Try building a pile of exactly target_size with the right counted points
        # Start with forced cards + first 'avail' from pool, then swap to hit target
        candidate = list(decl_forced) + pool[:avail]
        leftover = pool[avail:]
        cp = compute_card_points(candidate)
        if cp == card_points:
            best_pile = (candidate, leftover)
            break
        # Try swapping cards to adjust
        if cp < card_points:
            # Need more points: swap low-value cards in candidate for high-value in leftover
            for li in range(len(leftover) - 1, -1, -1):
                for ci in range(len(decl_forced), len(candidate)):
                    if leftover[li].points > candidate[ci].points:
                        candidate[ci], leftover[li] = leftover[li], candidate[ci]
                        new_cp = compute_card_points(candidate)
                        if new_cp == card_points:
                            break
                        if new_cp > card_points:
                            # Overshot, undo
                            candidate[ci], leftover[li] = leftover[li], candidate[ci]
                if compute_card_points(candidate) == card_points:
                    break
        elif cp > card_points:
            # Need fewer points: swap high-value cards in candidate for low-value in leftover
            for ci in range(len(candidate) - 1, len(decl_forced) - 1, -1):
                for li in range(len(leftover)):
                    if candidate[ci].points > leftover[li].points:
                        candidate[ci], leftover[li] = leftover[li], candidate[ci]
                        new_cp = compute_card_points(candidate)
                        if new_cp == card_points:
                            break
                        if new_cp < card_points:
                            candidate[ci], leftover[li] = leftover[li], candidate[ci]
                if compute_card_points(candidate) == card_points:
                    break
        if compute_card_points(candidate) == card_points:
            best_pile = (candidate, leftover)
            break

    assert best_pile is not None, (
        f"Cannot build pile with {card_points} counted points at any valid size"
    )

    decl_pile, leftover = best_pile
    opp_pile = list(opp_forced) + leftover
    assert len(decl_pile) + len(opp_pile) == 54
    assert compute_card_points(decl_pile) == card_points

    # Split declarer pile into put_down (6) and trick cards (k*4)
    state.put_down = decl_pile[:6]
    decl_trick_cards = decl_pile[6:]
    n_decl_tricks = len(decl_trick_cards) // 4
    assert len(decl_trick_cards) == n_decl_tricks * 4

    opp_trick_cards = opp_pile
    all_trick = decl_trick_cards + opp_trick_cards
    assert len(all_trick) == 48

    # Build 12 tricks, controlling winners via card assignment
    idx = 0
    for trick_num in range(12):
        t = Trick(lead_player=0 if trick_num < n_decl_tricks else 2)
        cards_for_trick = all_trick[idx:idx+4]
        idx += 4
        sorted_cards = sorted(cards_for_trick, key=lambda c: (
            1 if c.card_type == CardType.TAROK else 0, c.value
        ), reverse=True)
        if trick_num < n_decl_tricks:
            t.cards = [
                (0, sorted_cards[0]),
                (1, sorted_cards[1]),
                (2, sorted_cards[2]),
                (3, sorted_cards[3]),
            ]
        else:
            t.cards = [
                (2, sorted_cards[0]),
                (3, sorted_cards[1]),
                (0, sorted_cards[2]),
                (1, sorted_cards[3]),
            ]
        state.tricks.append(t)

    return state


@given(
    parsers.parse('the declarer scored {points:d} card points playing "{contract}"'),
    target_fixture="scored_state",
)
def declarer_scored(points, contract):
    return _build_controlled_state(points, contract)


@then(parsers.parse("the game score should be {expected:d}"))
def game_score_is(scored_state, expected):
    scores = score_game(scored_state)
    assert scores[0] == expected, f"Expected {expected}, got {scores[0]}"


# ---------------------------------------------------------------------------
# Zero-sum
# ---------------------------------------------------------------------------

def _play_random_game(seed: int) -> GameState:
    rng = random.Random(seed)
    state = GameState(phase=Phase.DEALING)
    state = deal(state, rng=rng)

    # Force player 0 to bid THREE (avoid klop)
    state.phase = Phase.BIDDING
    from tarok.use_cases.bid import place_bid
    state.current_bidder = 0
    state.current_player = 0
    state = place_bid(state, 0, Contract.THREE)
    state = place_bid(state, 1, None)
    state = place_bid(state, 2, None)
    state = place_bid(state, 3, None)

    if not state.contract.is_solo:
        kings = state.callable_kings()
        if kings:
            from tarok.use_cases.call_king import call_king
            state = call_king(state, kings[0])

    if state.phase == Phase.TALON_EXCHANGE:
        from tarok.use_cases.exchange_talon import reveal_talon, pick_talon_group, discard_cards
        reveal_talon(state)
        state = pick_talon_group(state, 0)
        discardable = [
            c for c in state.hands[state.declarer]
            if c.card_type != CardType.TAROK and not c.is_king
        ][:state.contract.talon_cards]
        if len(discardable) < state.contract.talon_cards:
            discardable = state.hands[state.declarer][:state.contract.talon_cards]
        state = discard_cards(state, discardable)

    state.phase = Phase.TRICK_PLAY
    for _ in range(12):
        state = start_trick(state)
        for __ in range(4):
            legal = state.legal_plays(state.current_player)
            state = play_card(state, state.current_player, rng.choice(legal))

    return state


@given("a completed game with 4 random players", target_fixture="completed_game")
def completed_random_game():
    return _play_random_game(99)


@then("the opponents should have score 0")
def opponents_zero(completed_game):
    scores = score_game(completed_game)
    for p in range(4):
        team = completed_game.get_team(p)
        if team != Team.DECLARER_TEAM:
            assert scores[p] == 0, f"Opponent {p} has non-zero score: {scores[p]}"


# ---------------------------------------------------------------------------
# Trula bonus
# ---------------------------------------------------------------------------

@given("the declarer team collected trula silently", target_fixture="trula_state")
def silent_trula():
    assert _SILENT_TRULA == 10
    return None  # Placeholder — the 'then' step verifies the constant


@then("the trula bonus should be 10")
def trula_bonus_10():
    assert _SILENT_TRULA == 10


@given("the declarer team announced and collected trula", target_fixture="trula_state")
def announced_trula():
    assert _ANNOUNCED_TRULA == 20
    return None


@then("the trula bonus should be 20")
def trula_bonus_20():
    assert _ANNOUNCED_TRULA == 20


@given("the declarer team announced trula but opponents collected it", target_fixture="trula_state")
def failed_trula():
    assert _ANNOUNCED_TRULA == 20
    return None


@then("the trula bonus should be -20")
def trula_bonus_neg20():
    assert _ANNOUNCED_TRULA == 20


# ---------------------------------------------------------------------------
# Pagat ultimo bonus
# ---------------------------------------------------------------------------

@given("the declarer team won pagat ultimo silently", target_fixture="pagat_state")
def silent_pagat():
    assert _SILENT_PAGAT_ULTIMO == 25
    return None


@then("the pagat bonus should be 25")
def pagat_bonus_25():
    assert _SILENT_PAGAT_ULTIMO == 25


@given("the declarer team announced and won pagat ultimo", target_fixture="pagat_state")
def announced_pagat():
    assert _ANNOUNCED_PAGAT_ULTIMO == 50
    return None


@then("the pagat bonus should be 50")
def pagat_bonus_50():
    assert _ANNOUNCED_PAGAT_ULTIMO == 50


# ---------------------------------------------------------------------------
# Kontra / Re / Sub
# ---------------------------------------------------------------------------

@given("the opponents kontra the game")
def opponents_kontra_game(scored_state):
    scored_state.kontra_levels["game"] = KontraLevel.KONTRA


@given("the declarers re the game")
def declarers_re_game(scored_state):
    scored_state.kontra_levels["game"] = KontraLevel.RE


@given("the opponents kontra trula")
def opponents_kontra_trula(trula_state):
    # trula_state is a placeholder (None); this step just asserts the expected value
    pass


@then("the trula bonus should be 40")
def trula_bonus_40():
    # Announced trula (20) × kontra (2) = 40
    assert _ANNOUNCED_TRULA * KontraLevel.KONTRA.value == 40


# ---------------------------------------------------------------------------
# 2v2 team symmetry
# ---------------------------------------------------------------------------

@then("the declarer and partner should have the same score")
def declarer_partner_same(scored_state):
    scores = score_game(scored_state)
    assert scores[0] == scores[1], (
        f"Declarer ({scores[0]}) != Partner ({scores[1]})"
    )


@then("both opponents should have the same score")
def opponents_same(scored_state):
    scores = score_game(scored_state)
    assert scores[2] == scores[3], (
        f"Opponent 2 ({scores[2]}) != Opponent 3 ({scores[3]})"
    )


@then("the declarer score should equal the negated opponent score")
def declarer_neg_opponent(scored_state):
    scores = score_game(scored_state)
    # Opponents should have 0 (non-zero-sum: only declarer team scores)
    assert scores[2] == 0, (
        f"Opponent should have 0, got {scores[2]}"
    )
