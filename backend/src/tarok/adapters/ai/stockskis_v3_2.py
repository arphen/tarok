"""StockŠkis v3.2 — hybrid heuristic blending v2 and v3 strengths.

Designed to be a single bot that an RL agent can't "crack" by specializing
against either v2 or v3 individually.  Key design principles:

1. **Bidding**: v3-style hand evaluation (interacting features, queen backing,
   tarok density) but with thresholds halfway between v2 and v3 so it doesn't
   over-bid or under-bid.
2. **Card tracker**: v3-style (game phase, tricks won, points taken, inference)
   — strictly more information is always better.
3. **Klop / Berač**: v3 phase-aware ducking with v2's slightly more forgiving
   acceptance threshold in last seat (blend of 3 and 4 points).
4. **Leading**: v3 phase-aware logic for taroks, but with toned-down early
   aggression in suit leads.  v3's void-tracking for defense is kept.
5. **Following**: v3 positional awareness (2nd/3rd/4th seat) with v2's more
   conservative šmir weights so the RL agent can't exploit over-generous
   point feeding.
6. **Defending**: v3's void tracking (this is strictly better than v2).
7. **Announcements**: Midway threshold — kontra at 7+ taroks + škis, pagat
   ultimo at 7+ taroks.
"""

from __future__ import annotations

import math
import random

from tarok.entities.card import (
    Card,
    CardType,
    Suit,
    SuitRank,
    PAGAT,
    MOND,
    SKIS,
    DECK,
)
from tarok.entities.game_state import (
    Announcement,
    Contract,
    GameState,
    Phase,
    Trick,
)


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

def _worth_over(card: Card) -> int:
    if card.card_type == CardType.TAROK:
        return 10 + card.value
    assert card.suit is not None
    suit_base = {Suit.HEARTS: 33, Suit.DIAMONDS: 41, Suit.CLUBS: 49, Suit.SPADES: 57}
    return suit_base[card.suit] + card.value


_ALL_CARDS: list[Card] = list(DECK)


# ---------------------------------------------------------------------------
# Enhanced card tracker (from v3 — strictly better than v2)
# ---------------------------------------------------------------------------

class _CardTracker:
    """Card counting + inference about opponent holdings."""

    __slots__ = (
        "played", "remaining", "hand", "taroks_remaining",
        "suits_remaining", "taroks_in_hand", "suit_counts",
        "player_voids", "num_players", "tricks_won_by",
        "points_taken_by", "hand_set",
    )

    def __init__(self, state: GameState, player_idx: int):
        self.num_players = state.num_players
        self.hand = state.hands[player_idx]
        self.hand_set = set(self.hand)

        self.played: set[Card] = set()
        for trick in state.tricks:
            for _, card in trick.cards:
                self.played.add(card)
        if state.current_trick:
            for _, card in state.current_trick.cards:
                self.played.add(card)
        for card in state.put_down:
            self.played.add(card)

        self.remaining: set[Card] = set(_ALL_CARDS) - self.played - self.hand_set

        self.taroks_in_hand = [c for c in self.hand if c.card_type == CardType.TAROK]
        self.taroks_remaining = [c for c in self.remaining if c.card_type == CardType.TAROK]

        self.suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
        for c in self.hand:
            if c.suit is not None:
                self.suit_counts[c.suit] += 1

        self.suits_remaining: dict[Suit, list[Card]] = {s: [] for s in Suit}
        for c in self.remaining:
            if c.suit is not None:
                self.suits_remaining[c.suit].append(c)

        self.player_voids: dict[int, set[Suit]] = {p: set() for p in range(self.num_players)}
        for trick in state.tricks:
            if not trick.cards:
                continue
            lead_card = trick.cards[0][1]
            if lead_card.suit is not None:
                for p, c in trick.cards[1:]:
                    if c.card_type == CardType.TAROK and p != player_idx:
                        self.player_voids[p].add(lead_card.suit)

        self.tricks_won_by: dict[int, int] = {p: 0 for p in range(self.num_players)}
        self.points_taken_by: dict[int, int] = {p: 0 for p in range(self.num_players)}
        for trick in state.tricks:
            if trick.cards:
                lead_suit = trick.cards[0][1].suit if trick.cards[0][1].card_type != CardType.TAROK else None
                best_card = trick.cards[0][1]
                winner = trick.cards[0][0]
                for p, c in trick.cards[1:]:
                    if c.beats(best_card, lead_suit):
                        best_card = c
                        winner = p
                self.tricks_won_by[winner] += 1
                pts = sum(c.points for _, c in trick.cards)
                self.points_taken_by[winner] += pts

    def higher_taroks_out(self, value: int) -> int:
        return sum(1 for c in self.taroks_remaining if c.value > value)

    def suit_is_master(self, suit: Suit, value: int) -> bool:
        for c in self.suits_remaining[suit]:
            if c.value > value:
                return False
        return True

    def tricks_left(self, state: GameState) -> int:
        return max(1, len(self.hand) - len(state.tricks))

    def game_phase(self, state: GameState) -> str:
        played = len(state.tricks)
        if played <= 3:
            return "early"
        elif played <= 8:
            return "mid"
        else:
            return "late"

    def opponents_max_tarok(self) -> int:
        if not self.taroks_remaining:
            return 0
        return max(c.value for c in self.taroks_remaining)

    def count_remaining_in_suit(self, suit: Suit) -> int:
        return len(self.suits_remaining.get(suit, []))

    def opponent_likely_has_suit(self, player: int, suit: Suit) -> bool:
        return suit not in self.player_voids[player]


# ---------------------------------------------------------------------------
# Bidding — v3 evaluation, thresholds between v2 and v3
# ---------------------------------------------------------------------------

def _evaluate_hand_for_bid(hand: list[Card], num_players: int = 4) -> list[int]:
    taroks = []
    kings = 0
    suits: dict[Suit, list[Card]] = {s: [] for s in Suit}

    for card in hand:
        if card.card_type == CardType.TAROK:
            taroks.append(card)
        else:
            assert card.suit is not None
            suits[card.suit].append(card)
            if card.is_king:
                kings += 1

    tarok_count = len(taroks)
    high_taroks = sum(1 for t in taroks if t.value >= 15)
    has_skis = any(t.value == SKIS for t in taroks)
    has_mond = any(t.value == MOND for t in taroks)
    has_pagat = any(t.value == PAGAT for t in taroks)

    voids = sum(1 for s in Suit if len(suits[s]) == 0)
    singletons = sum(1 for s in Suit if len(suits[s]) == 1)

    # v3-style interacting feature evaluation
    rating = 0.0
    rating += tarok_count * 6
    rating += high_taroks * 4.5          # between v2 (4) and v3 (5)
    if has_skis:
        rating += 13                     # between v2 (12) and v3 (14)
    if has_mond:
        rating += 10.5                   # between v2 (10) and v3 (11)
        if has_skis:
            rating += 3                  # mond protected by škis (v3 feature)
    if has_pagat:
        if tarok_count >= 7:
            rating += 6.5               # between v2 (5) and v3 (8)
        elif tarok_count >= 5:
            rating += 3
        else:
            rating -= 2                  # unprotected pagat is liability (v3)

    # Suit evaluation — v3 style with queen backing
    rating += kings * 8
    for s in Suit:
        count = len(suits[s])
        has_king = any(c.is_king for c in suits[s])
        has_queen = any(c.value == SuitRank.QUEEN.value for c in suits[s])

        if count == 0:
            rating += 7.5               # between v2 (7) and v3 (8)
        elif count == 1:
            if has_king:
                rating += 4
            else:
                rating += 2.5           # between v2 (2) and v3 (3)
        elif count == 2 and has_king:
            rating += 2
        elif count >= 3 and not has_king:
            rating -= count * 2

        if has_king and has_queen and count >= 2:
            rating += 2                  # queen backing (v3 feature)

    # Tarok density bonus (v3 feature)
    if tarok_count >= 8:
        rating += (tarok_count - 7) * 3

    max_rating = 125.0                   # between v2 (120) and v3 (130)
    ratio = min(1.0, rating / max_rating)
    is_3p = num_players == 3

    # Thresholds: midpoints between v2 and v3
    TRI = 0.245 + (0.10 if is_3p else 0.0)
    DVE = 0.31 + (0.12 if is_3p else 0.0)
    ENA = 0.39 + (0.13 if is_3p else 0.0)
    SOLO_TRI = 0.51
    SOLO_DVA = 0.59
    SOLO_ENA = 0.67
    SOLO_BREZ = 0.77

    thresholds = [
        (0, TRI), (1, DVE), (2, ENA),
        (3, SOLO_TRI), (4, SOLO_DVA), (5, SOLO_ENA), (6, SOLO_BREZ),
    ]

    # Berač: v3's stricter gating (no kings, no singletons from v4)
    has_all_suits = all(len(suits[s]) > 0 for s in Suit)
    has_singleton = any(len(suits[s]) == 1 for s in Suit)
    can_berac = (
        ratio < 0.17                     # between v2 (0.18) and v3 (0.16)
        and tarok_count <= 2
        and has_all_suits
        and not has_singleton            # v4 improvement
        and high_taroks == 0
        and kings == 0                   # v3 requirement
    )

    modes = [mode_id for mode_id, threshold in thresholds if ratio >= threshold]
    if can_berac:
        modes.append(7)

    return modes if modes else [-1]


_MODE_TO_CONTRACT: dict[int, Contract] = {
    0: Contract.THREE,
    1: Contract.TWO,
    2: Contract.ONE,
    3: Contract.SOLO_THREE,
    4: Contract.SOLO_TWO,
    5: Contract.SOLO_ONE,
    6: Contract.SOLO,
    7: Contract.BERAC,
}


# ---------------------------------------------------------------------------
# Card-play: split by game type and phase
# ---------------------------------------------------------------------------

def _evaluate_card_play(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    is_leading: bool,
    tracker: _CardTracker,
) -> float:
    is_klop = state.contract is not None and state.contract.is_klop
    is_berac = state.contract is not None and state.contract.is_berac

    if is_klop or is_berac:
        return _eval_klop_berac(card, hand, state, player_idx, is_leading, tracker)

    is_declarer = state.declarer == player_idx
    is_partner = state.partner is not None and state.partner == player_idx
    is_playing = is_declarer or is_partner

    if is_leading:
        return _eval_leading(card, hand, state, player_idx, is_playing, tracker)
    else:
        return _eval_following(card, hand, state, player_idx, is_playing, tracker)


# ---------------------------------------------------------------------------
# Klop / Berač — v3 phase-aware with slightly softer acceptance
# ---------------------------------------------------------------------------

def _eval_klop_berac(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    is_leading: bool,
    tracker: _CardTracker,
) -> float:
    wo = _worth_over(card)
    pts = card.points
    score = 0.0
    phase = tracker.game_phase(state)

    if is_leading:
        if card.card_type == CardType.TAROK:
            score = -wo * 2.3             # between v2 (-2.0) and v3 (-2.5)
            if card.value == PAGAT:
                if len(tracker.taroks_in_hand) == 1 and phase == "late":
                    score = -5
                else:
                    score -= 500
            # Late game safe tarok lead (v3 feature)
            if phase == "late" and card.value <= 5 and tracker.higher_taroks_out(card.value) == 0:
                score = 10
        else:
            assert card.suit is not None
            count = tracker.suit_counts.get(card.suit, 1)
            remaining = tracker.count_remaining_in_suit(card.suit)

            opponents_void = sum(
                1 for p in range(state.num_players)
                if p != player_idx and card.suit in tracker.player_voids[p]
            )

            if opponents_void > 0:
                score = -wo * 3.5 - opponents_void * 22  # between v2 and v3
            elif remaining == 0:
                score = -wo * 3           # v3: we're the only one with this suit
            else:
                score = -wo + remaining * 2.5 - count * 2.5
                if count == 1:
                    score += 5            # prefer voiding (v3 feature)
    else:
        trick = state.current_trick
        if trick and trick.cards:
            best_card = trick.cards[0][1]
            for _, c in trick.cards:
                if c.beats(best_card, trick.lead_suit):
                    best_card = c

            is_last = len(trick.cards) == state.num_players - 1
            trick_pts = sum(c.points for _, c in trick.cards) + pts

            if card.beats(best_card, trick.lead_suit):
                # Accept in last seat if trick value is very low
                if is_last and trick_pts <= 3:  # v3 threshold (v2 used 4)
                    score = -wo * 0.5
                else:
                    score = -wo * 3 - trick_pts * 10
            else:
                score = pts * 5 + wo * 0.25  # between v2 (0.2) and v3 (0.3)
                # Dump cards from dangerous suits (v3 feature)
                if card.suit is not None:
                    suit_danger = sum(
                        1 for p in range(state.num_players)
                        if p != player_idx and card.suit in tracker.player_voids[p]
                    )
                    if suit_danger > 0:
                        score += pts * 3
    return score


# ---------------------------------------------------------------------------
# Leading — v3 phase-aware with moderated aggression
# ---------------------------------------------------------------------------

def _eval_leading(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    is_playing: bool,
    tracker: _CardTracker,
) -> float:
    wo = _worth_over(card)
    pts = card.points
    score = 0.0
    phase = tracker.game_phase(state)
    tricks_left = tracker.tricks_left(state)

    if card.card_type == CardType.TAROK:
        higher_out = tracker.higher_taroks_out(card.value)

        if card.value == SKIS:
            # v3 phase-aware škis leading
            if phase == "early":
                score = 85                # between v2 (80) and v3 (90)
            elif phase == "mid":
                score = 65                # between v2 (~40) and v3 (70)
            else:
                if tricks_left <= 1:
                    score = -300
                else:
                    score = 45

        elif card.value == MOND:
            skis_out = any(c.value == SKIS for c in tracker.taroks_remaining)
            has_skis = any(c.value == SKIS for c in tracker.taroks_in_hand)

            if skis_out and not has_skis:
                score = -120              # between v2 (-100) and v3 (-150)
            elif has_skis:
                score = 65                # safe: we have both
            else:
                # Škis is gone, mond is master
                score = 65 + len(tracker.taroks_remaining) * 3

        elif card.value == PAGAT:
            if tricks_left <= 2 and len(tracker.taroks_in_hand) <= 2:
                score = 250               # between v2 (200) and v3 (300)
            elif tricks_left == 1:
                score = 400               # v3: last trick, must go now
            else:
                score = -220              # between v2 (-200) and v3 (-250)

        else:
            # Regular tarok
            if higher_out == 0:
                score = 52 + wo           # between v2 (50) and v3 (55)
                if is_playing:
                    score += 8            # declarer bonus (v3 gives 10)
            else:
                if is_playing:
                    if phase == "early":
                        score = math.pow(max(0, wo - 11) / 3, 1.5) + 7  # between v2 (5) and v3 (10)
                    else:
                        score = math.pow(max(0, wo - 11) / 3, 1.5)
                else:
                    if higher_out <= 2:   # v3 threshold (v2 used ≤1)
                        score = 27 + wo * 0.5
                    else:
                        score = wo * 0.25 # between v2 (0.3) and v3 (0.2)
    else:
        assert card.suit is not None
        count = tracker.suit_counts[card.suit]
        remaining = tracker.count_remaining_in_suit(card.suit)

        if is_playing:
            if card.is_king:
                if tracker.suit_is_master(card.suit, card.value):
                    score = 28 + pts * 2  # between v2 (25) and v3 (30)
                elif count >= 3:
                    score = 15 + pts      # v3: king with support
                else:
                    score = pts - 6       # between v2 (-5) and v3 (-8)
            elif count == 1:
                score = 16 - pts          # between v2 (15) and v3 (18)
            elif count == 2:
                if not card.is_king:
                    score = 12 - pts      # v3: doubleton voiding
                else:
                    score = pts
            else:
                score = -pts * 1.5 - count * 2
        else:
            # Defender leading — v3 style with void awareness
            if count >= 3:
                if card.is_king:
                    score = pts * 2 + 3
                else:
                    score = count * 3 - pts * 1.5
            elif count == 1:
                score = 9 - pts           # between v2 (10) and v3 (8)
            else:
                score = -count * 2 - pts

            # v3: Don't lead suits declarer is void in (they'll trump)
            if state.declarer is not None:
                decl = state.declarer
                if card.suit in tracker.player_voids.get(decl, set()):
                    score -= 15
                else:
                    # Lead suits partner is void in (they can trump)
                    if state.partner is not None and state.partner != player_idx:
                        partner = state.partner
                        if card.suit in tracker.player_voids.get(partner, set()):
                            score += 10

    return score


# ---------------------------------------------------------------------------
# Following — v3 positional play with moderated šmir
# ---------------------------------------------------------------------------

def _eval_following(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    is_playing: bool,
    tracker: _CardTracker,
) -> float:
    wo = _worth_over(card)
    pts = card.points
    score = 0.0
    tricks_left = tracker.tricks_left(state)
    phase = tracker.game_phase(state)

    trick = state.current_trick
    if trick is None or not trick.cards:
        return 0.0

    lead_card = trick.cards[0][1]
    num_played = len(trick.cards)
    is_last = num_played == state.num_players - 1

    best_card = lead_card
    best_player = trick.cards[0][0]
    for p, c in trick.cards:
        if c.beats(best_card, trick.lead_suit):
            best_card = c
            best_player = p

    if is_playing:
        best_is_ally = (
            best_player == state.declarer
            or (state.partner is not None and best_player == state.partner)
        )
    else:
        best_is_ally = (
            best_player != state.declarer
            and (state.partner is None or best_player != state.partner)
        )

    trick_pts = sum(c.points for _, c in trick.cards) + pts
    would_win = card.beats(best_card, trick.lead_suit)

    # === Special card handling ===

    # Pagat
    if card.card_type == CardType.TAROK and card.value == PAGAT:
        if tricks_left <= 2 and len(tracker.taroks_in_hand) <= 2:
            return 500 if would_win else -500  # v2 values (v3 same)
        elif tricks_left == 1:
            return 600 if would_win else -600  # v3 values
        else:
            return -225                     # between v2 (-200) and v3 (-250)

    # Mond
    if card.card_type == CardType.TAROK and card.value == MOND:
        if not would_win:
            return -550                     # between v2 (-500) and v3 (-600)
        # Check if škis can still capture (v3 feature)
        skis_out = any(c.value == SKIS for c in tracker.taroks_remaining)
        if skis_out and not is_last:
            return -100
        return trick_pts * 2 + 35          # between v2 (30) and v3 (40)

    # Škis
    if card.card_type == CardType.TAROK and card.value == SKIS:
        if tricks_left <= 1:
            return -350                     # between v2 (-300) and v3 (-400)
        return trick_pts * 2.2 + 32         # between v2 and v3

    # === Normal following logic ===
    if would_win:
        if best_is_ally:
            if is_last:
                # Ally wins, we're last — šmir
                score = pts * 4.2           # between v2 (4.0) and v3 (4.5)
            else:
                # Don't over-trump ally
                score = -wo * 0.55 - math.pow(wo / 3, 1.5)
        else:
            if is_last:
                # Last seat — win cheaply with lowest possible card
                score = trick_pts * 3.2 - wo * 0.9  # between v2 and v3
            elif num_played == 1:
                # v3: 2nd seat awareness — risk of over-trump
                if card.card_type == CardType.TAROK and tracker.higher_taroks_out(card.value) > 0:
                    score = trick_pts * 0.8 - wo * 0.3
                else:
                    score = trick_pts * 1.8 + wo * 0.2
            else:
                # 3rd seat
                score = trick_pts * 1.8 + wo * 0.25
    else:
        # Can't win
        if best_is_ally:
            # Šmir — feed points to ally (moderated vs v3)
            if is_last:
                score = pts * 5.5           # between v2 (5.0) and v3 (6.0)
            elif phase == "late":
                score = pts * 3.5           # between v2 (2.5) and v3 (4.0)
            else:
                score = pts * 2.7           # between v2 (2.5) and v3 (3.0)
        else:
            # Opponent winning — dump cheapest cards
            score = -(pts * 3) - wo * 0.35  # between v2 (0.3) and v3 (0.4)
            # v3: dump liability cards from suits opponents are void in
            if card.suit is not None:
                for p in range(state.num_players):
                    if p != player_idx and card.suit in tracker.player_voids[p]:
                        score += 2

    return score


# ---------------------------------------------------------------------------
# Talon — v3 style (marginally better)
# ---------------------------------------------------------------------------

def _evaluate_talon_group(
    group: list[Card],
    hand: list[Card],
    called_king: Card | None,
) -> float:
    suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
    for c in hand:
        if c.suit is not None:
            suit_counts[c.suit] += 1

    total = 0.0
    for card in group:
        total += card.points * 2 + _worth_over(card) * 0.3
        if card.card_type == CardType.TAROK:
            total += 9                    # between v2 (8) and v3 (10)
        if called_king and card.suit == called_king.suit:
            total += 4.5                  # between v2 (4) and v3 (5)

    combined = hand + group
    new_suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
    for c in combined:
        if c.suit is not None:
            new_suit_counts[c.suit] += 1

    called_suit = called_king.suit if called_king else None
    for s in Suit:
        if s == called_suit:
            continue
        if suit_counts[s] <= 1 and new_suit_counts[s] <= 1:
            total += 5.5                  # between v2 (5) and v3 (6)
        if suit_counts[s] == 0 and new_suit_counts[s] == 0:
            total += 3.5                  # between v2 (3) and v3 (4)

    return total


# ---------------------------------------------------------------------------
# Discard — aggressive void-building (same approach as v2/v3)
# ---------------------------------------------------------------------------

def _choose_discards(
    hand: list[Card],
    must_discard: int,
    called_king: Card | None,
    contract: Contract | None,
) -> list[Card]:
    discardable = [
        c for c in hand
        if c.card_type != CardType.TAROK and not c.is_king
    ]

    if len(discardable) < must_discard:
        extra_taroks = sorted(
            [c for c in hand if c.card_type == CardType.TAROK
             and c.value not in (PAGAT, MOND, SKIS)
             and c not in discardable],
            key=lambda c: c.value,
        )
        discardable.extend(extra_taroks)

    called_suit = called_king.suit if called_king else None

    by_suit: dict[Suit, list[Card]] = {s: [] for s in Suit}
    for c in discardable:
        if c.suit is not None:
            by_suit[c.suit].append(c)

    hand_suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
    for c in hand:
        if c.suit is not None:
            hand_suit_counts[c.suit] += 1

    result: list[Card] = []

    void_candidates = sorted(
        [(s, cards) for s, cards in by_suit.items()
         if s != called_suit and cards],
        key=lambda x: hand_suit_counts[x[0]],
    )

    for suit, cards in void_candidates:
        if len(result) >= must_discard:
            break
        suit_in_hand = [c for c in hand if c.suit == suit and not c.is_king]
        discardable_from_suit = [c for c in suit_in_hand if c in discardable]
        if len(discardable_from_suit) + len(result) <= must_discard:
            for c in discardable_from_suit:
                if c not in result:
                    result.append(c)

    if len(result) < must_discard:
        remaining_cards = [c for c in discardable if c not in result]
        remaining_cards.sort(key=lambda c: (c.points, _worth_over(c)))
        for c in remaining_cards:
            if len(result) >= must_discard:
                break
            result.append(c)

    return result[:must_discard]


# ---------------------------------------------------------------------------
# Announcements — midway thresholds
# ---------------------------------------------------------------------------

def _should_announce(
    state: GameState,
    player_idx: int,
    hand: list[Card],
    tracker: _CardTracker,
) -> int:
    is_declarer = state.declarer == player_idx
    is_partner = state.partner is not None and state.partner == player_idx
    is_playing = is_declarer or is_partner

    taroks = [c for c in hand if c.card_type == CardType.TAROK]
    tarok_count = len(taroks)
    has_pagat = any(c.value == PAGAT for c in taroks)
    has_mond = any(c.value == MOND for c in taroks)
    has_skis = any(c.value == SKIS for c in taroks)
    kings_in_hand = [c for c in hand if c.is_king]

    if not is_playing:
        existing = state.kontra_levels.get("game")
        if existing is None:
            # Kontra at 7+ taroks + škis (v2 = 7, v3 = 8)
            if tarok_count >= 7 and has_skis:
                return 5
        return 0

    if has_pagat and has_mond and has_skis:
        already = any(
            Announcement.TRULA in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 1

    if len(kings_in_hand) == 4:
        already = any(
            Announcement.KINGS in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 2

    # Pagat ultimo at 7+ taroks (v2 = 7, v3 = 8)
    if has_pagat and tarok_count >= 7:
        already = any(
            Announcement.PAGAT_ULTIMO in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 3

    return 0


# ---------------------------------------------------------------------------
# StockŠkis v3.2 Player
# ---------------------------------------------------------------------------

class StockSkisPlayerV3_2:
    """Hybrid heuristic bot blending v2 conservative play with v3 awareness.

    Parameters
    ----------
    name : str
        Display name.
    strength : float
        0.0 = random, 1.0 = pure heuristic.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        name: str = "StockŠkis-v3.2",
        strength: float = 1.0,
        seed: int | None = None,
    ):
        self._name = name
        self.strength = max(0.0, min(1.0, strength))
        self._rng = random.Random(seed)
        self.experiences: list = []

    @property
    def name(self) -> str:
        return self._name

    def set_training(self, training: bool) -> None:
        pass

    def clear_experiences(self) -> None:
        pass

    def finalize_game(self, reward: float) -> None:
        pass

    async def choose_bid(
        self,
        state: GameState,
        player_idx: int,
        legal_bids: list[Contract | None],
    ) -> Contract | None:
        hand = state.hands[player_idx]
        modes = _evaluate_hand_for_bid(hand, state.num_players)

        if -1 in modes and len(modes) == 1:
            return None

        desired = [_MODE_TO_CONTRACT[m] for m in modes if m in _MODE_TO_CONTRACT]
        desired.sort(key=lambda c: c.strength, reverse=True)

        for contract in desired:
            if contract in legal_bids:
                return contract
        return None

    async def choose_king(
        self,
        state: GameState,
        player_idx: int,
        callable_kings: list[Card],
    ) -> Card:
        hand = state.hands[player_idx]
        best_king = callable_kings[0]
        best_score = -1

        for king in callable_kings:
            assert king.suit is not None
            count = sum(1 for c in hand if c.suit == king.suit)
            low = sum(1 for c in hand if c.suit == king.suit and c.value <= 4)
            has_queen = any(
                c.suit == king.suit and c.value == SuitRank.QUEEN.value
                for c in hand
            )
            score = count * 3 + low * 1.5  # between v2 (low*1) and v3 (low*2)
            if has_queen:
                score += 1                 # v3 queen support
            if score > best_score:
                best_score = score
                best_king = king
        return best_king

    async def choose_talon_group(
        self,
        state: GameState,
        player_idx: int,
        talon_groups: list[list[Card]],
    ) -> int:
        hand = state.hands[player_idx]
        best_idx = 0
        best_score = -float("inf")
        for i, group in enumerate(talon_groups):
            score = _evaluate_talon_group(group, hand, state.called_king)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    async def choose_discard(
        self,
        state: GameState,
        player_idx: int,
        must_discard: int,
    ) -> list[Card]:
        hand = state.hands[player_idx]
        return _choose_discards(hand, must_discard, state.called_king, state.contract)

    async def choose_announcements(
        self,
        state: GameState,
        player_idx: int,
    ) -> list[Announcement]:
        return []

    async def choose_announce_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> int:
        hand = state.hands[player_idx]
        tracker = _CardTracker(state, player_idx)
        return _should_announce(state, player_idx, hand, tracker)

    async def choose_card(
        self,
        state: GameState,
        player_idx: int,
        legal_plays: list[Card],
    ) -> Card:
        if len(legal_plays) == 1:
            return legal_plays[0]

        hand = state.hands[player_idx]
        tracker = _CardTracker(state, player_idx)
        is_leading = (
            state.current_trick is None or len(state.current_trick.cards) == 0
        )

        evals: list[tuple[Card, float]] = []
        for card in legal_plays:
            score = _evaluate_card_play(
                card, hand, state, player_idx, is_leading, tracker,
            )
            if self.strength < 1.0:
                noise = self._rng.gauss(0, 1) * (1 - self.strength) * 20
                score += noise
            evals.append((card, score))

        evals.sort(key=lambda x: x[1], reverse=True)
        return evals[0][0]
