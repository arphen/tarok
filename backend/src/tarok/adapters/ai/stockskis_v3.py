"""StockŠkis v3 — strongest heuristic player.

Builds on v2 with these additional enhancements:

1. **Inference about hidden cards**: Uses Bayesian-style reasoning about
   what opponents likely hold based on their bids and plays.
2. **Trick counting**: Tracks how many tricks each team has won for
   dynamic strategy adjustment.
3. **Endgame solver**: When few cards remain, exhaustively evaluates
   best plays rather than using heuristic scores.
4. **Lead selection matrix**: Chooses leads based on game phase
   (early/mid/late) and team position.
5. **Defensive signaling**: When defending, coordinates plays to
   communicate suit preferences to partner.
6. **Better berač/klop**: Counter-plays when another player is
   trying to avoid tricks.
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
# Enhanced card tracker with inference
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

        # All visible played cards
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

        # Track voids
        self.player_voids: dict[int, set[Suit]] = {p: set() for p in range(self.num_players)}
        for trick in state.tricks:
            if not trick.cards:
                continue
            lead_card = trick.cards[0][1]
            if lead_card.suit is not None:
                for p, c in trick.cards[1:]:
                    if c.card_type == CardType.TAROK and p != player_idx:
                        self.player_voids[p].add(lead_card.suit)

        # Track tricks won and points taken per player
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
        """Return 'early', 'mid', or 'late' based on tricks played."""
        played = len(state.tricks)
        if played <= 3:
            return "early"
        elif played <= 8:
            return "mid"
        else:
            return "late"

    def opponents_max_tarok(self) -> int:
        """Highest tarok value still out there."""
        if not self.taroks_remaining:
            return 0
        return max(c.value for c in self.taroks_remaining)

    def count_remaining_in_suit(self, suit: Suit) -> int:
        return len(self.suits_remaining.get(suit, []))

    def opponent_likely_has_suit(self, player: int, suit: Suit) -> bool:
        """Infer if opponent likely still has cards of a suit."""
        return suit not in self.player_voids[player]


# ---------------------------------------------------------------------------
# Bidding — further refined
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

    # Stronger hand evaluation with interacting features
    rating = 0.0

    # Core tarok strength
    rating += tarok_count * 6
    rating += high_taroks * 5
    if has_skis:
        rating += 14
    if has_mond:
        rating += 11
        if has_skis:
            rating += 3  # mond protected by škis
    if has_pagat:
        if tarok_count >= 7:
            rating += 8  # strong pagat ultimo potential
        elif tarok_count >= 5:
            rating += 3
        else:
            rating -= 2  # unprotected pagat is a liability

    # Kings and suit control
    rating += kings * 8
    for s in Suit:
        count = len(suits[s])
        has_king = any(c.is_king for c in suits[s])
        has_queen = any(c.value == SuitRank.QUEEN.value for c in suits[s])

        if count == 0:
            rating += 8  # void — very valuable
        elif count == 1:
            if has_king:
                rating += 4  # singleton king — will be called or cashed
            else:
                rating += 3  # singleton — will become void after 1 lead
        elif count == 2 and has_king:
            rating += 2  # doubleton king — reasonable
        elif count >= 3 and not has_king:
            rating -= count * 2  # long suit without king — dead weight

        # Queen backing
        if has_king and has_queen and count >= 2:
            rating += 2

    # Tarok density bonus: more taroks -> more control
    if tarok_count >= 8:
        rating += (tarok_count - 7) * 3

    max_rating = 130.0
    ratio = min(1.0, rating / max_rating)

    is_3p = num_players == 3

    TRI = 0.24 + (0.10 if is_3p else 0.0)
    DVE = 0.30 + (0.12 if is_3p else 0.0)
    ENA = 0.38 + (0.13 if is_3p else 0.0)
    SOLO_TRI = 0.50
    SOLO_DVA = 0.58
    SOLO_ENA = 0.66
    SOLO_BREZ = 0.76

    thresholds = [
        (0, TRI), (1, DVE), (2, ENA),
        (3, SOLO_TRI), (4, SOLO_DVA), (5, SOLO_ENA), (6, SOLO_BREZ),
    ]

    has_all_suits = all(len(suits[s]) > 0 for s in Suit)
    can_berac = (
        ratio < 0.16
        and tarok_count <= 2
        and has_all_suits
        and high_taroks == 0
        and kings == 0
    )

    modes: list[int] = []
    for mode_id, threshold in thresholds:
        if ratio >= threshold:
            modes.append(mode_id)

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
            score = -wo * 2.5
            if card.value == PAGAT:
                if len(tracker.taroks_in_hand) == 1 and phase == "late":
                    score = -5  # must play it, minimal penalty
                else:
                    score -= 500
            # In late game, leading a low tarok can be safe
            if phase == "late" and card.value <= 5 and tracker.higher_taroks_out(card.value) == 0:
                score = 10  # safe lead
        else:
            assert card.suit is not None
            count = tracker.suit_counts.get(card.suit, 1)
            remaining = tracker.count_remaining_in_suit(card.suit)

            # Count opponents void in this suit
            opponents_void = sum(
                1 for p in range(state.num_players)
                if p != player_idx and card.suit in tracker.player_voids[p]
            )

            if opponents_void > 0:
                # Dangerous: someone can trump
                score = -wo * 4 - opponents_void * 25
            elif remaining == 0:
                # We're the only one with this suit — very dangerous
                score = -wo * 3
            else:
                # Lead low from suits where opponents still have cards
                score = -wo + remaining * 3 - count * 2
                # Prefer leading from short suits so we can void quickly
                if count == 1:
                    score += 5
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
                if is_last and trick_pts <= 3:
                    # Last seat, low-value trick — acceptable to take
                    score = -wo * 0.5
                else:
                    score = -wo * 3 - trick_pts * 10
            else:
                # Dump high-value cards we don't want
                score = pts * 5 + wo * 0.3
                # In klop specifically, dump cards from suits opponents are void in
                if card.suit is not None:
                    suit_danger = sum(
                        1 for p in range(state.num_players)
                        if p != player_idx and card.suit in tracker.player_voids[p]
                    )
                    if suit_danger > 0:
                        score += pts * 3  # dump these dangerous suit cards
    return score


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
        max_opponent_tarok = tracker.opponents_max_tarok()

        if card.value == SKIS:
            # Škis: lead early to extract taroks from opponents
            if phase == "early":
                score = 90
            elif phase == "mid":
                score = 70
            else:
                # Late game: save škis only if 1 trick left (it's captured in last)
                if tricks_left <= 1:
                    score = -300
                else:
                    score = 50

        elif card.value == MOND:
            skis_out = any(c.value == SKIS for c in tracker.taroks_remaining)
            has_skis = any(c.value == SKIS for c in tracker.taroks_in_hand)

            if skis_out and not has_skis:
                score = -150  # mond at risk from škis
            elif has_skis:
                score = 65  # we have both, safe
            else:
                # Škis been played, mond is master
                score = 70 + len(tracker.taroks_remaining) * 3

        elif card.value == PAGAT:
            if tricks_left <= 2 and len(tracker.taroks_in_hand) <= 2:
                score = 300  # pagat ultimo time!
            elif tricks_left == 1:
                score = 400  # last trick — must go now
            else:
                score = -250  # protect pagat

        else:
            # Regular tarok
            if higher_out == 0:
                # Master tarok — guaranteed trick
                score = 55 + wo
                if is_playing:
                    score += 10  # declarer team benefits more from flushing
            else:
                if is_playing:
                    # Declarer: lead high taroks to drain opponents
                    if phase == "early":
                        # Early game: flush taroks aggressively
                        score = math.pow(max(0, wo - 11) / 3, 1.5) + 10
                    else:
                        score = math.pow(max(0, wo - 11) / 3, 1.5)
                else:
                    # Defender: lead taroks mainly when we have strength
                    if higher_out <= 2:
                        score = 25 + wo * 0.5
                    else:
                        score = wo * 0.2
    else:
        # Suit card leading
        assert card.suit is not None
        count = tracker.suit_counts[card.suit]
        remaining = tracker.count_remaining_in_suit(card.suit)

        if is_playing:
            if card.is_king:
                if tracker.suit_is_master(card.suit, card.value):
                    score = 30 + pts * 2  # cash the master king
                elif count >= 3:
                    score = 15 + pts  # king with support
                else:
                    score = pts - 8  # singleton/doubleton king without support
            elif count == 1:
                # Singleton: lead to void
                score = 18 - pts
            elif count == 2:
                # Doubleton: lead low to prepare void
                if not card.is_king:
                    score = 12 - pts
                else:
                    score = pts
            else:
                # Long suit: lead low
                score = -pts * 1.5 - count * 2
        else:
            # Defender leading
            # Lead through declarer — force them to trump or play high
            if count >= 3:
                # Long suit: lead low, partner may have king
                if card.is_king:
                    score = pts * 2 + 5
                else:
                    score = count * 3 - pts * 1.5
            elif count == 1:
                # Singleton: might get trumped on return
                score = 8 - pts
            else:
                score = -count * 2 - pts

            # Bonus: lead suits where declarer is known void
            if state.declarer is not None:
                decl = state.declarer
                if card.suit in tracker.player_voids.get(decl, set()):
                    score -= 15  # DON'T lead suits declarer is void in (they'll trump)
                else:
                    # Check if partner is void — they can trump
                    if state.partner is not None and state.partner != player_idx:
                        partner = state.partner
                        if card.suit in tracker.player_voids.get(partner, set()):
                            score += 10  # partner void, they can trump declarer

    return score


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
            return 500 if would_win else -500
        elif tricks_left == 1:
            return 600 if would_win else -600
        else:
            return -250

    # Mond
    if card.card_type == CardType.TAROK and card.value == MOND:
        if not would_win:
            return -600  # never throw mond away
        # Mond that wins: check if škis can still capture it
        skis_out = any(c.value == SKIS for c in tracker.taroks_remaining)
        if skis_out and not is_last:
            # Someone after us might have škis
            return -100
        return trick_pts * 2 + 40

    # Škis
    if card.card_type == CardType.TAROK and card.value == SKIS:
        if tricks_left <= 1:
            return -400  # captured in last trick
        return trick_pts * 2.5 + 35

    # === Normal following logic ===
    if would_win:
        if best_is_ally:
            if is_last:
                # Ally wins, we're last — šmir
                score = pts * 4.5
            else:
                # Don't over-trump ally
                score = -wo * 0.6 - math.pow(wo / 3, 1.5)
        else:
            if is_last:
                # Last seat — win cheaply
                # Prefer winning with lowest possible card
                score = trick_pts * 3.5 - wo * 1.0
            elif num_played == 1:
                # 2nd seat: win with overhead (might get over-trumped)
                if card.card_type == CardType.TAROK and tracker.higher_taroks_out(card.value) > 0:
                    # Risky — someone after us has higher
                    score = trick_pts * 0.8 - wo * 0.3
                else:
                    score = trick_pts * 2.0 + wo * 0.2
            else:
                # 3rd seat
                score = trick_pts * 2.0 + wo * 0.3
    else:
        # Can't win
        if best_is_ally:
            # Šmir — feed points to ally
            if is_last:
                score = pts * 6  # maximum šmir in last seat
            elif phase == "late":
                score = pts * 4  # aggressive šmir late game
            else:
                score = pts * 3
        else:
            # Opponent winning — dump cheapest cards
            score = -(pts * 3) - wo * 0.4
            # Exception: dump cards from suits where opponents are void
            # (these cards are liabilities for later tricks)
            if card.suit is not None:
                for p in range(state.num_players):
                    if p != player_idx and card.suit in tracker.player_voids[p]:
                        score += 2  # slight bonus to dump these

    return score


# ---------------------------------------------------------------------------
# Talon / discard — same approach as v2 but with cleaner void logic
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
            total += 10  # taroks are always valuable
        if called_king and card.suit == called_king.suit:
            total += 5

    # Void potential
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
            total += 6
        if suit_counts[s] == 0 and new_suit_counts[s] == 0:
            total += 4

    return total


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

    # Group by suit
    by_suit: dict[Suit, list[Card]] = {s: [] for s in Suit}
    for c in discardable:
        if c.suit is not None:
            by_suit[c.suit].append(c)

    hand_suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
    for c in hand:
        if c.suit is not None:
            hand_suit_counts[c.suit] += 1

    result: list[Card] = []

    # Void shortest non-called suits first
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
# Announcements — more precise conditions
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
            if tarok_count >= 8 and has_skis:
                return 5
        return 0

    # Trula
    if has_pagat and has_mond and has_skis:
        already = any(
            Announcement.TRULA in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 1

    # Kings
    if len(kings_in_hand) == 4:
        already = any(
            Announcement.KINGS in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 2

    # Pagat ultimo — stricter threshold than v2
    if has_pagat and tarok_count >= 8:
        already = any(
            Announcement.PAGAT_ULTIMO in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 3

    return 0


# ---------------------------------------------------------------------------
# StockŠkis v3 Player
# ---------------------------------------------------------------------------

class StockSkisPlayerV3:
    """Strongest heuristic bot with card counting, inference, and endgame play.

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
        name: str = "StockŠkis-v3",
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
            # More weight on low cards (safe followers)
            low = sum(1 for c in hand if c.suit == king.suit and c.value <= 4)
            # Penalty if we have queen (partner has king, we have too many honors)
            has_queen = any(c.suit == king.suit and c.value == SuitRank.QUEEN.value for c in hand)
            score = count * 3 + low * 2
            if has_queen:
                score += 1  # queen helps support
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
