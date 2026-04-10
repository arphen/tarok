"""StockŠkis v2 — improved heuristic player.

Builds on the original StockŠkis port with these enhancements:

1. **Card counting**: Tracks played cards to reason about remaining threats.
2. **Positional awareness**: Plays differently in 2nd seat vs 4th seat.
3. **Void-building**: Smarter talon/discard choices to create voids.
4. **Partner signaling**: Better šmiranje (point-feeding) and partner protection.
5. **Endgame play**: Counts remaining taroks/suits for last tricks.
6. **Improved bidding**: Considers suit distribution (voids, singletons).
7. **Better klop play**: Ducks more aggressively, tracks who is dangerous.
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
    """Global ordering value (same as v1 for compatibility)."""
    if card.card_type == CardType.TAROK:
        return 10 + card.value
    assert card.suit is not None
    suit_base = {Suit.HEARTS: 33, Suit.DIAMONDS: 41, Suit.CLUBS: 49, Suit.SPADES: 57}
    return suit_base[card.suit] + card.value


_ALL_CARDS: list[Card] = list(DECK)
_ALL_TAROKS: list[Card] = [c for c in DECK if c.card_type == CardType.TAROK]


# ---------------------------------------------------------------------------
# Card-counting helper — derives info from GameState
# ---------------------------------------------------------------------------

class _CardTracker:
    """Derives card-counting info from game state (no hidden state)."""

    __slots__ = (
        "played", "remaining", "hand", "taroks_remaining",
        "suits_remaining", "taroks_in_hand", "suit_counts",
        "player_voids", "num_players",
    )

    def __init__(self, state: GameState, player_idx: int):
        self.num_players = state.num_players
        self.hand = state.hands[player_idx]

        # All cards that have been played (visible information)
        self.played: set[Card] = set()
        for trick in state.tricks:
            for _, card in trick.cards:
                self.played.add(card)
        if state.current_trick:
            for _, card in state.current_trick.cards:
                self.played.add(card)
        # Cards we put down are also out of play
        for card in state.put_down:
            self.played.add(card)

        # Cards still in play (not in our hand, not played)
        hand_set = set(self.hand)
        self.remaining: set[Card] = set(_ALL_CARDS) - self.played - hand_set

        # Tarok tracking
        self.taroks_in_hand = [c for c in self.hand if c.card_type == CardType.TAROK]
        self.taroks_remaining = [c for c in self.remaining if c.card_type == CardType.TAROK]

        # Suit tracking in hand
        self.suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
        for c in self.hand:
            if c.suit is not None:
                self.suit_counts[c.suit] += 1

        # Remaining suit cards
        self.suits_remaining: dict[Suit, list[Card]] = {s: [] for s in Suit}
        for c in self.remaining:
            if c.suit is not None:
                self.suits_remaining[c.suit].append(c)

        # Track which players have shown voids (played tarok on suit lead)
        self.player_voids: dict[int, set[Suit]] = {p: set() for p in range(self.num_players)}
        for trick in state.tricks:
            if not trick.cards:
                continue
            lead_card = trick.cards[0][1]
            if lead_card.suit is not None:
                for p, c in trick.cards[1:]:
                    if c.card_type == CardType.TAROK and p != player_idx:
                        self.player_voids[p].add(lead_card.suit)

    def higher_taroks_out(self, value: int) -> int:
        """Count taroks higher than `value` still in opponents' hands."""
        return sum(1 for c in self.taroks_remaining if c.value > value)

    def suit_is_master(self, suit: Suit, value: int) -> bool:
        """Is our card the highest remaining card in this suit?"""
        for c in self.suits_remaining[suit]:
            if c.value > value:
                return False
        return True

    def tricks_left(self, state: GameState) -> int:
        total_tricks = len(self.hand)  # approximate
        return max(1, total_tricks - len(state.tricks))


# ---------------------------------------------------------------------------
# Bidding — improved with suit distribution analysis
# ---------------------------------------------------------------------------

def _evaluate_hand_for_bid(hand: list[Card], num_players: int = 4) -> list[int]:
    """Improved bid evaluation considering suit distribution."""
    taroks = []
    kings = 0
    suits: dict[Suit, list[Card]] = {s: [] for s in Suit}
    voids = 0
    singletons = 0

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

    for s in Suit:
        if len(suits[s]) == 0:
            voids += 1
        elif len(suits[s]) == 1:
            singletons += 1

    # Compute hand rating — weighted score
    rating = 0.0

    # Taroks are the backbone of hand strength
    rating += tarok_count * 6
    rating += high_taroks * 4
    if has_skis:
        rating += 12
    if has_mond:
        rating += 10
    if has_pagat and tarok_count >= 6:
        rating += 5  # pagat is useful with protection

    # Kings contribute to trick-winning
    rating += kings * 8

    # Voids are very valuable (can trump in)
    rating += voids * 7
    rating += singletons * 2

    # Suit length penalty — long suits without king are weak
    for s in Suit:
        count = len(suits[s])
        has_king = any(c.is_king for c in suits[s])
        if count >= 3 and not has_king:
            rating -= count * 2

    # Normalize to 0-1 range (max realistic rating ~120)
    max_rating = 120.0
    ratio = min(1.0, rating / max_rating)

    is_3p = num_players == 3

    # Thresholds calibrated from play experience
    TRI = 0.25 + (0.10 if is_3p else 0.0)
    DVE = 0.32 + (0.12 if is_3p else 0.0)
    ENA = 0.40 + (0.13 if is_3p else 0.0)
    SOLO_TRI = 0.52
    SOLO_DVA = 0.60
    SOLO_ENA = 0.68
    SOLO_BREZ = 0.78

    thresholds = [
        (0, TRI),
        (1, DVE),
        (2, ENA),
        (3, SOLO_TRI),
        (4, SOLO_DVA),
        (5, SOLO_ENA),
        (6, SOLO_BREZ),
    ]

    # Berač: very weak hand, all suits covered, few taroks
    has_all_suits = all(len(suits[s]) > 0 for s in Suit)
    can_berac = (
        ratio < 0.18
        and tarok_count <= 2
        and has_all_suits
        and high_taroks == 0
    )

    modes: list[int] = []
    for mode_id, threshold in thresholds:
        if ratio >= threshold:
            modes.append(mode_id)

    if can_berac:
        modes.append(7)

    if not modes:
        return [-1]

    return modes


# ---------------------------------------------------------------------------
# Map mode IDs → Contract enum
# ---------------------------------------------------------------------------

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
# Card-play heuristic — major upgrade
# ---------------------------------------------------------------------------

def _evaluate_card_play(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    is_leading: bool,
    tracker: _CardTracker,
) -> float:
    """Improved card play evaluation with card counting and positional play."""
    wo = _worth_over(card)
    pts = card.points

    is_declarer = state.declarer == player_idx
    is_partner = state.partner is not None and state.partner == player_idx
    is_playing = is_declarer or is_partner
    is_klop = state.contract is not None and state.contract.is_klop
    is_berac = state.contract is not None and state.contract.is_berac
    tricks_left = tracker.tricks_left(state)

    # === KLOP / BERAC ===
    if is_klop or is_berac:
        return _eval_klop_berac(card, hand, state, player_idx, is_leading, tracker)

    # === NORMAL GAMES ===
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
    """Klop/Berač: avoid taking tricks, dump points when safe."""
    wo = _worth_over(card)
    pts = card.points
    score = 0.0

    if is_leading:
        if card.card_type == CardType.TAROK:
            # Leading tarok in klop: very bad unless it's low
            score = -wo * 2
            # Never lead pagat unless it's the only tarok and low remaining
            if card.value == PAGAT and len(tracker.taroks_in_hand) > 1:
                score -= 500
        else:
            assert card.suit is not None
            count = tracker.suit_counts.get(card.suit, 1)
            remaining_in_suit = len(tracker.suits_remaining.get(card.suit, []))

            # Lead from suits where opponents have many cards (less likely to
            # be void and trump). Check if any opponent is void.
            opponents_void = sum(
                1 for p in range(state.num_players)
                if p != player_idx and card.suit in tracker.player_voids[p]
            )

            if opponents_void > 0:
                # Someone can trump this suit — dangerous to lead
                score = -wo * 3 - opponents_void * 20
            else:
                # Safe-ish: lead low cards from suits with many remaining
                score = -wo + remaining_in_suit * 2 - count * 3
    else:
        trick = state.current_trick
        if trick and trick.cards:
            best_card = trick.cards[0][1]
            for _, c in trick.cards:
                if c.beats(best_card, trick.lead_suit):
                    best_card = c

            if card.beats(best_card, trick.lead_suit):
                # We'd win the trick — penalize heavily
                trick_pts = sum(c.points for _, c in trick.cards) + pts
                score = -wo * 3 - trick_pts * 10

                # But if we're last to play and trick has 0 points, acceptable
                cards_in_trick = len(trick.cards)
                if cards_in_trick == state.num_players - 1 and trick_pts <= 4:
                    score = -wo  # mild penalty — we must take it
            else:
                # We won't win — dump our most expensive cards
                score = pts * 5 + wo * 0.2
    return score


def _eval_leading(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    is_playing: bool,
    tracker: _CardTracker,
) -> float:
    """Evaluate a card for leading a trick."""
    wo = _worth_over(card)
    pts = card.points
    score = 0.0
    tricks_left = tracker.tricks_left(state)

    if card.card_type == CardType.TAROK:
        higher_out = tracker.higher_taroks_out(card.value)

        if card.value == SKIS:
            # Škis always wins — lead it to extract value
            # Best when opponents still have high taroks
            if len(tracker.taroks_remaining) > 2:
                score = 80
            else:
                score = 40

        elif card.value == MOND:
            # Lead mond only when škis is gone or we have škis
            skis_out = any(c.value == SKIS for c in tracker.taroks_remaining)
            if skis_out:
                score = -100  # don't risk losing mond to škis
            else:
                score = 60  # mond is safe, lead it

        elif card.value == PAGAT:
            # Pagat ultimo: save for last trick
            if tricks_left <= 2 and len(tracker.taroks_in_hand) <= 2:
                score = 200  # ultimo time!
            else:
                score = -200  # protect pagat

        else:
            # Regular tarok: lead high taroks to flush opponents
            if is_playing:
                # Declarer: lead taroks to drain opponents' taroks
                if higher_out == 0:
                    # We have the master tarok — very strong lead
                    score = 50 + wo
                else:
                    score = math.pow(max(0, wo - 11) / 3, 1.5) + 5
            else:
                # Defender: lead taroks to force out declarer's taroks
                if higher_out <= 1:
                    score = 30 + wo * 0.5
                else:
                    score = wo * 0.3

    else:
        # Suit card leading
        assert card.suit is not None
        count = tracker.suit_counts[card.suit]
        remaining = len(tracker.suits_remaining.get(card.suit, []))

        if is_playing:
            # Declarer team: lead kings to collect points, or short suits
            if card.is_king:
                # Lead king if it's master (highest remaining)
                if tracker.suit_is_master(card.suit, card.value):
                    score = 25 + pts * 2
                else:
                    # King might get trumped — risky
                    score = pts - 5
            elif count == 1:
                # Singleton: lead it to void the suit (then trump future leads)
                score = 15 - pts  # prefer leading low singletons
            else:
                # Multi-card suit: lead low to probe
                score = -pts * 1.5 - count * 2
        else:
            # Defenders: lead suits to force declarer to trump
            # Especially suits where we know declarer might be short

            # Lead from long suits to exhaust declarer
            if count >= 3:
                # Lead low from long suit — partner might have the king
                if card.is_king:
                    score = pts * 2  # partner can šmir
                else:
                    score = count * 3 - pts * 1.5
            elif count == 1:
                # Singleton lead: could be good if partner has suit strength
                score = 10 - pts
            else:
                score = -count * 2 - pts

    return score


def _eval_following(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    is_playing: bool,
    tracker: _CardTracker,
) -> float:
    """Evaluate a card when following in a trick."""
    wo = _worth_over(card)
    pts = card.points
    score = 0.0
    tricks_left = tracker.tricks_left(state)

    trick = state.current_trick
    if trick is None or not trick.cards:
        return 0.0

    lead_card = trick.cards[0][1]
    num_cards_played = len(trick.cards)
    is_last = num_cards_played == state.num_players - 1  # 4th seat

    # Find current best card
    best_card = lead_card
    best_player = trick.cards[0][0]
    for p, c in trick.cards:
        if c.beats(best_card, trick.lead_suit):
            best_card = c
            best_player = p

    # Is the current winner an ally?
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

    # --- Pagat protection ---
    if card.card_type == CardType.TAROK and card.value == PAGAT:
        if tricks_left <= 2 and len(tracker.taroks_in_hand) <= 2:
            # Pagat ultimo!
            if would_win:
                return 500
            else:
                return -500  # don't throw pagat into a losing trick late
        else:
            return -200  # save it

    # --- Mond protection ---
    if card.card_type == CardType.TAROK and card.value == MOND:
        if not would_win:
            return -500  # never dump mond into a loss

    # --- Škis ---
    if card.card_type == CardType.TAROK and card.value == SKIS:
        if tricks_left <= 1:
            return -300  # don't play škis in last trick (captured)
        else:
            return trick_pts * 2 + 30  # škis wins, collect points

    if would_win:
        if best_is_ally:
            # Ally already winning — don't over-trump
            if is_last:
                # Last seat: ally wins, šmir (dump high-value cards)
                score = pts * 4
            else:
                # Not last: over-trumping ally is wasteful
                score = -wo * 0.5 - math.pow(wo / 3, 1.5)
        else:
            # Opponent winning — take the trick
            if is_last:
                # Last seat — we know exact trick value. Win cheaply.
                # Play the lowest winning card
                score = trick_pts * 3 - wo * 0.8
            else:
                # Middle seat — might get over-trumped
                score = trick_pts * 1.5 + wo * 0.2
    else:
        # Can't win
        if best_is_ally:
            # Ally winning — šmir! (feed them points)
            if is_last:
                score = pts * 5  # maximize points given to ally
            else:
                # Not last, but ally leads — still feed points if safe
                # Risk: opponent after us might take it
                score = pts * 2.5
        else:
            # Opponent winning — dump lowest value cards
            score = -(pts * 3) - wo * 0.3

    return score


# ---------------------------------------------------------------------------
# Talon selection — void-building aware
# ---------------------------------------------------------------------------

def _evaluate_talon_group(
    group: list[Card],
    hand: list[Card],
    called_king: Card | None,
) -> float:
    """Score a talon group. Higher = better. Considers void-building."""
    # Current suit distribution
    suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
    for c in hand:
        if c.suit is not None:
            suit_counts[c.suit] += 1

    total = 0.0
    for card in group:
        total += card.points * 2 + _worth_over(card) * 0.3

        if card.card_type == CardType.TAROK:
            total += 8  # taroks are always useful

        if called_king and card.suit == called_king.suit:
            total += 4  # strengthens partnership suit

    # Bonus: does picking this group help us void a suit?
    # Simulate the hand after picking this group + discarding
    combined = hand + group
    new_suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
    for c in combined:
        if c.suit is not None:
            new_suit_counts[c.suit] += 1

    # Prefer groups that keep our short suits short (or create new voids)
    for s in Suit:
        if s == (called_king.suit if called_king else None):
            continue  # don't void called suit
        if suit_counts[s] <= 1 and new_suit_counts[s] <= 1:
            total += 5  # maintains potential void
        if suit_counts[s] == 0 and new_suit_counts[s] == 0:
            total += 3  # preserves existing void

    return total


# ---------------------------------------------------------------------------
# Discard heuristic — void-building focused
# ---------------------------------------------------------------------------

def _choose_discards(
    hand: list[Card],
    must_discard: int,
    called_king: Card | None,
    contract: Contract | None,
) -> list[Card]:
    """Improved discard: aggressively build voids."""
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

    # Group discardable cards by suit
    by_suit: dict[Suit, list[Card]] = {s: [] for s in Suit}
    for c in discardable:
        if c.suit is not None:
            by_suit[c.suit].append(c)

    # Count all suit cards in hand (to evaluate void potential)
    hand_suit_counts: dict[Suit, int] = {s: 0 for s in Suit}
    for c in hand:
        if c.suit is not None:
            hand_suit_counts[c.suit] += 1

    result: list[Card] = []

    # Strategy: completely void shortest non-called suits first
    void_candidates = sorted(
        [(s, cards) for s, cards in by_suit.items()
         if s != called_suit and cards],
        key=lambda x: hand_suit_counts[x[0]],  # shortest first
    )

    for suit, cards in void_candidates:
        if len(result) >= must_discard:
            break
        # Can we void this suit entirely?
        suit_in_hand = [c for c in hand if c.suit == suit and not c.is_king]
        # Only void if we have enough discard slots
        discardable_from_suit = [c for c in suit_in_hand if c in discardable]
        if len(discardable_from_suit) + len(result) <= must_discard:
            # Void the whole suit
            for c in discardable_from_suit:
                if c not in result:
                    result.append(c)

    # Fill remaining slots with lowest-value cards
    if len(result) < must_discard:
        remaining = [c for c in discardable if c not in result]
        remaining.sort(key=lambda c: (c.points, _worth_over(c)))
        for c in remaining:
            if len(result) >= must_discard:
                break
            result.append(c)

    return result[:must_discard]


# ---------------------------------------------------------------------------
# Announcement heuristic — more aggressive
# ---------------------------------------------------------------------------

def _should_announce(
    state: GameState,
    player_idx: int,
    hand: list[Card],
    tracker: _CardTracker,
) -> int:
    """Announcement decisions with card-counting awareness."""
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
        # Defenders: kontra with strong hands
        existing = state.kontra_levels.get("game")
        if existing is None:
            # Strong defense: many taroks + good suit coverage
            if tarok_count >= 7 and has_skis:
                return 5  # kontra
        return 0

    # Trula: need all three trula cards
    if has_pagat and has_mond and has_skis:
        already = any(
            Announcement.TRULA in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 1

    # Kings: need all four kings
    if len(kings_in_hand) == 4:
        already = any(
            Announcement.KINGS in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 2

    # Pagat ultimo: need pagat + strong tarok protection
    if has_pagat and tarok_count >= 7:
        already = any(
            Announcement.PAGAT_ULTIMO in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 3

    return 0


# ---------------------------------------------------------------------------
# StockŠkis v2 Player
# ---------------------------------------------------------------------------

class StockSkisPlayerV2:
    """Enhanced heuristic Tarok bot with card counting and positional play.

    Parameters
    ----------
    name : str
        Display name.
    strength : float
        0.0 = random, 1.0 = pure heuristic. Intermediate adds noise.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "StockŠkis-v2",
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

    # ------------------------------------------------------------------
    # PlayerPort implementation
    # ------------------------------------------------------------------

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

        # Call king in the suit where we have the most supporting cards
        # (more cards = more chances to šmir and protect the king)
        best_king = callable_kings[0]
        best_score = -1

        for king in callable_kings:
            assert king.suit is not None
            count = sum(1 for c in hand if c.suit == king.suit)
            # Slight bonus if we have low cards in that suit (can follow without risk)
            low_cards = sum(1 for c in hand if c.suit == king.suit and c.value <= 4)
            score = count * 3 + low_cards
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
