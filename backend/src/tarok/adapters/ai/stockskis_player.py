"""StockŠkis-style heuristic player — implements PlayerPort.

A pure-Python port of the StockŠkis engine heuristics from mytja/Tarok.
This gives our RL agents a competent, deterministic opponent to train
against without any IPC/subprocess overhead.

Original engine: https://github.com/mytja/Tarok (GPLv3)
Heuristics ported from stockskis/lib/src/stockskis_base.dart
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
)
from tarok.entities.game_state import (
    Announcement,
    Contract,
    GameState,
    Phase,
    Trick,
)


# ---------------------------------------------------------------------------
# Card helpers (mirroring StockŠkis asset-path conventions)
# ---------------------------------------------------------------------------

# StockŠkis uses "worthOver" (global sort order from 11..64) for evaluation.
# We assign a comparable sort key so that heuristic penalties translate cleanly.

def _worth_over(card: Card) -> int:
    """Global ordering value comparable to StockŠkis' worthOver field.

    Taroks: 11 (Pagat) .. 32 (Škis)
    Suits:  Hearts 1..8=33..40, Diamonds=41..48, Clubs=49..56, Spades=57..64
    """
    if card.card_type == CardType.TAROK:
        return 10 + card.value  # I→11, Škis(22)→32
    assert card.suit is not None
    suit_base = {Suit.HEARTS: 33, Suit.DIAMONDS: 41, Suit.CLUBS: 49, Suit.SPADES: 57}
    return suit_base[card.suit] + card.value  # SuitRank 1..8


def _card_suit_key(card: Card) -> str:
    """Return the StockŠkis-style card type key."""
    if card.card_type == CardType.TAROK:
        return "taroki"
    assert card.suit is not None
    return {
        Suit.HEARTS: "src",
        Suit.DIAMONDS: "kara",
        Suit.CLUBS: "kriz",
        Suit.SPADES: "pik",
    }[card.suit]


# ---------------------------------------------------------------------------
# Hand evaluation for bidding
# ---------------------------------------------------------------------------

def _evaluate_hand_for_bid(hand: list[Card], num_players: int = 4) -> list[int]:
    """Return list of game-mode IDs the bot would bid, from StockŠkis' suggest().

    Game mode IDs (matching StockŠkis convention):
      0=tri, 1=dva, 2=ena, 3=solo_tri, 4=solo_dva, 5=solo_ena,
      6=solo_brez, 7=berac, 8=odprti_berac, 9=barvni_valat, 10=valat
      -1=pass
    """
    taroks = 0
    kings = 0
    my_rating = 0
    max_rating = 0

    for card in hand:
        wo = _worth_over(card)
        my_rating += wo + card.points ** 2
        if card.card_type == CardType.TAROK:
            taroks += 1
        if card.is_king:
            kings += 1

    # Maximum possible rating: best 12 cards out of 54
    # (simplified — StockŠkis iterates highest cards)
    sorted_wo = sorted((_worth_over(c) + c.points ** 2 for c in _all_cards()), reverse=True)
    max_rating = sum(sorted_wo[:len(hand)])

    if max_rating == 0:
        return [-1]

    ratio = my_rating / max_rating
    is_3p = num_players == 3

    # Thresholds from StockŠkis (NORMALNI path)
    TRI = 0.29 + (0.13 if is_3p else 0.0) - taroks * 0.003
    DVE = 0.33 + (0.14 if is_3p else 0.0) - taroks * 0.0035
    ENA = 0.42 + (0.15 if is_3p else 0.0) - taroks * 0.004
    SOLO_TRI = 0.55 - taroks * 0.007
    SOLO_DVA = 0.62 - taroks * 0.007
    SOLO_ENA = 0.68 - taroks * 0.007
    SOLO_BREZ = 0.80

    thresholds = [
        (0, TRI),
        (1, DVE),
        (2, ENA),
        (3, SOLO_TRI),
        (4, SOLO_DVA),
        (5, SOLO_ENA),
        (6, SOLO_BREZ),
    ]

    # Berač: check if hand is weak enough
    suit_counts = {s: 0 for s in Suit}
    for c in hand:
        if c.suit is not None:
            suit_counts[c.suit] += 1
    has_all_suits = all(v > 0 for v in suit_counts.values())
    low_taroks = sum(1 for c in hand if c.card_type == CardType.TAROK and c.value <= 10)

    berac_threshold = 0.20 + (0.05 if is_3p else 0.0)
    can_berac = ratio < berac_threshold and taroks <= 3 and has_all_suits

    modes: list[int] = []

    # Find the highest mode the bot qualifies for
    for mode_id, threshold in thresholds:
        if ratio >= threshold:
            modes.append(mode_id)

    if can_berac:
        modes.append(7)  # berac

    if not modes:
        return [-1]  # pass

    return modes


_ALL_CARDS_CACHE: list[Card] | None = None


def _all_cards() -> list[Card]:
    """Lazily build and cache a full 54-card deck for rating computations."""
    global _ALL_CARDS_CACHE
    if _ALL_CARDS_CACHE is None:
        from tarok.entities.card import DECK
        _ALL_CARDS_CACHE = list(DECK)
    return _ALL_CARDS_CACHE


# ---------------------------------------------------------------------------
# Map StockŠkis mode IDs → our Contract enum
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
    # 8: odprti_berac — not in our rules
    # 9: barvni_valat — not in our rules
    # 10: valat — special announcement
}

_CONTRACT_STRENGTH = {c: c.strength for c in Contract}


# ---------------------------------------------------------------------------
# Card-play heuristic (simplified StockŠkis suggestCard / evaluateMoves)
# ---------------------------------------------------------------------------

def _evaluate_card_play(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    is_leading: bool,
) -> float:
    """Heuristic evaluation for playing *card*. Higher = better."""
    wo = _worth_over(card)
    pts = card.points

    # Determine role information
    is_declarer = state.declarer == player_idx
    is_partner = state.partner is not None and state.partner == player_idx
    is_playing = is_declarer or is_partner  # On the declarer team
    is_klop = state.contract is not None and state.contract.is_klop
    is_berac = state.contract is not None and state.contract.is_berac

    # Count suits/taroks in hand
    taroks_in_hand = sum(1 for c in hand if c.card_type == CardType.TAROK)
    suit_counts: dict[str, int] = {}
    for c in hand:
        key = _card_suit_key(c)
        suit_counts[key] = suit_counts.get(key, 0) + 1

    card_type = _card_suit_key(card)
    score = 0.0

    # === KLOP / BERAC: avoid taking tricks / avoid high cards ===
    if is_klop or is_berac:
        if is_leading:
            # Lead with lowest card — prefer suit cards first, avoid taroks
            if card.card_type == CardType.TAROK:
                score = -wo
                # Protect pagat
                if card.value == PAGAT and taroks_in_hand > 1:
                    score -= 1000
            else:
                # Prefer suits where we have few cards
                count = suit_counts.get(card_type, 1)
                score = -wo - count * 5
        else:
            # Following: play lowest possible to avoid taking the trick
            trick = state.current_trick
            if trick and trick.cards:
                lead_card = trick.cards[0][1]
                # Currently winning card
                best_card = lead_card
                for _, c in trick.cards:
                    if c.beats(best_card, trick.lead_suit):
                        best_card = c
                best_wo = _worth_over(best_card)

                if card.beats(best_card, trick.lead_suit):
                    # This card would take the trick — big penalty
                    score = -wo * 3 - pts * 10
                else:
                    # This card won't take the trick — dump high-value cards
                    score = pts + wo * 0.1
            else:
                score = -wo
        return score

    # === NORMAL GAMES (three, two, one, solo, etc.) ===
    if is_leading:
        # Leading: play strategically
        if card.card_type == CardType.TAROK:
            # StockŠkis: evaluation = pow((worthOver - 11) / 3, 1.5)
            adjusted = max(0, wo - 11)
            score = math.pow(adjusted / 3, 1.5)
            # Avoid leading pagat early
            if card.value == PAGAT and taroks_in_hand > 1:
                score -= 50
            # Avoid leading mond early unless playing
            if card.value == MOND and not is_playing:
                score -= 20
        else:
            # Suit cards: prefer leading from short suits (void-building)
            count = suit_counts.get(card_type, 1)
            if is_playing:
                # Declarer: lead kings to set up, or lead from short suits
                score = pts * 2 - count * 3
                if card.is_king:
                    score += 10
            else:
                # Defender: lead from long suits to exhaust declarer's suit cards
                suit_worth = sum(
                    c.points for c in hand if _card_suit_key(c) == card_type
                )
                score = suit_worth - math.pow(count, 1.5) - math.pow(pts / 2, 2)
    else:
        # Following a trick
        trick = state.current_trick
        if trick is None or not trick.cards:
            return 0.0

        lead_card = trick.cards[0][1]
        lead_type = _card_suit_key(lead_card)

        # Find current best card in trick
        best_card = lead_card
        best_player = trick.cards[0][0]
        for p, c in trick.cards:
            if c.beats(best_card, trick.lead_suit):
                best_card = c
                best_player = p

        # Is the current trick winner on our team?
        best_is_ally = False
        if is_playing:
            # Our ally if they're also on declarer team
            best_is_ally = (
                best_player == state.declarer
                or (state.partner is not None and best_player == state.partner)
            )
        else:
            # Our ally if they're also a defender
            best_is_ally = (
                best_player != state.declarer
                and (state.partner is None or best_player != state.partner)
            )

        # Calculate trick value
        trick_pts = sum(c.points for _, c in trick.cards) + pts
        best_wo = _worth_over(best_card)

        would_win = card.beats(best_card, trick.lead_suit)

        penalty = 0.0

        if would_win:
            if is_playing:
                # Declarer team winning — good, weighted by trick value
                score = trick_pts * 2 + wo * 0.5
            else:
                # Defender winning — also good if taking from declarer
                if not best_is_ally:
                    score = trick_pts * 2 + wo * 0.3
                else:
                    # Allied player already winning — don't over-trump
                    score = -wo * 0.5
                    penalty += math.pow(wo / 3, 1.5)
        else:
            # Can't win this trick
            if best_is_ally:
                # Ally is winning — šmir (dump high-value cards for them)
                score = pts * 3 + wo * 0.2
            else:
                # Opponent is winning — dump low-value cards
                score = -(pts * 2) - wo * 0.3

        # Pagat ultimo awareness: protect pagat for last trick
        if card.card_type == CardType.TAROK and card.value == PAGAT:
            tricks_played = len(state.tricks)
            total_tricks = 12  # 4 players, 48 cards / 4
            if tricks_played < total_tricks - 1:
                penalty += 200  # Don't play pagat until ultimo

        # Mond protection: avoid losing mond
        if card.card_type == CardType.TAROK and card.value == MOND:
            if not would_win:
                penalty += 500  # Never dump mond into a losing trick

        # Škis handling: škis always wins but is captured in the last trick
        if card.card_type == CardType.TAROK and card.value == SKIS:
            tricks_played = len(state.tricks)
            total_tricks = 12
            if tricks_played >= total_tricks - 1:
                penalty += 300  # Don't play škis in last trick
            else:
                score += trick_pts  # Škis is good value to play early

        score -= penalty

    return score


# ---------------------------------------------------------------------------
# Talon selection heuristic
# ---------------------------------------------------------------------------

def _evaluate_talon_group(
    group: list[Card],
    hand: list[Card],
    called_king: Card | None,
) -> float:
    """Score a talon group. Higher = better pick for declarer."""
    total = 0.0
    for card in group:
        total += card.points * 2 + _worth_over(card) * 0.3
        # Bonus for taroks (strengthen hand)
        if card.card_type == CardType.TAROK:
            total += 5
        # Bonus if it's the called king's suit (strengthens partnership suit)
        if called_king and card.suit == called_king.suit:
            total += 3
    return total


# ---------------------------------------------------------------------------
# Discard heuristic
# ---------------------------------------------------------------------------

def _choose_discards(
    hand: list[Card],
    must_discard: int,
    called_king: Card | None,
    contract: Contract | None,
) -> list[Card]:
    """Choose cards to put down. Can't discard taroks or kings (except in talon exchange edge cases)."""
    discardable = [
        c for c in hand
        if c.card_type != CardType.TAROK and not c.is_king
    ]

    # If we can't discard enough non-tarok non-king cards, allow lowest taroks
    if len(discardable) < must_discard:
        # Add lowest taroks (excluding trula: pagat, mond, škis)
        extra_taroks = sorted(
            [c for c in hand if c.card_type == CardType.TAROK
             and c.value not in (PAGAT, MOND, SKIS)
             and c not in discardable],
            key=lambda c: c.value,
        )
        discardable.extend(extra_taroks)

    # Strategy: discard cards from suits where we have the fewest (void-building)
    # But protect called king's suit
    called_suit = called_king.suit if called_king else None

    def discard_priority(card: Card) -> tuple[int, int]:
        # Lower = more likely to discard
        suit_count = sum(1 for c in hand if c.suit == card.suit) if card.suit else 99
        suit_penalty = 0 if card.suit != called_suit else 50
        return (suit_penalty + suit_count, card.points)

    discardable.sort(key=discard_priority)
    return discardable[:must_discard]


# ---------------------------------------------------------------------------
# Announcement heuristic
# ---------------------------------------------------------------------------

def _should_announce(
    state: GameState,
    player_idx: int,
    hand: list[Card],
) -> int:
    """Return an announcement action index (0=pass). Simplified from StockŠkis predict()."""
    is_declarer = state.declarer == player_idx
    is_partner = state.partner is not None and state.partner == player_idx
    is_playing = is_declarer or is_partner

    taroks = [c for c in hand if c.card_type == CardType.TAROK]
    tarok_count = len(taroks)
    has_pagat = any(c.value == PAGAT for c in taroks)
    has_mond = any(c.value == MOND for c in taroks)
    has_skis = any(c.value == SKIS for c in taroks)

    # Only declarer team can announce (in standard rules)
    if not is_playing:
        # Defenders can kontra. For simplicity, strong defenders give kontra.
        # action 5 = kontra_game
        existing = state.kontra_levels.get("game")
        if existing is None or (hasattr(existing, 'is_opponent_turn') and existing.is_opponent_turn):
            if tarok_count >= 8:
                return 5  # kontra the game
        return 0

    # Trula announcement: if we have all 3 trula cards
    if has_pagat and has_mond and has_skis:
        already = any(
            Announcement.TRULA in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 1  # announce trula

    # Pagat ultimo: if we have pagat + many taroks (good protection)
    if has_pagat and tarok_count >= 8:
        already = any(
            Announcement.PAGAT_ULTIMO in anns
            for anns in state.announcements.values()
        )
        if not already:
            return 3  # announce pagat ultimo

    return 0  # pass


# ---------------------------------------------------------------------------
# StockŠkis Player — implements PlayerPort
# ---------------------------------------------------------------------------

class StockSkisPlayer:
    """Heuristic Tarok bot ported from the StockŠkis engine.

    This player implements the same ``PlayerPort`` protocol as ``RLAgent``
    so it can be dropped into ``GameLoop`` as an opponent.  It stores
    no experiences and has no neural network — all decisions are based on
    hand-crafted heuristics from the original Dart engine.

    Parameters
    ----------
    name : str
        Display name for this player.
    strength : float
        Controls randomness (0.0 = fully random, 1.0 = pure heuristic).
        Intermediate values add noise to evaluations for variety.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "StockŠkis",
        strength: float = 1.0,
        seed: int | None = None,
    ):
        self._name = name
        self.strength = max(0.0, min(1.0, strength))
        self._rng = random.Random(seed)
        # Dummy attributes so the trainer can check isinstance / hasattr
        self.experiences: list = []

    @property
    def name(self) -> str:
        return self._name

    def set_training(self, training: bool) -> None:
        """No-op — heuristic player doesn't train."""
        pass

    def clear_experiences(self) -> None:
        """No-op — nothing to clear."""
        pass

    def finalize_game(self, reward: float) -> None:
        """No-op — no experiences to reward."""
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
            return None  # pass

        # Map mode IDs to contracts and pick the best legal one
        desired = [_MODE_TO_CONTRACT[m] for m in modes if m in _MODE_TO_CONTRACT]
        # Sort by strength descending — bid as high as we qualify for
        desired.sort(key=lambda c: c.strength, reverse=True)

        for contract in desired:
            if contract in legal_bids:
                return contract

        # If none of our desired bids are legal (outbid already), pass
        return None

    async def choose_king(
        self,
        state: GameState,
        player_idx: int,
        callable_kings: list[Card],
    ) -> Card:
        hand = state.hands[player_idx]
        # Call king of the suit where we have the most cards
        best_king = callable_kings[0]
        best_count = -1
        for king in callable_kings:
            count = sum(1 for c in hand if c.suit == king.suit)
            if count > best_count:
                best_count = count
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
        # Unused — the game loop calls choose_announce_action instead
        return []

    async def choose_announce_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> int:
        hand = state.hands[player_idx]
        return _should_announce(state, player_idx, hand)

    async def choose_card(
        self,
        state: GameState,
        player_idx: int,
        legal_plays: list[Card],
    ) -> Card:
        if len(legal_plays) == 1:
            return legal_plays[0]

        hand = state.hands[player_idx]
        is_leading = (
            state.current_trick is None or len(state.current_trick.cards) == 0
        )

        # Evaluate each legal card
        evals: list[tuple[Card, float]] = []
        for card in legal_plays:
            score = _evaluate_card_play(card, hand, state, player_idx, is_leading)
            # Add noise based on (1 - strength) for variety
            if self.strength < 1.0:
                noise = self._rng.gauss(0, 1) * (1 - self.strength) * 20
                score += noise
            evals.append((card, score))

        # Pick the highest-scoring card
        evals.sort(key=lambda x: x[1], reverse=True)
        return evals[0][0]
