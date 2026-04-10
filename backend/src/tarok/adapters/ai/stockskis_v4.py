"""StockSkis v4 — v3 plus tighter berač gating and clearer lead roles.

Builds on v3 with these additional heuristics:

1. Do not bid berač with more than two taroks.
2. Do not bid berač with any singleton suit.
3. If you are the declarer's partner, prefer opening with your highest tarok.
4. If you are in the opposition, preserve taroks and lead low-value suit cards
   that are likely to pull taroks from the declarer team.
"""

from __future__ import annotations

import math

from tarok.adapters.ai.stockskis_v3 import (
    StockSkisPlayerV3,
    _CardTracker,
    _choose_discards,
    _eval_following,
    _eval_klop_berac,
    _evaluate_talon_group,
    _should_announce,
    _worth_over,
)
from tarok.entities.card import Card, CardType, Suit, SuitRank, PAGAT, MOND, SKIS
from tarok.entities.game_state import Announcement, Contract, GameState


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

    rating = 0.0
    rating += tarok_count * 6
    rating += high_taroks * 5
    if has_skis:
        rating += 14
    if has_mond:
        rating += 11
        if has_skis:
            rating += 3
    if has_pagat:
        if tarok_count >= 7:
            rating += 8
        elif tarok_count >= 5:
            rating += 3
        else:
            rating -= 2

    for suit in Suit:
        count = len(suits[suit])
        has_king = any(card.is_king for card in suits[suit])
        has_queen = any(card.value == SuitRank.QUEEN.value for card in suits[suit])

        if count == 0:
            rating += 8
        elif count == 1:
            rating += 4 if has_king else 3
        elif count == 2 and has_king:
            rating += 2
        elif count >= 3 and not has_king:
            rating -= count * 2

        if has_king and has_queen and count >= 2:
            rating += 2

    if tarok_count >= 8:
        rating += (tarok_count - 7) * 3

    ratio = min(1.0, rating / 130.0)
    is_3p = num_players == 3

    thresholds = [
        (0, 0.24 + (0.10 if is_3p else 0.0)),
        (1, 0.30 + (0.12 if is_3p else 0.0)),
        (2, 0.38 + (0.13 if is_3p else 0.0)),
        (3, 0.50),
        (4, 0.58),
        (5, 0.66),
        (6, 0.76),
    ]

    has_all_suits = all(len(suits[suit]) > 0 for suit in Suit)
    has_singleton = any(len(suits[suit]) == 1 for suit in Suit)
    can_berac = (
        ratio < 0.16
        and tarok_count <= 2
        and has_all_suits
        and not has_singleton
        and high_taroks == 0
        and kings == 0
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


def _eval_leading(
    card: Card,
    hand: list[Card],
    state: GameState,
    player_idx: int,
    tracker: _CardTracker,
) -> float:
    wo = _worth_over(card)
    pts = card.points
    phase = tracker.game_phase(state)
    tricks_left = tracker.tricks_left(state)

    is_declarer = state.declarer == player_idx
    is_partner = state.partner is not None and state.partner == player_idx
    is_opposition = not is_declarer and not is_partner

    if card.card_type == CardType.TAROK:
        higher_out = tracker.higher_taroks_out(card.value)

        if card.value == PAGAT:
            if tricks_left <= 2 and len(tracker.taroks_in_hand) <= 2:
                return 300
            if tricks_left == 1:
                return 400
            return -250

        if is_partner:
            if card.value == SKIS:
                if tricks_left <= 1:
                    return -300
                return 125 if phase == "early" else 105 if phase == "mid" else 70

            if card.value == MOND:
                skis_out = any(c.value == SKIS for c in tracker.taroks_remaining)
                has_skis = any(c.value == SKIS for c in tracker.taroks_in_hand)
                if skis_out and not has_skis:
                    return 72
                return 95 + len(tracker.taroks_remaining) * 2

            return 90 + wo * 1.4 - higher_out * 4

        if is_opposition:
            if higher_out == 0 and phase == "late":
                return 20 + wo * 0.2
            if higher_out <= 1 and tricks_left <= 3:
                return 8 + wo * 0.1
            return -45 - wo * 0.4 - higher_out * 6

        if card.value == SKIS:
            if phase == "early":
                return 90
            if phase == "mid":
                return 70
            if tricks_left <= 1:
                return -300
            return 50

        if card.value == MOND:
            skis_out = any(c.value == SKIS for c in tracker.taroks_remaining)
            has_skis = any(c.value == SKIS for c in tracker.taroks_in_hand)
            if skis_out and not has_skis:
                return -150
            if has_skis:
                return 65
            return 70 + len(tracker.taroks_remaining) * 3

        if higher_out == 0:
            return 65 + wo

        if phase == "early":
            return math.pow(max(0, wo - 11) / 3, 1.5) + 10
        return math.pow(max(0, wo - 11) / 3, 1.5)

    assert card.suit is not None
    count = tracker.suit_counts[card.suit]
    remaining = tracker.count_remaining_in_suit(card.suit)

    if is_partner:
        if card.is_king and tracker.suit_is_master(card.suit, card.value):
            return 18 + pts
        if count == 1:
            return 6 - pts
        if count == 2 and not card.is_king:
            return 4 - pts
        return -pts - count

    if is_declarer:
        if card.is_king:
            if tracker.suit_is_master(card.suit, card.value):
                return 30 + pts * 2
            if count >= 3:
                return 15 + pts
            return pts - 8
        if count == 1:
            return 18 - pts
        if count == 2:
            return 12 - pts if not card.is_king else pts
        return -pts * 1.5 - count * 2

    declarer_team_voids = 0
    if state.declarer is not None and card.suit in tracker.player_voids.get(state.declarer, set()):
        declarer_team_voids += 1
    if state.partner is not None and card.suit in tracker.player_voids.get(state.partner, set()):
        declarer_team_voids += 1

    score = 18 - pts * 4 - count * 1.5
    if card.is_king:
        score -= 25
    if card.value <= SuitRank.PIP_2.value:
        score += 8
    if declarer_team_voids > 0:
        score += declarer_team_voids * 14 - pts * 2
    elif remaining > 0:
        score += min(remaining, 4)

    return score


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
        return _eval_leading(card, hand, state, player_idx, tracker)
    return _eval_following(card, hand, state, player_idx, is_playing, tracker)


class StockSkisPlayerV4(StockSkisPlayerV3):
    """Strong heuristic bot with tighter berač rules and refined opening leads."""

    def __init__(
        self,
        name: str = "StockŠkis-v4",
        strength: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(name=name, strength=strength, seed=seed)

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

        desired = [_MODE_TO_CONTRACT[mode] for mode in modes if mode in _MODE_TO_CONTRACT]
        desired.sort(key=lambda contract: contract.strength, reverse=True)

        for contract in desired:
            if contract in legal_bids:
                return contract
        return None

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
        is_leading = state.current_trick is None or len(state.current_trick.cards) == 0

        evals: list[tuple[Card, float]] = []
        for card in legal_plays:
            score = _evaluate_card_play(card, hand, state, player_idx, is_leading, tracker)
            if self.strength < 1.0:
                noise = self._rng.gauss(0, 1) * (1 - self.strength) * 20
                score += noise
            evals.append((card, score))

        evals.sort(key=lambda item: item[1], reverse=True)
        return evals[0][0]