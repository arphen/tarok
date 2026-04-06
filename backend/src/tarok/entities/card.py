"""Card domain entity for Slovenian Tarok.

54 cards total:
  - 22 Taroks: I (Pagat) through XXI (Mond) + Škis (the Fool)
  - 32 Suit cards: 4 suits × 8 cards (King, Queen, Knight, Jack + 4 pips)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CardType(Enum):
    TAROK = "tarok"
    SUIT = "suit"


class Suit(Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


class SuitRank(Enum):
    """Ranks within a suit, ordered lowest to highest."""
    PIP_1 = 1  # Red: 1, Black: 7
    PIP_2 = 2  # Red: 2, Black: 8
    PIP_3 = 3  # Red: 3, Black: 9
    PIP_4 = 4  # Red: 4, Black: 10
    JACK = 5
    KNIGHT = 6
    QUEEN = 7
    KING = 8


# Display labels for pip cards by suit color
_RED_PIP_LABELS = {SuitRank.PIP_1: "1", SuitRank.PIP_2: "2", SuitRank.PIP_3: "3", SuitRank.PIP_4: "4"}
_BLACK_PIP_LABELS = {SuitRank.PIP_1: "7", SuitRank.PIP_2: "8", SuitRank.PIP_3: "9", SuitRank.PIP_4: "10"}

_FACE_LABELS = {
    SuitRank.JACK: "J",
    SuitRank.KNIGHT: "C",  # Kavall/Cavalier
    SuitRank.QUEEN: "Q",
    SuitRank.KING: "K",
}

_TAROK_ROMAN = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
    6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X",
    11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV",
    16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX",
    21: "XXI", 22: "Škis",
}

SUIT_SYMBOLS = {
    Suit.HEARTS: "♥",
    Suit.DIAMONDS: "♦",
    Suit.CLUBS: "♣",
    Suit.SPADES: "♠",
}

_POINT_VALUES = {
    SuitRank.KING: 5,
    SuitRank.QUEEN: 4,
    SuitRank.KNIGHT: 3,
    SuitRank.JACK: 2,
}

SKIS = 22
MOND = 21
PAGAT = 1


@dataclass(frozen=True, slots=True)
class Card:
    card_type: CardType
    value: int  # Taroks: 1–22 (22=Škis). Suit cards: SuitRank.value
    suit: Suit | None = None

    @property
    def points(self) -> int:
        if self.card_type == CardType.TAROK:
            if self.value in (PAGAT, MOND, SKIS):
                return 5
            return 1
        rank = SuitRank(self.value)
        return _POINT_VALUES.get(rank, 1)

    @property
    def is_trula(self) -> bool:
        return self.card_type == CardType.TAROK and self.value in (PAGAT, MOND, SKIS)

    @property
    def is_king(self) -> bool:
        return self.card_type == CardType.SUIT and self.value == SuitRank.KING.value

    @property
    def label(self) -> str:
        if self.card_type == CardType.TAROK:
            return _TAROK_ROMAN[self.value]
        assert self.suit is not None
        rank = SuitRank(self.value)
        if rank.value <= 4:
            is_red = self.suit in (Suit.HEARTS, Suit.DIAMONDS)
            pip_labels = _RED_PIP_LABELS if is_red else _BLACK_PIP_LABELS
            return f"{pip_labels[rank]}{SUIT_SYMBOLS[self.suit]}"
        return f"{_FACE_LABELS[rank]}{SUIT_SYMBOLS[self.suit]}"

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Sort key: taroks first by value, then suits grouped."""
        if self.card_type == CardType.TAROK:
            return (0, self.value, 0)
        assert self.suit is not None
        return (1, list(Suit).index(self.suit), self.value)

    def beats(self, other: Card, lead_suit: Suit | None) -> bool:
        """Does this card beat `other` given the lead suit?"""
        if self.card_type == CardType.TAROK and other.card_type == CardType.TAROK:
            # Škis always wins... except it's special (captured if played last trick)
            if self.value == SKIS:
                return True
            if other.value == SKIS:
                return False
            return self.value > other.value
        if self.card_type == CardType.TAROK:
            return True  # Tarok beats any suit card
        if other.card_type == CardType.TAROK:
            return False
        # Both suit cards
        if self.suit == other.suit:
            return self.value > other.value
        # Different suits — only the lead suit wins
        return self.suit == lead_suit

    def __repr__(self) -> str:
        return f"Card({self.label})"


def tarok(value: int) -> Card:
    """Create a tarok card."""
    assert 1 <= value <= 22
    return Card(CardType.TAROK, value)


def suit_card(suit: Suit, rank: SuitRank) -> Card:
    """Create a suit card."""
    return Card(CardType.SUIT, rank.value, suit)


def _build_deck() -> tuple[Card, ...]:
    cards: list[Card] = []
    for v in range(1, 23):
        cards.append(tarok(v))
    for s in Suit:
        for r in SuitRank:
            cards.append(suit_card(s, r))
    return tuple(cards)


DECK: tuple[Card, ...] = _build_deck()
assert len(DECK) == 54
