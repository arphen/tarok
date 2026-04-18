"""Thin type shim — representational types backed by Rust engine constants.

NO game logic lives here. Scoring, legal moves, trick evaluation — all in Rust.
These are labels, enums, and data bags for the Python/UI layer.
"""

from __future__ import annotations

from enum import Enum

import tarok_engine as te

# ---------------------------------------------------------------------------
# Card‑related enums
# ---------------------------------------------------------------------------


class CardType(Enum):
    TAROK = "tarok"
    SUIT = "suit"


class Suit(Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


_SUIT_BY_IDX = {0: Suit.HEARTS, 1: Suit.DIAMONDS, 2: Suit.CLUBS, 3: Suit.SPADES}


class SuitRank(Enum):
    PIP_1 = 1
    PIP_2 = 2
    PIP_3 = 3
    PIP_4 = 4
    JACK = 5
    KNIGHT = 6
    QUEEN = 7
    KING = 8


# ---------------------------------------------------------------------------
# Card — thin wrapper around a Rust u8 card index
# ---------------------------------------------------------------------------


class Card:
    """Lightweight card backed by a Rust u8 index (0‑53)."""

    __slots__ = ("_idx",)

    def __init__(self, idx: int) -> None:
        self._idx = idx

    # -- properties derived from the index layout (no game logic) ----------

    @property
    def card_type(self) -> CardType:
        return CardType.TAROK if self._idx < 22 else CardType.SUIT

    @property
    def value(self) -> int:
        if self._idx < 22:
            return self._idx + 1  # taroks 1‑22
        return ((self._idx - 22) % 8) + 1  # suit rank 1‑8

    @property
    def suit(self) -> Suit | None:
        if self._idx < 22:
            return None
        return _SUIT_BY_IDX[(self._idx - 22) // 8]

    @property
    def suit_rank(self) -> SuitRank | None:
        if self._idx < 22:
            return None
        return SuitRank(self.value)

    @property
    def label(self) -> str:
        return te.RustGameState.card_label(self._idx)

    @property
    def points(self) -> int:
        return te.RustGameState.card_points(self._idx)

    @property
    def is_king(self) -> bool:
        return self._idx >= 22 and (self._idx - 22) % 8 == 7

    @property
    def sort_key(self) -> int:
        return self._idx

    # -- identity ----------------------------------------------------------

    def __hash__(self) -> int:
        return self._idx

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            return self._idx == other._idx
        return NotImplemented

    def __repr__(self) -> str:
        return f"Card({self.label})"

    def __lt__(self, other: Card) -> bool:
        return self._idx < other._idx


# ---------------------------------------------------------------------------
# Deck & convenience constructors
# ---------------------------------------------------------------------------

DECK: list[Card] = [Card(i) for i in range(54)]

# Tarok value constants (NOT Card objects — used as tarok(PAGAT) etc.)
PAGAT = 1  # tarok value 1
MOND = 21  # tarok value 21
SKIS = 22  # tarok value 22


def tarok(value: int) -> Card:
    """Create a tarok card by value (1‑22)."""
    return DECK[value - 1]


def suit_card(suit: Suit, rank: SuitRank) -> Card:
    """Create a suit card by Suit and SuitRank."""
    suit_idx = list(Suit).index(suit)
    return DECK[22 + suit_idx * 8 + (rank.value - 1)]


# ---------------------------------------------------------------------------
# Game‑state enums
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    DEALING = "dealing"
    BIDDING = "bidding"
    KING_CALLING = "king_calling"
    TALON_EXCHANGE = "talon_exchange"
    ANNOUNCEMENTS = "announcements"
    TRICK_PLAY = "trick_play"
    SCORING = "scoring"
    FINISHED = "finished"


class Contract(int, Enum):
    """Contract values match the frontend CONTRACT_NAMES keys."""

    KLOP = -99
    THREE = 3
    TWO = 2
    ONE = 1
    SOLO_THREE = -3
    SOLO_TWO = -2
    SOLO_ONE = -1
    SOLO = 0
    BERAC = -100
    BARVNI_VALAT = -101

    # -- lookup tables derived from Rust -----------------------------------

    @property
    def is_biddable(self) -> bool:
        return self not in (Contract.KLOP, Contract.BARVNI_VALAT)

    @property
    def strength(self) -> int:
        return _CONTRACT_STRENGTH[self]

    @property
    def talon_cards(self) -> int:
        return _CONTRACT_TALON_CARDS.get(self, 0)

    @property
    def base_value(self) -> int:
        return _CONTRACT_BASE_VALUE[self]

    def is_solo(self) -> bool:
        return self in (Contract.SOLO_THREE, Contract.SOLO_TWO, Contract.SOLO_ONE, Contract.SOLO)

    @property
    def is_klop(self) -> bool:
        return self == Contract.KLOP

    @property
    def is_berac(self) -> bool:
        return self == Contract.BERAC

    @property
    def is_barvni_valat(self) -> bool:
        return self == Contract.BARVNI_VALAT


# Lookup tables matching Rust Contract impl
_CONTRACT_STRENGTH: dict[Contract, int] = {
    Contract.KLOP: 0,
    Contract.THREE: 1,
    Contract.TWO: 2,
    Contract.ONE: 3,
    Contract.SOLO_THREE: 4,
    Contract.SOLO_TWO: 5,
    Contract.SOLO_ONE: 6,
    Contract.SOLO: 7,
    Contract.BERAC: 8,
    Contract.BARVNI_VALAT: 9,
}

_CONTRACT_TALON_CARDS: dict[Contract, int] = {
    Contract.THREE: 3,
    Contract.TWO: 2,
    Contract.ONE: 1,
    Contract.SOLO_THREE: 3,
    Contract.SOLO_TWO: 2,
    Contract.SOLO_ONE: 1,
}

_CONTRACT_BASE_VALUE: dict[Contract, int] = {
    Contract.KLOP: 0,
    Contract.THREE: 10,
    Contract.TWO: 20,
    Contract.ONE: 30,
    Contract.SOLO_THREE: 40,
    Contract.SOLO_TWO: 50,
    Contract.SOLO_ONE: 60,
    Contract.SOLO: 80,
    Contract.BERAC: 70,
    Contract.BARVNI_VALAT: 125,
}


class PlayerRole(str, Enum):
    DECLARER = "declarer"
    PARTNER = "partner"
    OPPONENT = "opponent"


class Team(str, Enum):
    DECLARER_TEAM = "declarer_team"
    OPPONENT_TEAM = "opponent_team"


class Announcement(str, Enum):
    TRULA = "trula"
    KINGS = "kings"
    PAGAT_ULTIMO = "pagat_ultimo"
    VALAT = "valat"


class KontraLevel(Enum):
    NONE = 1
    KONTRA = 2
    RE = 4
    SUB = 8

    @property
    def next_level(self) -> KontraLevel | None:
        _next = {
            KontraLevel.NONE: KontraLevel.KONTRA,
            KontraLevel.KONTRA: KontraLevel.RE,
            KontraLevel.RE: KontraLevel.SUB,
        }
        return _next.get(self)

    @property
    def is_opponent_turn(self) -> bool:
        return self in (KontraLevel.NONE, KontraLevel.RE)


# ---------------------------------------------------------------------------
# Bid (simple record)
# ---------------------------------------------------------------------------


class Bid:
    __slots__ = ("player", "contract")

    def __init__(self, player: int, contract: Contract | None) -> None:
        self.player = player
        self.contract = contract


# ---------------------------------------------------------------------------
# Trick (display‑only snapshot, NOT the Rust Trick)
# ---------------------------------------------------------------------------


class Trick:
    """Read‑only trick snapshot for observer / serialisation layer."""

    __slots__ = ("lead_player", "cards", "_winner")

    def __init__(
        self,
        lead_player: int = 0,
        cards: list[tuple[int, Card]] | None = None,
        winner: int | None = None,
    ) -> None:
        self.lead_player = lead_player
        self.cards = cards if cards is not None else []
        self._winner = winner

    def winner(self) -> int:
        if self._winner is not None:
            return self._winner
        raise ValueError("winner not set on this trick snapshot")


# ---------------------------------------------------------------------------
# GameState — bag of attributes; NO game logic
# ---------------------------------------------------------------------------


class GameState:
    """Lightweight state view passed to observers and player ports."""

    def __init__(self, **kwargs) -> None:
        self.dealer: int = kwargs.get("dealer", 0)
        self.num_players: int = kwargs.get("num_players", 4)
        self.phase: Phase = kwargs.get("phase", Phase.DEALING)
        self.hands: list[list[Card]] = kwargs.get("hands", [[] for _ in range(4)])
        self.contract: Contract | None = kwargs.get("contract", None)
        self.declarer: int | None = kwargs.get("declarer", None)
        self.partner: int | None = kwargs.get("partner", None)
        self.called_king: Card | None = kwargs.get("called_king", None)
        self.talon: list[Card] = kwargs.get("talon", [])
        self.talon_revealed: list[list[Card]] = kwargs.get("talon_revealed", [])
        self.put_down: list[Card] = kwargs.get("put_down", [])
        self.bids: list = kwargs.get("bids", [])
        self.tricks: list = kwargs.get("tricks", [])
        self.current_trick: object | None = kwargs.get("current_trick", None)
        self.announcements: dict = kwargs.get("announcements", {})
        self.kontra_levels: dict = kwargs.get("kontra_levels", {})
        self.roles: dict = kwargs.get("roles", {})
        self.scores: dict = kwargs.get("scores", {})
        self.current_player: int = kwargs.get("current_player", 0)
        self.initial_tarok_counts: dict = kwargs.get("initial_tarok_counts", {})

    # -- derived properties (pure lookups, no game logic) ------------------

    @property
    def tricks_played(self) -> int:
        return len(self.tricks)

    @property
    def is_partner_revealed(self) -> bool:
        return self.partner is not None

    def get_team(self, player_idx: int) -> Team:
        role = self.roles.get(player_idx, PlayerRole.OPPONENT)
        if role in (PlayerRole.DECLARER, PlayerRole.PARTNER):
            return Team.DECLARER_TEAM
        return Team.OPPONENT_TEAM


# ---------------------------------------------------------------------------
# Rust‑delegated scoring helpers
# ---------------------------------------------------------------------------


def compute_card_points(cards: list[Card]) -> int:
    """Delegate to Rust engine."""
    return te.RustGameState.compute_card_points([c._idx for c in cards])


def score_game(gs: te.RustGameState) -> list[int]:
    """Delegate to Rust engine."""
    return list(gs.score_game())


# Scoring constants (from Rust scoring module)
TOTAL_GAME_POINTS = 70
POINT_HALF = 35
