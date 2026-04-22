"""Thin type shim — representational types backed by Rust engine constants.

NO game logic lives here. Scoring, legal moves, trick evaluation — all in Rust.
These are labels, enums, and data bags for the Python/UI layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

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

    def beats(self, other: Card, lead_suit: Suit | None) -> bool:
        """Compatibility shim delegating trick comparison to Rust."""
        lead_suit_idx = None if lead_suit is None else list(Suit).index(lead_suit)
        return bool(te.RustGameState.card_beats(self._idx, other._idx, lead_suit_idx))


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
    KING_ULTIMO = "king_ultimo"
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

    def __init__(
        self,
        *,
        dealer: int = 0,
        num_players: int = 4,
        phase: Phase = Phase.DEALING,
        hands: list[list[Card]] | None = None,
        contract: Contract | None = None,
        declarer: int | None = None,
        partner: int | None = None,
        called_king: Card | None = None,
        talon: list[Card] | None = None,
        talon_revealed: list[list[Card]] | None = None,
        put_down: list[Card] | None = None,
        bids: list | None = None,
        tricks: list | None = None,
        current_trick: object | None = None,
        announcements: dict | None = None,
        kontra_levels: dict | None = None,
        roles: dict | None = None,
        scores: dict | None = None,
        current_player: int = 0,
        current_bidder: int | None = None,
        initial_tarok_counts: dict | None = None,
        # Rust-backed dynamic helpers populated by bridge code.
        legal_bids: Callable[[int], list[int | None]] | None = None,
        legal_plays: Callable[[int], list[Card]] | None = None,
        callable_kings: Callable[[], list[Card]] | None = None,
        # Snapshot carry-forward fields — internal state threaded between functional wrappers.
        _trick_in_progress: Any = None,
        _talon_groups: Any = None,
        _bid_passed: list[bool] | None = None,
        _bid_highest: Any = None,
        _bid_winner: Any = None,
        # Link to underlying Rust game state when available.
        _rust_gs: Any = None,
    ) -> None:
        self.dealer = dealer
        self.num_players = num_players
        self.phase = phase
        self.hands: list[list[Card]] = hands if hands is not None else [[] for _ in range(4)]
        self.contract = contract
        self.declarer = declarer
        self.partner = partner
        self.called_king = called_king
        self.talon: list[Card] = talon if talon is not None else []
        self.talon_revealed: list[list[Card]] = talon_revealed if talon_revealed is not None else []
        self.put_down: list[Card] = put_down if put_down is not None else []
        self.bids: list = bids if bids is not None else []
        self.tricks: list = tricks if tricks is not None else []
        self.current_trick = current_trick
        self.announcements: dict = announcements if announcements is not None else {}
        self.kontra_levels: dict = kontra_levels if kontra_levels is not None else {}
        self.roles: dict = roles if roles is not None else {}
        self.scores: dict = scores if scores is not None else {}
        self.current_player = current_player
        self.current_bidder: int | None = (
            current_bidder if current_bidder is not None else current_player
        )
        self.initial_tarok_counts: dict = (
            initial_tarok_counts if initial_tarok_counts is not None else {}
        )
        self.legal_bids: Callable[[int], list[int | None]] = (
            legal_bids if legal_bids is not None else lambda _: []
        )
        self.legal_plays: Callable[[int], list[Card]] = (
            legal_plays if legal_plays is not None else lambda _: []
        )
        self.callable_kings: Callable[[], list[Card]] = (
            callable_kings if callable_kings is not None else lambda: []
        )
        self._trick_in_progress = _trick_in_progress
        self._talon_groups = _talon_groups
        self._bid_passed: list[bool] = _bid_passed if _bid_passed is not None else [False] * 4
        self._bid_highest = _bid_highest
        self._bid_winner = _bid_winner
        self._rust_gs = _rust_gs

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


def score_game(gs: Any) -> list[int]:
    """Delegate to Rust engine (accepts Rust state or bridged GameState)."""
    rust_gs = getattr(gs, "_rust_gs", gs)
    return list(rust_gs.score_game())


# Scoring constants (from Rust scoring module)
TOTAL_GAME_POINTS = 70
POINT_HALF = 35
