"""Game state domain entity.

Represents the full state machine for a 4-player Slovenian Tarok game.
Partners are determined by king calling (2v2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tarok.entities.card import Card, CardType, Suit, SuitRank, SKIS, PAGAT, MOND


class Phase(Enum):
    DEALING = "dealing"
    BIDDING = "bidding"
    KING_CALLING = "king_calling"
    TALON_EXCHANGE = "talon_exchange"
    ANNOUNCEMENTS = "announcements"
    TRICK_PLAY = "trick_play"
    SCORING = "scoring"
    FINISHED = "finished"


class Contract(Enum):
    """Contracts ordered by ascending strength (higher outbids lower)."""
    KLOP = -99  # All pass → each player for themselves, avoid taking points
    THREE = 3   # Take 3 talon cards
    TWO = 2     # Take 2 talon cards
    ONE = 1     # Take 1 talon card
    SOLO_THREE = -3  # Solo but pick 3 (no partner)
    SOLO_TWO = -2
    SOLO_ONE = -1
    SOLO = 0    # No talon, no partner
    BERAC = -100  # Declarer bids to take 0 tricks, no talon, plays solo
    BARVNI_VALAT = -101  # Colour valat: suits beat taroks, must take all tricks

    @property
    def is_solo(self) -> bool:
        return self.value <= 0 and self not in (
            Contract.KLOP, Contract.BERAC, Contract.BARVNI_VALAT,
        )

    @property
    def is_klop(self) -> bool:
        return self == Contract.KLOP

    @property
    def is_berac(self) -> bool:
        return self == Contract.BERAC

    @property
    def is_barvni_valat(self) -> bool:
        return self == Contract.BARVNI_VALAT

    @property
    def requires_overplay(self) -> bool:
        """Must play a higher card than what's on the table."""
        return self in (Contract.KLOP, Contract.BERAC)

    @property
    def talon_cards(self) -> int:
        if self in (Contract.KLOP, Contract.BERAC, Contract.BARVNI_VALAT):
            return 0
        return abs(self.value)

    @property
    def strength(self) -> int:
        """Higher number = stronger bid. KLOP is not biddable."""
        order = {
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
        return order[self]

    @property
    def is_biddable(self) -> bool:
        """Can a player actively bid this contract?"""
        return self not in (Contract.KLOP, Contract.BARVNI_VALAT)


class PlayerRole(Enum):
    DECLARER = "declarer"
    PARTNER = "partner"      # Holds the called king
    OPPONENT = "opponent"


class Team(Enum):
    DECLARER_TEAM = "declarer_team"
    OPPONENT_TEAM = "opponent_team"


class Announcement(Enum):
    TRULA = "trula"          # Declarer team will collect all 3 trula cards
    KINGS = "kings"          # Declarer team will collect all 4 kings
    PAGAT_ULTIMO = "pagat_ultimo"  # Pagat played in the last trick and wins
    VALAT = "valat"          # Win all 12 tricks
    BARVNI_VALAT = "barvni_valat"  # Colour valat — announced after solo, must take all tricks


class KontraLevel(Enum):
    """Counter-doubling levels on the base game or individual announcements.

    Opponents say KONTRA (double), declarers respond with RE (4×),
    opponents can respond with SUB/MORT (8×).
    """
    NONE = 1
    KONTRA = 2      # Opponents double
    RE = 4          # Declarers re-double
    SUB = 8         # Opponents re-re-double (sub-kontra / mort)

    @property
    def next_level(self) -> "KontraLevel | None":
        """The next escalation level, or None if maxed out."""
        _chain = {
            KontraLevel.NONE: KontraLevel.KONTRA,
            KontraLevel.KONTRA: KontraLevel.RE,
            KontraLevel.RE: KontraLevel.SUB,
            KontraLevel.SUB: None,
        }
        return _chain[self]

    @property
    def is_opponent_turn(self) -> bool:
        """Is it the opponents' turn to escalate?"""
        return self in (KontraLevel.NONE, KontraLevel.RE)


@dataclass
class Trick:
    lead_player: int
    cards: list[tuple[int, Card]] = field(default_factory=list)  # (player_idx, card)

    @property
    def lead_suit(self) -> Suit | None:
        if not self.cards:
            return None
        _, lead_card = self.cards[0]
        if lead_card.card_type == CardType.TAROK:
            return None  # Taroks don't establish a suit lead
        return lead_card.suit

    @property
    def is_complete(self) -> bool:
        return len(self.cards) == 4

    def winner(self, is_last_trick: bool = False, contract_name: str | None = None) -> int:
        assert self.is_complete
        from tarok.engine.conditions import TrickContext
        from tarok.engine.trick_eval import evaluate_trick
        ctx = TrickContext(
            cards=tuple(self.cards),
            lead_suit=self.lead_suit,
            is_last_trick=is_last_trick,
            contract_name=contract_name,
        )
        trace = evaluate_trick(ctx)
        return trace.result[0]

    def winner_trace(self, is_last_trick: bool = False, contract_name: str | None = None):
        """Return full Trace from trick evaluation (for audit trails)."""
        assert self.is_complete
        from tarok.engine.conditions import TrickContext
        from tarok.engine.trick_eval import evaluate_trick
        ctx = TrickContext(
            cards=tuple(self.cards),
            lead_suit=self.lead_suit,
            is_last_trick=is_last_trick,
            contract_name=contract_name,
        )
        return evaluate_trick(ctx)

    @property
    def points(self) -> int:
        return sum(c.points for _, c in self.cards)


@dataclass
class Bid:
    player: int
    contract: Contract | None  # None = pass


@dataclass
class GameState:
    """Immutable-ish representation of the full game state."""

    num_players: int = 4
    phase: Phase = Phase.DEALING

    # Cards
    hands: list[list[Card]] = field(default_factory=lambda: [[] for _ in range(4)])
    talon: list[Card] = field(default_factory=list)

    # Bidding
    bids: list[Bid] = field(default_factory=list)
    current_bidder: int = 0  # forehand (first to bid)
    declarer: int | None = None
    contract: Contract | None = None

    # King calling
    called_king: Card | None = None
    partner: int | None = None  # Revealed when king is played (or never for solo)

    # Talon exchange
    talon_revealed: list[list[Card]] = field(default_factory=list)
    put_down: list[Card] = field(default_factory=list)  # Cards declarer put back

    # Announcements
    announcements: dict[int, list[Announcement]] = field(default_factory=dict)
    # Kontra/Re/Sub levels: 'game' key for base contract, Announcement keys for bonuses
    kontra_levels: dict[str, KontraLevel] = field(default_factory=dict)

    # Trick play
    tricks: list[Trick] = field(default_factory=list)
    current_trick: Trick | None = None
    current_player: int = 0

    # Deal-time stats (preserved for analytics)
    initial_tarok_counts: list[int] = field(default_factory=list)

    # Scoring
    declarer_team_tricks: list[Trick] = field(default_factory=list)
    opponent_team_tricks: list[Trick] = field(default_factory=list)
    scores: dict[int, int] = field(default_factory=dict)  # player -> score delta

    # Roles
    roles: dict[int, PlayerRole] = field(default_factory=dict)

    # Tracking
    round_number: int = 0
    dealer: int = 0

    def get_team(self, player: int) -> Team:
        role = self.roles.get(player, PlayerRole.OPPONENT)
        if role in (PlayerRole.DECLARER, PlayerRole.PARTNER):
            return Team.DECLARER_TEAM
        return Team.OPPONENT_TEAM

    def legal_plays(self, player: int) -> list[Card]:
        """Which cards can this player legally play in the current trick?

        Delegates to the rules engine for all business logic.
        """
        from tarok.engine.conditions import MoveContext
        from tarok.engine.legal_moves import generate_legal_moves

        hand = self.hands[player]
        if not hand:
            return []

        lead_card = None
        lead_suit = None
        best_card = None
        trick_card_count = 0
        trick_cards: tuple[Card, ...] = ()

        if self.current_trick is not None and self.current_trick.cards:
            lead_card = self.current_trick.cards[0][1]
            lead_suit = self.current_trick.lead_suit
            trick_card_count = len(self.current_trick.cards)
            trick_cards = tuple(c for _, c in self.current_trick.cards)
            # Find the current best card in the trick
            best_card = lead_card
            for _, c in self.current_trick.cards:
                if c.beats(best_card, lead_suit):
                    best_card = c

        contract_name = self.contract.value if self.contract else None
        # Normalise contract name to the string key used by conditions
        if self.contract is not None:
            _cname_map = {
                Contract.KLOP: "klop", Contract.BERAC: "berac",
                Contract.BARVNI_VALAT: "barvni_valat",
                Contract.THREE: "three", Contract.TWO: "two", Contract.ONE: "one",
                Contract.SOLO_THREE: "solo_three", Contract.SOLO_TWO: "solo_two",
                Contract.SOLO_ONE: "solo_one", Contract.SOLO: "solo",
            }
            contract_name = _cname_map.get(self.contract)

        ctx = MoveContext(
            hand=tuple(hand),
            lead_card=lead_card,
            lead_suit=lead_suit,
            best_card=best_card,
            contract_name=contract_name,
            is_first_trick=(self.tricks_played == 0),
            is_last_trick=self.is_last_trick,
            trick_card_count=trick_card_count,
            trick_cards=trick_cards,
        )

        trace = generate_legal_moves(ctx)
        return trace.result

    def legal_plays_trace(self, player: int):
        """Return full Trace from legal move generation (for audit trails)."""
        from tarok.engine.conditions import MoveContext
        from tarok.engine.legal_moves import generate_legal_moves

        hand = self.hands[player]
        if not hand:
            from tarok.engine.core import Trace
            return Trace(result=[], triggered_rule="EmptyHand", priority=-1)

        lead_card = None
        lead_suit = None
        best_card = None
        trick_card_count = 0
        trick_cards: tuple[Card, ...] = ()

        if self.current_trick is not None and self.current_trick.cards:
            lead_card = self.current_trick.cards[0][1]
            lead_suit = self.current_trick.lead_suit
            trick_card_count = len(self.current_trick.cards)
            trick_cards = tuple(c for _, c in self.current_trick.cards)
            best_card = lead_card
            for _, c in self.current_trick.cards:
                if c.beats(best_card, lead_suit):
                    best_card = c

        contract_name = None
        if self.contract is not None:
            _cname_map = {
                Contract.KLOP: "klop", Contract.BERAC: "berac",
                Contract.BARVNI_VALAT: "barvni_valat",
                Contract.THREE: "three", Contract.TWO: "two", Contract.ONE: "one",
                Contract.SOLO_THREE: "solo_three", Contract.SOLO_TWO: "solo_two",
                Contract.SOLO_ONE: "solo_one", Contract.SOLO: "solo",
            }
            contract_name = _cname_map.get(self.contract)

        ctx = MoveContext(
            hand=tuple(hand),
            lead_card=lead_card,
            lead_suit=lead_suit,
            best_card=best_card,
            contract_name=contract_name,
            is_first_trick=(self.tricks_played == 0),
            is_last_trick=self.is_last_trick,
            trick_card_count=trick_card_count,
            trick_cards=trick_cards,
        )

        return generate_legal_moves(ctx)

    @property
    def forehand(self) -> int:
        """The obligated player (obvezen) — first player after the dealer."""
        return (self.dealer + 1) % self.num_players

    def legal_bids(self, player: int) -> list[Contract | None]:
        """Which bids can this player make? None = pass.

        Obvezen (forehand) rules:
        - Only forehand may bid THREE.
        - Forehand can *match* the current highest bid (>= instead of >).
        - Other players must strictly outbid (>).
        """
        is_forehand = player == self.forehand
        biddable = [
            c for c in Contract
            if c.is_biddable and (c != Contract.THREE or is_forehand)
        ]

        # Find the current highest bid
        highest = max(
            (b.contract for b in self.bids if b.contract is not None),
            key=lambda c: c.strength,
            default=None,
        )

        options: list[Contract | None] = [None]  # Can always pass
        for c in biddable:
            if highest is None:
                options.append(c)
            elif is_forehand and c.strength >= highest.strength:
                # Forehand can match the current highest bid
                options.append(c)
            elif c.strength > highest.strength:
                options.append(c)

        return options

    def callable_kings(self) -> list[Card]:
        """Which kings can the declarer call?"""
        assert self.declarer is not None
        hand = self.hands[self.declarer]
        all_kings = [
            Card(CardType.SUIT, SuitRank.KING.value, s) for s in Suit
        ]
        # Can call any king NOT in own hand
        callable = [k for k in all_kings if k not in hand]
        if not callable:
            # Has all 4 kings — can call a queen
            callable = [
                Card(CardType.SUIT, SuitRank.QUEEN.value, s)
                for s in Suit
                if Card(CardType.SUIT, SuitRank.QUEEN.value, s) not in hand
            ]
        return callable

    @property
    def is_partner_revealed(self) -> bool:
        if self.called_king is None:
            return False
        for trick in self.tricks:
            for _, card in trick.cards:
                if card == self.called_king:
                    return True
        if self.current_trick:
            for _, card in self.current_trick.cards:
                if card == self.called_king:
                    return True
        return False

    @property
    def tricks_played(self) -> int:
        return len(self.tricks)

    @property
    def is_last_trick(self) -> bool:
        return self.tricks_played == 11 and self.current_trick is not None

    def visible_state_for(self, player: int) -> dict:
        """Return the game state visible to a specific player (for AI observation)."""
        return {
            "phase": self.phase.value,
            "hand": list(self.hands[player]),
            "hand_size": [len(h) for h in self.hands],
            "talon_size": len(self.talon),
            "bids": [(b.player, b.contract.value if b.contract else None) for b in self.bids],
            "contract": self.contract.value if self.contract else None,
            "declarer": self.declarer,
            "called_king": self.called_king,
            "partner_revealed": self.is_partner_revealed,
            "partner": self.partner if self.is_partner_revealed else None,
            "current_trick": (
                [(p, c) for p, c in self.current_trick.cards]
                if self.current_trick
                else []
            ),
            "tricks_played": self.tricks_played,
            "my_role": self.roles.get(player, PlayerRole.OPPONENT).value,
            "played_cards": [
                (p, c) for trick in self.tricks for p, c in trick.cards
            ],
        }
