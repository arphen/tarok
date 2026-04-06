"""Named condition and filter/transform functions for the rules engine.

Each function is registered in a dict so that YAML rule configs can reference
them by string name. All functions operate on a standardised TrickContext or
MoveContext dataclass to avoid coupling to GameState directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tarok.entities.card import Card, CardType, Suit, SKIS, PAGAT, MOND


# ---------------------------------------------------------------------------
# Contexts — lightweight views passed into condition/filter functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrickContext:
    """Read-only snapshot of a completed trick for evaluation."""
    cards: tuple[tuple[int, Card], ...]  # (player_idx, card) in play order
    lead_suit: Suit | None
    is_last_trick: bool = False
    contract_name: str | None = None

    @property
    def lead_card(self) -> Card:
        return self.cards[0][1]


@dataclass(frozen=True)
class MoveContext:
    """Read-only snapshot for legal-move generation."""
    hand: tuple[Card, ...]
    lead_card: Card | None          # None ⇒ player is leading
    lead_suit: Suit | None
    best_card: Card | None          # Current best card in the trick
    contract_name: str | None = None
    is_first_trick: bool = False
    is_last_trick: bool = False
    trick_card_count: int = 0       # How many cards already in the trick
    trick_cards: tuple[Card, ...] = ()  # Cards already played in this trick
    # Pre-computed suit/tarok indexes for O(1) lookups (auto-computed if None)
    by_suit: dict[Suit, tuple[Card, ...]] | None = None
    taroks: tuple[Card, ...] | None = None

    def __post_init__(self) -> None:
        if self.by_suit is None or self.taroks is None:
            _by_suit: dict[Suit, list[Card]] = {}
            _taroks: list[Card] = []
            for c in self.hand:
                if c.card_type == CardType.TAROK:
                    _taroks.append(c)
                elif c.suit is not None:
                    _by_suit.setdefault(c.suit, []).append(c)
            object.__setattr__(self, 'by_suit', {k: tuple(v) for k, v in _by_suit.items()})
            object.__setattr__(self, 'taroks', tuple(_taroks))


# ---------------------------------------------------------------------------
# Trick-eval condition registry
# ---------------------------------------------------------------------------

TrickConditionFn = Callable[[TrickContext], bool]
_trick_conditions: dict[str, TrickConditionFn] = {}


def trick_condition(name: str):
    """Decorator to register a trick-evaluation condition function."""
    def decorator(fn: TrickConditionFn) -> TrickConditionFn:
        _trick_conditions[name] = fn
        return fn
    return decorator


def get_trick_condition(name: str) -> TrickConditionFn:
    return _trick_conditions[name]


# ---------------------------------------------------------------------------
# Trick-eval transform registry
# ---------------------------------------------------------------------------

TrickTransformFn = Callable[[TrickContext], tuple[int, int]]
_trick_transforms: dict[str, TrickTransformFn] = {}


def trick_transform(name: str):
    """Decorator to register a trick-evaluation transform (returns winner, points)."""
    def decorator(fn: TrickTransformFn) -> TrickTransformFn:
        _trick_transforms[name] = fn
        return fn
    return decorator


def get_trick_transform(name: str) -> TrickTransformFn:
    return _trick_transforms[name]


# ---------------------------------------------------------------------------
# Legal-move condition registry
# ---------------------------------------------------------------------------

MoveConditionFn = Callable[[MoveContext], bool]
_move_conditions: dict[str, MoveConditionFn] = {}


def move_condition(name: str):
    def decorator(fn: MoveConditionFn) -> MoveConditionFn:
        _move_conditions[name] = fn
        return fn
    return decorator


def get_move_condition(name: str) -> MoveConditionFn:
    return _move_conditions[name]


# ---------------------------------------------------------------------------
# Legal-move filter registry
# ---------------------------------------------------------------------------

MoveFilterFn = Callable[[MoveContext], list[Card]]
_move_filters: dict[str, MoveFilterFn] = {}


def move_filter(name: str):
    def decorator(fn: MoveFilterFn) -> MoveFilterFn:
        _move_filters[name] = fn
        return fn
    return decorator


def get_move_filter(name: str) -> MoveFilterFn:
    return _move_filters[name]


# ---------------------------------------------------------------------------
# Legal-move ban filter registry  (negative filters, applied _after_ pipeline)
# ---------------------------------------------------------------------------

BanFilterFn = Callable[[MoveContext, list[Card]], list[Card]]
_ban_filters: dict[str, BanFilterFn] = {}


def ban_filter(name: str):
    def decorator(fn: BanFilterFn) -> BanFilterFn:
        _ban_filters[name] = fn
        return fn
    return decorator


def get_ban_filter(name: str) -> BanFilterFn:
    return _ban_filters[name]


# ===================================================================
# TRICK EVALUATION — conditions & transforms
# ===================================================================

@trick_condition("always")
def _always(_ctx: TrickContext) -> bool:
    return True


@trick_condition("skis_played")
def _skis_played(ctx: TrickContext) -> bool:
    """Škis (the Fool, tarok 22) was played in the trick."""
    return any(c.card_type == CardType.TAROK and c.value == SKIS for _, c in ctx.cards)


@trick_condition("skis_played_last_trick")
def _skis_played_last_trick(ctx: TrickContext) -> bool:
    """Škis played in the last trick of the game — it gets captured."""
    return ctx.is_last_trick and any(
        c.card_type == CardType.TAROK and c.value == SKIS for _, c in ctx.cards
    )


@trick_transform("standard_winner")
def _standard_winner(ctx: TrickContext) -> tuple[int, int]:
    """Standard trick resolution: highest card wins, sum points."""
    best_player, best_card = ctx.cards[0]
    for player, card in ctx.cards[1:]:
        if card.beats(best_card, ctx.lead_suit):
            best_player, best_card = player, card
    points = sum(c.points for _, c in ctx.cards)
    return best_player, points


@trick_transform("skis_captured")
def _skis_captured(ctx: TrickContext) -> tuple[int, int]:
    """Last-trick Škis rule: Škis is captured by the trick winner.

    Standard winner takes the trick AND the Škis points are included.
    (In the last trick, the Fool cannot escape — it's captured.)
    """
    # Same as standard — Škis stays in the trick and the winner gets all points
    return _standard_winner(ctx)


# ===================================================================
# LEGAL MOVE GENERATION — conditions & filters
# ===================================================================

@move_condition("always")
def _move_always(_ctx: MoveContext) -> bool:
    return True


@move_condition("is_leading")
def _is_leading(ctx: MoveContext) -> bool:
    return ctx.lead_card is None


@move_condition("has_lead_suit")
def _has_lead_suit(ctx: MoveContext) -> bool:
    if ctx.lead_suit is None:
        return False
    return ctx.lead_suit in ctx.by_suit


@move_condition("lead_is_tarok")
def _lead_is_tarok(ctx: MoveContext) -> bool:
    return ctx.lead_card is not None and ctx.lead_card.card_type == CardType.TAROK


@move_condition("has_taroks")
def _has_taroks(ctx: MoveContext) -> bool:
    return bool(ctx.taroks)


@move_condition("overplay_required")
def _overplay_required(ctx: MoveContext) -> bool:
    return ctx.contract_name in ("klop", "berac")


@move_condition("has_lead_suit_and_overplay")
def _has_lead_suit_overplay(ctx: MoveContext) -> bool:
    if ctx.lead_suit is None:
        return False
    return ctx.lead_suit in ctx.by_suit and ctx.contract_name in ("klop", "berac")


@move_condition("lead_is_tarok_and_overplay")
def _lead_tarok_overplay(ctx: MoveContext) -> bool:
    if ctx.lead_card is None or ctx.lead_card.card_type != CardType.TAROK:
        return False
    return bool(ctx.taroks) and ctx.contract_name in ("klop", "berac")


@move_condition("cant_follow_suit_has_taroks_overplay")
def _cant_follow_overplay(ctx: MoveContext) -> bool:
    if ctx.lead_card is None:
        return False
    if ctx.lead_card.card_type == CardType.TAROK:
        return False
    if ctx.lead_suit is not None and ctx.lead_suit in ctx.by_suit:
        return False
    return bool(ctx.taroks) and ctx.contract_name in ("klop", "berac")


@move_condition("cant_follow_suit_has_taroks")
def _cant_follow_has_taroks(ctx: MoveContext) -> bool:
    """Can't follow lead suit but has taroks (non-overplay)."""
    if ctx.lead_card is None:
        return False
    if ctx.lead_card.card_type == CardType.TAROK:
        return False
    if ctx.lead_suit is not None and ctx.lead_suit in ctx.by_suit:
        return False
    return bool(ctx.taroks)


@move_condition("pagat_exposed_in_overplay")
def _pagat_exposed(ctx: MoveContext) -> bool:
    """Pagat is in the legal move list AND there are other taroks AND overplay is on."""
    if ctx.contract_name not in ("klop", "berac"):
        return False
    return any(c.value == PAGAT for c in ctx.taroks) and len(ctx.taroks) > 1


# --- Filters ---

@move_filter("play_anything")
def _play_anything(ctx: MoveContext) -> list[Card]:
    return list(ctx.hand)


@move_filter("follow_suit")
def _follow_suit(ctx: MoveContext) -> list[Card]:
    return list(ctx.by_suit.get(ctx.lead_suit, ()))


@move_filter("follow_suit_overplay")
def _follow_suit_overplay(ctx: MoveContext) -> list[Card]:
    """Follow suit, but only cards that beat the current best."""
    same_suit = list(ctx.by_suit.get(ctx.lead_suit, ()))
    if ctx.best_card is not None:
        higher = [c for c in same_suit if c.beats(ctx.best_card, ctx.lead_suit)]
        if higher:
            return higher
    return same_suit


@move_filter("follow_tarok")
def _follow_tarok(ctx: MoveContext) -> list[Card]:
    return list(ctx.taroks)


@move_filter("follow_tarok_overplay")
def _follow_tarok_overplay(ctx: MoveContext) -> list[Card]:
    """Play taroks, but only ones that beat the current best."""
    taroks = list(ctx.taroks)
    if ctx.best_card is not None:
        higher = [c for c in taroks if c.beats(ctx.best_card, None)]
        if higher:
            return higher
    return taroks


@move_filter("trump_in")
def _trump_in(ctx: MoveContext) -> list[Card]:
    """Can't follow suit — must play tarok."""
    return list(ctx.taroks)


@move_filter("trump_in_overplay")
def _trump_in_overplay(ctx: MoveContext) -> list[Card]:
    """Can't follow suit — must play a tarok higher than the best tarok on the table."""
    taroks = list(ctx.taroks)
    if ctx.best_card is not None:
        higher = [c for c in taroks if c.beats(ctx.best_card, None)]
        if higher:
            return higher
    return taroks


# --- Ban Filters (negative / post-pipeline) ---

@ban_filter("ban_pagat_if_other_taroks")
def _ban_pagat(ctx: MoveContext, cards: list[Card]) -> list[Card]:
    """Remove Pagat from legal moves if other taroks are available.

    In Klop/Berac the Pagat must be forced out — you can't voluntarily play
    it if you have another tarok. EXCEPTION: if Mond and Škis are both
    already in the trick, Pagat MUST be played (it wins via the trula trick
    rule), so the ban is lifted.
    """
    has_pagat = any(c.card_type == CardType.TAROK and c.value == PAGAT for c in cards)
    if not has_pagat:
        return cards
    # Exception: Mond + Škis already on the table → Pagat wins, force it OUT
    if _mond_and_skis_in_trick(ctx.trick_cards):
        return [c for c in cards if c.card_type == CardType.TAROK and c.value == PAGAT]
    non_pagat = [c for c in cards if not (c.card_type == CardType.TAROK and c.value == PAGAT)]
    if non_pagat:
        return non_pagat
    return cards  # Pagat is the only card — must play it


@ban_filter("force_pagat_if_mond_skis_in_trick")
def _force_pagat_mond_skis(ctx: MoveContext, cards: list[Card]) -> list[Card]:
    """In overplay games, if Mond and Škis are already in the trick and
    the player has Pagat, they MUST play it (Pagat wins via trula rule).
    """
    has_pagat = any(c.card_type == CardType.TAROK and c.value == PAGAT for c in cards)
    if not has_pagat:
        return cards
    if _mond_and_skis_in_trick(ctx.trick_cards):
        return [c for c in cards if c.card_type == CardType.TAROK and c.value == PAGAT]
    return cards


# ===================================================================
# HELPERS
# ===================================================================

def _mond_and_skis_in_trick(trick_cards: tuple[Card, ...]) -> bool:
    """Check if both Mond (XXI) and Škis (the Fool) are in the trick."""
    values = {c.value for c in trick_cards if c.card_type == CardType.TAROK}
    return MOND in values and SKIS in values


def _all_trula_in_trick(cards: tuple[tuple[int, Card], ...]) -> bool:
    """Check if Pagat, Mond, and Škis are all in the same trick."""
    values = {c.value for _, c in cards if c.card_type == CardType.TAROK}
    return {PAGAT, MOND, SKIS}.issubset(values)


# ===================================================================
# TRICK EVALUATION — Pagat-Mond-Škis (trula trick) rule
# ===================================================================

@trick_condition("all_trula_in_trick")
def _all_trula_cond(ctx: TrickContext) -> bool:
    """All three trula cards (Pagat, Mond, Škis) are in the same trick."""
    return _all_trula_in_trick(ctx.cards)


@trick_transform("pagat_wins_trula_trick")
def _pagat_wins_trula(ctx: TrickContext) -> tuple[int, int]:
    """When all 3 trula cards meet in one trick, Pagat always wins."""
    pagat_player = next(
        p for p, c in ctx.cards
        if c.card_type == CardType.TAROK and c.value == PAGAT
    )
    points = sum(c.points for _, c in ctx.cards)
    return pagat_player, points


# ===================================================================
# TRICK EVALUATION — Barvni Valat (colour valat) rule
# ===================================================================

@trick_condition("is_barvni_valat")
def _is_barvni_valat(ctx: TrickContext) -> bool:
    """Contract is barvni_valat — suit cards beat taroks."""
    return ctx.contract_name == "barvni_valat"


@trick_transform("barvni_valat_winner")
def _barvni_valat_winner(ctx: TrickContext) -> tuple[int, int]:
    """In barvni valat, suit cards beat taroks. Among suit cards,
    follow normal rules (lead suit wins, higher rank wins within suit).
    Taroks only win if NO suit cards were played.
    """
    lead_suit = ctx.lead_suit
    # Separate suit cards and taroks
    suit_cards = [(p, c) for p, c in ctx.cards if c.card_type == CardType.SUIT]

    if suit_cards:
        # Suit cards present — they beat all taroks
        # Among suit cards: lead suit wins, then higher rank
        best_player, best_card = suit_cards[0]
        for player, card in suit_cards[1:]:
            if card.suit == best_card.suit:
                if card.value > best_card.value:
                    best_player, best_card = player, card
            elif card.suit == lead_suit:
                best_player, best_card = player, card
        points = sum(c.points for _, c in ctx.cards)
        return best_player, points
    else:
        # All taroks — standard tarok resolution
        return _standard_winner(ctx)


# ===================================================================
# LEGAL MOVE — Mond+Škis force Pagat condition
# ===================================================================

@move_condition("mond_skis_in_trick_has_pagat_overplay")
def _mond_skis_force_pagat(ctx: MoveContext) -> bool:
    """In overplay games, Mond+Škis on table and player holds Pagat."""
    if ctx.contract_name not in ("klop", "berac"):
        return False
    if not _mond_and_skis_in_trick(ctx.trick_cards):
        return False
    return any(c.card_type == CardType.TAROK and c.value == PAGAT for c in ctx.hand)
