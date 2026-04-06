"""Named condition and filter/transform functions for the rules engine.

Each function is registered in a dict so that YAML rule configs can reference
them by string name. All functions operate on a standardised TrickContext or
MoveContext dataclass to avoid coupling to GameState directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tarok.entities.card import Card, CardType, Suit, SKIS, PAGAT


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
    return any(c.suit == ctx.lead_suit for c in ctx.hand)


@move_condition("lead_is_tarok")
def _lead_is_tarok(ctx: MoveContext) -> bool:
    return ctx.lead_card is not None and ctx.lead_card.card_type == CardType.TAROK


@move_condition("has_taroks")
def _has_taroks(ctx: MoveContext) -> bool:
    return any(c.card_type == CardType.TAROK for c in ctx.hand)


@move_condition("overplay_required")
def _overplay_required(ctx: MoveContext) -> bool:
    return ctx.contract_name in ("klop", "berac")


@move_condition("has_lead_suit_and_overplay")
def _has_lead_suit_overplay(ctx: MoveContext) -> bool:
    if ctx.lead_suit is None:
        return False
    has_suit = any(c.suit == ctx.lead_suit for c in ctx.hand)
    return has_suit and ctx.contract_name in ("klop", "berac")


@move_condition("lead_is_tarok_and_overplay")
def _lead_tarok_overplay(ctx: MoveContext) -> bool:
    if ctx.lead_card is None or ctx.lead_card.card_type != CardType.TAROK:
        return False
    has_taroks = any(c.card_type == CardType.TAROK for c in ctx.hand)
    return has_taroks and ctx.contract_name in ("klop", "berac")


@move_condition("cant_follow_suit_has_taroks_overplay")
def _cant_follow_overplay(ctx: MoveContext) -> bool:
    if ctx.lead_card is None:
        return False
    if ctx.lead_card.card_type == CardType.TAROK:
        return False
    if ctx.lead_suit is not None and any(c.suit == ctx.lead_suit for c in ctx.hand):
        return False
    has_taroks = any(c.card_type == CardType.TAROK for c in ctx.hand)
    return has_taroks and ctx.contract_name in ("klop", "berac")


@move_condition("cant_follow_suit_has_taroks")
def _cant_follow_has_taroks(ctx: MoveContext) -> bool:
    """Can't follow lead suit but has taroks (non-overplay)."""
    if ctx.lead_card is None:
        return False
    if ctx.lead_card.card_type == CardType.TAROK:
        return False
    if ctx.lead_suit is not None and any(c.suit == ctx.lead_suit for c in ctx.hand):
        return False
    return any(c.card_type == CardType.TAROK for c in ctx.hand)


@move_condition("pagat_exposed_in_overplay")
def _pagat_exposed(ctx: MoveContext) -> bool:
    """Pagat is in the legal move list AND there are other taroks AND overplay is on."""
    if ctx.contract_name not in ("klop", "berac"):
        return False
    has_pagat = any(c.card_type == CardType.TAROK and c.value == PAGAT for c in ctx.hand)
    tarok_count = sum(1 for c in ctx.hand if c.card_type == CardType.TAROK)
    return has_pagat and tarok_count > 1


# --- Filters ---

@move_filter("play_anything")
def _play_anything(ctx: MoveContext) -> list[Card]:
    return list(ctx.hand)


@move_filter("follow_suit")
def _follow_suit(ctx: MoveContext) -> list[Card]:
    return [c for c in ctx.hand if c.suit == ctx.lead_suit]


@move_filter("follow_suit_overplay")
def _follow_suit_overplay(ctx: MoveContext) -> list[Card]:
    """Follow suit, but only cards that beat the current best."""
    same_suit = [c for c in ctx.hand if c.suit == ctx.lead_suit]
    if ctx.best_card is not None:
        higher = [c for c in same_suit if c.beats(ctx.best_card, ctx.lead_suit)]
        if higher:
            return higher
    return same_suit


@move_filter("follow_tarok")
def _follow_tarok(ctx: MoveContext) -> list[Card]:
    return [c for c in ctx.hand if c.card_type == CardType.TAROK]


@move_filter("follow_tarok_overplay")
def _follow_tarok_overplay(ctx: MoveContext) -> list[Card]:
    """Play taroks, but only ones that beat the current best."""
    taroks = [c for c in ctx.hand if c.card_type == CardType.TAROK]
    if ctx.best_card is not None:
        higher = [c for c in taroks if c.beats(ctx.best_card, None)]
        if higher:
            return higher
    return taroks


@move_filter("trump_in")
def _trump_in(ctx: MoveContext) -> list[Card]:
    """Can't follow suit — must play tarok."""
    return [c for c in ctx.hand if c.card_type == CardType.TAROK]


@move_filter("trump_in_overplay")
def _trump_in_overplay(ctx: MoveContext) -> list[Card]:
    """Can't follow suit — must play a tarok higher than the best tarok on the table."""
    taroks = [c for c in ctx.hand if c.card_type == CardType.TAROK]
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
    it if you have another tarok.
    """
    has_pagat = any(c.card_type == CardType.TAROK and c.value == PAGAT for c in cards)
    if not has_pagat:
        return cards
    non_pagat = [c for c in cards if not (c.card_type == CardType.TAROK and c.value == PAGAT)]
    if non_pagat:
        return non_pagat
    return cards  # Pagat is the only card — must play it
