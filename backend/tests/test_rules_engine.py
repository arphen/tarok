"""YAML-driven BDD test runner for the rules engine.

Ingests test_scenarios.yaml and generates parametrized pytest cases
for both the legal-move and trick-evaluation engines.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tarok.entities.card import (
    Card,
    CardType,
    Suit,
    SuitRank,
    tarok,
    suit_card,
)
from tarok.engine.conditions import MoveContext, TrickContext
from tarok.engine.legal_moves import generate_legal_moves
from tarok.engine.trick_eval import evaluate_trick

_SCENARIOS_PATH = Path(__file__).parent.parent / "src" / "tarok" / "engine" / "rules" / "test_scenarios.yaml"

# ---------------------------------------------------------------------------
# Card shorthand parser
# ---------------------------------------------------------------------------

_SUIT_MAP = {"H": Suit.HEARTS, "D": Suit.DIAMONDS, "C": Suit.CLUBS, "S": Suit.SPADES}
_RANK_MAP = {
    "1": SuitRank.PIP_1, "2": SuitRank.PIP_2, "3": SuitRank.PIP_3, "4": SuitRank.PIP_4,
    "J": SuitRank.JACK, "C": SuitRank.KNIGHT, "Q": SuitRank.QUEEN, "K": SuitRank.KING,
}

_TAROK_RE = re.compile(r"^T(\d+)$")
_SUIT_RE = re.compile(r"^([1-4JCQK])([HDCS])$")


def parse_card(shorthand: str) -> Card:
    """Parse a card shorthand like 'T15', 'KH', '1D' into a Card."""
    m = _TAROK_RE.match(shorthand)
    if m:
        return tarok(int(m.group(1)))

    m = _SUIT_RE.match(shorthand)
    if m:
        rank_str, suit_str = m.group(1), m.group(2)
        # Disambiguate 'C' as rank (Knight/Cavalier) vs suit (Clubs)
        rank = _RANK_MAP[rank_str]
        suit = _SUIT_MAP[suit_str]
        return suit_card(suit, rank)

    raise ValueError(f"Cannot parse card shorthand: {shorthand!r}")


_CONTRACT_MAP = {
    "three": "three",
    "two": "two",
    "one": "one",
    "solo_three": "solo_three",
    "solo_two": "solo_two",
    "solo_one": "solo_one",
    "solo": "solo",
    "klop": "klop",
    "berac": "berac",
}


# ---------------------------------------------------------------------------
# Load scenarios
# ---------------------------------------------------------------------------

def _load_scenarios():
    with open(_SCENARIOS_PATH) as f:
        return yaml.safe_load(f)


_data = _load_scenarios()


# ---------------------------------------------------------------------------
# Legal move tests
# ---------------------------------------------------------------------------

def _legal_move_ids():
    return [s["name"] for s in _data["legal_moves"]]


def _legal_move_scenarios():
    return _data["legal_moves"]


@pytest.mark.parametrize("scenario", _legal_move_scenarios(), ids=_legal_move_ids())
def test_legal_moves(scenario):
    given = scenario["given"]
    expected = scenario["expected"]

    hand = tuple(parse_card(c) for c in given["hand"])
    lead_card = parse_card(given["lead_card"]) if given.get("lead_card") else None
    best_card = parse_card(given["best_card"]) if given.get("best_card") else None
    contract = _CONTRACT_MAP.get(given.get("contract", ""), None)

    # Derive lead_suit
    lead_suit = None
    if lead_card and lead_card.card_type != CardType.TAROK:
        lead_suit = lead_card.suit

    ctx = MoveContext(
        hand=hand,
        lead_card=lead_card,
        lead_suit=lead_suit,
        best_card=best_card,
        contract_name=contract,
        is_first_trick=given.get("is_first_trick", False),
        is_last_trick=given.get("is_last_trick", False),
        trick_card_count=given.get("trick_card_count", 0),
    )

    trace = generate_legal_moves(ctx)

    # Assert cards match (order-independent)
    expected_cards = {parse_card(c) for c in expected["cards"]}
    actual_cards = set(trace.result)
    assert actual_cards == expected_cards, (
        f"Expected {expected_cards}, got {actual_cards}"
    )

    # Assert triggered rule
    assert trace.triggered_rule == expected["triggered_rule"], (
        f"Expected rule {expected['triggered_rule']!r}, got {trace.triggered_rule!r}"
    )

    # Assert ban rules if specified
    if "applied_bans" in expected:
        assert trace.context.get("applied_bans", []) == expected["applied_bans"]


# ---------------------------------------------------------------------------
# Trick evaluation tests
# ---------------------------------------------------------------------------

def _trick_eval_ids():
    return [s["name"] for s in _data["trick_eval"]]


def _trick_eval_scenarios():
    return _data["trick_eval"]


@pytest.mark.parametrize("scenario", _trick_eval_scenarios(), ids=_trick_eval_ids())
def test_trick_eval(scenario):
    given = scenario["given"]
    expected = scenario["expected"]

    cards = tuple(
        (entry[0], parse_card(entry[1]))
        for entry in given["cards"]
    )

    # Derive lead suit from first card
    _, lead_card = cards[0]
    lead_suit = lead_card.suit if lead_card.card_type != CardType.TAROK else None

    ctx = TrickContext(
        cards=cards,
        lead_suit=lead_suit,
        is_last_trick=given.get("is_last_trick", False),
        contract_name=given.get("contract", None),
    )

    trace = evaluate_trick(ctx)

    winner, _points = trace.result
    assert winner == expected["winner"], (
        f"Expected winner {expected['winner']}, got {winner}"
    )

    assert trace.triggered_rule == expected["triggered_rule"], (
        f"Expected rule {expected['triggered_rule']!r}, got {trace.triggered_rule!r}"
    )
