"""BDD steps for deck.feature"""

import pytest
from pytest_bdd import scenarios, given, then

from tarok.entities import DECK, CardType, Suit, SuitRank, PAGAT, MOND, SKIS, compute_card_points

scenarios("features/deck.feature")


@given("a fresh tarok deck", target_fixture="deck")
def fresh_deck():
    return list(DECK)


@then("the deck should contain 54 cards")
def deck_has_54(deck):
    assert len(deck) == 54


@then("the deck should contain 22 tarok cards")
def deck_has_22_taroks(deck):
    taroks = [c for c in deck if c.card_type == CardType.TAROK]
    assert len(taroks) == 22


@then("the deck should contain 32 suit cards")
def deck_has_32_suits(deck):
    suits = [c for c in deck if c.card_type == CardType.SUIT]
    assert len(suits) == 32


@then("each suit should have 8 cards")
def each_suit_has_8(deck):
    for s in Suit:
        count = sum(1 for c in deck if c.suit == s)
        assert count == 8, f"{s} has {count} cards, expected 8"


@then("all cards should be unique")
def all_unique(deck):
    assert len(set(deck)) == len(deck)


@then("the total raw card points should equal 106")
def total_raw_106(deck):
    total = sum(c.points for c in deck)
    assert total == 106


@then("the total counted card points should equal 70")
def total_counted_70(deck):
    assert compute_card_points(deck) == 70


@then("Pagat should be worth 5 points")
def pagat_5(deck):
    pagat = next(c for c in deck if c.card_type == CardType.TAROK and c.value == PAGAT)
    assert pagat.points == 5


@then("Mond should be worth 5 points")
def mond_5(deck):
    mond = next(c for c in deck if c.card_type == CardType.TAROK and c.value == MOND)
    assert mond.points == 5


@then("Škis should be worth 5 points")
def skis_5(deck):
    skis = next(c for c in deck if c.card_type == CardType.TAROK and c.value == SKIS)
    assert skis.points == 5


@then("all kings should be worth 5 points")
def kings_5(deck):
    kings = [c for c in deck if c.card_type == CardType.SUIT and c.value == SuitRank.KING.value]
    assert len(kings) == 4
    for k in kings:
        assert k.points == 5


@then("all queens should be worth 4 points")
def queens_4(deck):
    queens = [c for c in deck if c.card_type == CardType.SUIT and c.value == SuitRank.QUEEN.value]
    assert len(queens) == 4
    for q in queens:
        assert q.points == 4


@then("all knights should be worth 3 points")
def knights_3(deck):
    knights = [c for c in deck if c.card_type == CardType.SUIT and c.value == SuitRank.KNIGHT.value]
    assert len(knights) == 4
    for k in knights:
        assert k.points == 3


@then("all jacks should be worth 2 points")
def jacks_2(deck):
    jacks = [c for c in deck if c.card_type == CardType.SUIT and c.value == SuitRank.JACK.value]
    assert len(jacks) == 4
    for j in jacks:
        assert j.points == 2


@then("all pip cards should be worth 1 point")
def pips_1(deck):
    pips = [c for c in deck if c.card_type == CardType.SUIT and c.value <= SuitRank.PIP_4.value]
    assert len(pips) == 16
    for p in pips:
        assert p.points == 1


@then("plain taroks should be worth 1 point")
def plain_taroks_1(deck):
    plain = [c for c in deck if c.card_type == CardType.TAROK and c.value not in (PAGAT, MOND, SKIS)]
    assert len(plain) == 19
    for t in plain:
        assert t.points == 1
