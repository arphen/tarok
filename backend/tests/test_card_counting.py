"""Unit tests for card counting edge cases in compute_card_points."""

from tarok.entities import (
    Card,
    Suit,
    SuitRank,
    tarok,
    suit_card,
    PAGAT,
    MOND,
    SKIS,
    DECK,
    compute_card_points,
    TOTAL_GAME_POINTS,
)


def _filler(n: int) -> list[Card]:
    """Return n 1-point tarok filler cards (values 2..20)."""
    return [tarok(v) for v in range(2, 2 + n)]


class TestComputeCardPointsGroups:
    """Verify the groups-of-3 counting rule: sum-2 per full group."""

    def test_empty(self):
        assert compute_card_points([]) == 0

    def test_single_1pt_card(self):
        # Leftover 1: point value as-is (1)
        assert compute_card_points([tarok(2)]) == 1

    def test_single_5pt_card(self):
        # Leftover 1: point value as-is (5)
        assert compute_card_points([tarok(PAGAT)]) == 5

    def test_two_1pt_cards(self):
        # Leftover 2: sum(1+1) - 1 = 1
        assert compute_card_points([tarok(2), tarok(3)]) == 1

    def test_two_5pt_cards(self):
        # Leftover 2: sum(5+5) - 1 = 9
        assert compute_card_points([tarok(PAGAT), tarok(MOND)]) == 9

    def test_full_group_three_1pt(self):
        # Full group: sum(1+1+1) - 2 = 1
        assert compute_card_points(_filler(3)) == 1

    def test_full_group_three_5pt(self):
        # Full group: sum(5+5+5) - 2 = 13
        cards = [tarok(PAGAT), tarok(MOND), tarok(SKIS)]
        assert compute_card_points(cards) == 13

    def test_full_group_mixed(self):
        # King (5) + filler (1) + filler (1) → 7 - 2 = 5
        cards = [suit_card(Suit.HEARTS, SuitRank.KING), tarok(2), tarok(3)]
        assert compute_card_points(cards) == 5

    def test_six_cards_two_groups(self):
        # Group 1: King(5) + Queen(4) + Knight(3) = 12 - 2 = 10
        # Group 2: Jack(2) + filler(1) + filler(1) = 4 - 2 = 2
        # Total: 12
        cards = [
            suit_card(Suit.HEARTS, SuitRank.KING),
            suit_card(Suit.HEARTS, SuitRank.QUEEN),
            suit_card(Suit.HEARTS, SuitRank.KNIGHT),
            suit_card(Suit.HEARTS, SuitRank.JACK),
            tarok(2),
            tarok(3),
        ]
        assert compute_card_points(cards) == 12

    def test_four_cards_group_plus_leftover(self):
        # Group: 5+5+5 - 2 = 13; leftover 1: 1 as-is
        cards = [tarok(PAGAT), tarok(MOND), tarok(SKIS), tarok(2)]
        assert compute_card_points(cards) == 14

    def test_five_cards_group_plus_leftover_2(self):
        # Group: 5+5+5 - 2 = 13; leftover 2: 5+1 - 1 = 5
        cards = [
            tarok(PAGAT),
            tarok(MOND),
            tarok(SKIS),
            suit_card(Suit.HEARTS, SuitRank.KING),
            tarok(2),
        ]
        assert compute_card_points(cards) == 18

    def test_full_deck_is_70(self):
        assert compute_card_points(list(DECK)) == TOTAL_GAME_POINTS

    def test_full_deck_shuffled_is_70(self):
        import random

        deck = list(DECK)
        random.shuffle(deck)
        assert compute_card_points(deck) == TOTAL_GAME_POINTS

    def test_formula_consistency(self):
        """For N cards, deduction = (N//3)*2 + (1 if N%3==2 else 0)."""
        for n in range(0, 55):
            cards = _filler(min(n, 19))
            # Pad with suit cards if we need more than 19 filler taroks
            while len(cards) < n:
                cards.append(suit_card(Suit.HEARTS, SuitRank.KING))  # 5pt
            cards = cards[:n]
            raw = sum(c.points for c in cards)
            expected_deduction = (n // 3) * 2 + (1 if n % 3 == 2 else 0)
            assert compute_card_points(cards) == raw - expected_deduction, f"Failed for n={n}"

    def test_split_deck_sums_to_70(self):
        """Two halves of deck counted separately should sum to 70."""
        deck = list(DECK)
        mid = 27  # 27 cards each → 9 full groups
        part1 = deck[:mid]
        part2 = deck[mid:]
        assert compute_card_points(part1) + compute_card_points(part2) == TOTAL_GAME_POINTS
