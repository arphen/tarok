from __future__ import annotations

import asyncio

from tarok.adapters.ai.stockskis_v4 import StockSkisPlayerV4, _evaluate_hand_for_bid
from tarok.entities.card import Suit, SuitRank, tarok, suit_card
from tarok.entities.game_state import Contract, GameState, Phase, PlayerRole, Trick


class TestStockSkisV4BidEvaluation:
    def test_berac_requires_at_most_two_taroks(self):
        hand = [
            tarok(2),
            tarok(3),
            tarok(4),
            suit_card(Suit.HEARTS, SuitRank.PIP_1),
            suit_card(Suit.HEARTS, SuitRank.PIP_2),
            suit_card(Suit.HEARTS, SuitRank.PIP_3),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_1),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_2),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_3),
            suit_card(Suit.CLUBS, SuitRank.PIP_1),
            suit_card(Suit.CLUBS, SuitRank.PIP_2),
            suit_card(Suit.SPADES, SuitRank.PIP_1),
        ]

        assert 7 not in _evaluate_hand_for_bid(hand)

    def test_berac_rejects_singleton_suit(self):
        hand = [
            tarok(2),
            tarok(3),
            suit_card(Suit.HEARTS, SuitRank.PIP_1),
            suit_card(Suit.HEARTS, SuitRank.PIP_2),
            suit_card(Suit.HEARTS, SuitRank.PIP_3),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_1),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_2),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_3),
            suit_card(Suit.CLUBS, SuitRank.PIP_1),
            suit_card(Suit.CLUBS, SuitRank.PIP_2),
            suit_card(Suit.CLUBS, SuitRank.PIP_3),
            suit_card(Suit.SPADES, SuitRank.PIP_1),
        ]

        assert 7 not in _evaluate_hand_for_bid(hand)


class TestStockSkisV4Leading:
    def test_partner_prefers_highest_tarok_lead(self):
        player = StockSkisPlayerV4(seed=7)
        hand = [
            tarok(20),
            tarok(15),
            suit_card(Suit.HEARTS, SuitRank.PIP_1),
            suit_card(Suit.CLUBS, SuitRank.PIP_2),
        ]
        state = GameState(
            phase=Phase.TRICK_PLAY,
            hands=[[], hand, [], []],
            contract=Contract.THREE,
            declarer=0,
            partner=1,
            roles={0: PlayerRole.DECLARER, 1: PlayerRole.PARTNER, 2: PlayerRole.OPPONENT, 3: PlayerRole.OPPONENT},
            current_player=1,
        )

        chosen = asyncio.run(player.choose_card(state, 1, hand))

        assert chosen == tarok(20)

    def test_opposition_preserves_tarok_and_leads_low_suit(self):
        player = StockSkisPlayerV4(seed=9)
        hand = [
            tarok(20),
            tarok(10),
            suit_card(Suit.HEARTS, SuitRank.PIP_1),
            suit_card(Suit.SPADES, SuitRank.KING),
        ]
        prior_trick = Trick(
            lead_player=3,
            cards=[
                (3, suit_card(Suit.HEARTS, SuitRank.PIP_2)),
                (0, suit_card(Suit.HEARTS, SuitRank.PIP_3)),
                (1, tarok(5)),
                (2, suit_card(Suit.HEARTS, SuitRank.PIP_1)),
            ],
        )
        state = GameState(
            phase=Phase.TRICK_PLAY,
            hands=[[], [], hand, []],
            contract=Contract.THREE,
            declarer=0,
            partner=1,
            roles={0: PlayerRole.DECLARER, 1: PlayerRole.PARTNER, 2: PlayerRole.OPPONENT, 3: PlayerRole.OPPONENT},
            tricks=[prior_trick],
            current_player=2,
        )

        chosen = asyncio.run(player.choose_card(state, 2, hand))

        assert chosen == suit_card(Suit.HEARTS, SuitRank.PIP_1)