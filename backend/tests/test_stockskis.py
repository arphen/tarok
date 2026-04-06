"""Integration tests for StockŠkis heuristic player."""

from __future__ import annotations

import asyncio

import pytest

from tarok.adapters.ai.stockskis_player import (
    StockSkisPlayer,
    _evaluate_hand_for_bid,
    _worth_over,
)
from tarok.entities.card import Card, CardType, Suit, SuitRank, tarok, suit_card, DECK
from tarok.entities.game_state import Contract, GameState
from tarok.use_cases.game_loop import GameLoop


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestWorthOver:
    def test_pagat(self):
        assert _worth_over(tarok(1)) == 11

    def test_mond(self):
        assert _worth_over(tarok(21)) == 31

    def test_skis(self):
        assert _worth_over(tarok(22)) == 32

    def test_suit_card(self):
        king_hearts = suit_card(Suit.HEARTS, SuitRank.KING)
        assert _worth_over(king_hearts) == 33 + 8


class TestBidEvaluation:
    def test_weak_hand_bids_conservatively(self):
        # Hand of only low pip cards, no taroks → should not bid solo or higher
        hand = [
            suit_card(Suit.HEARTS, SuitRank.PIP_1),
            suit_card(Suit.HEARTS, SuitRank.PIP_2),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_1),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_2),
            suit_card(Suit.CLUBS, SuitRank.PIP_1),
            suit_card(Suit.CLUBS, SuitRank.PIP_2),
            suit_card(Suit.SPADES, SuitRank.PIP_1),
            suit_card(Suit.SPADES, SuitRank.PIP_2),
            suit_card(Suit.HEARTS, SuitRank.PIP_3),
            suit_card(Suit.DIAMONDS, SuitRank.PIP_3),
            suit_card(Suit.CLUBS, SuitRank.PIP_3),
            suit_card(Suit.SPADES, SuitRank.PIP_3),
        ]
        modes = _evaluate_hand_for_bid(hand)
        # Should not bid solo (mode>=6) or berac with this hand
        assert all(m < 6 for m in modes if m >= 0)

    def test_strong_hand_bids(self):
        # Very strong hand: high taroks + kings
        hand = [
            tarok(22),  # Škis
            tarok(21),  # Mond
            tarok(20),
            tarok(19),
            tarok(18),
            tarok(17),
            tarok(16),
            tarok(15),
            suit_card(Suit.HEARTS, SuitRank.KING),
            suit_card(Suit.DIAMONDS, SuitRank.KING),
            suit_card(Suit.CLUBS, SuitRank.KING),
            suit_card(Suit.SPADES, SuitRank.KING),
        ]
        modes = _evaluate_hand_for_bid(hand)
        # Should want to bid something (not just pass)
        assert any(m >= 0 for m in modes)


# ---------------------------------------------------------------------------
# Full game integration tests
# ---------------------------------------------------------------------------


class TestStockSkisFullGame:
    """Test that 4 StockŠkis players can complete a full game without errors."""

    def test_four_stockskis_complete_game(self):
        players = [StockSkisPlayer(name=f"Bot-{i}", seed=42 + i) for i in range(4)]
        game = GameLoop(players)

        state, scores = asyncio.run(game.run(dealer=0))

        # Game should reach finished state
        assert state.phase.value == "finished"
        # All 4 players should have scores
        assert len(scores) == 4
        # Only declarer team should score
        assert all(scores[p] == 0 for p in range(4) if state.get_team(p) != state.get_team(state.declarer)) or state.contract.is_klop

    def test_multiple_games_consistent(self):
        """Play 10 games to ensure consistency across different deals."""
        players = [StockSkisPlayer(name=f"Bot-{i}") for i in range(4)]

        for dealer in range(10):
            game = GameLoop(players)
            state, scores = asyncio.run(game.run(dealer=dealer % 4))
            assert state.phase.value == "finished"
            assert len(scores) == 4


class TestStockSkisVsRLAgent:
    """Test that StockŠkis players can play against RL agents."""

    def test_mixed_game(self):
        from tarok.adapters.ai.agent import RLAgent

        rl_agent = RLAgent(name="RL-0", hidden_size=64, device="cpu")
        rl_agent.set_training(False)
        bots = [StockSkisPlayer(name=f"Bot-{i}", seed=100 + i) for i in range(3)]

        players = [rl_agent] + bots
        game = GameLoop(players)

        state, scores = asyncio.run(game.run(dealer=0))
        assert state.phase.value == "finished"
        assert len(scores) == 4


class TestTrainerStockSkisIntegration:
    """Test the trainer with StockŠkis opponents."""

    def test_trainer_with_stockskis_ratio(self):
        from tarok.adapters.ai.agent import RLAgent
        from tarok.adapters.ai.trainer import PPOTrainer

        agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
        trainer = PPOTrainer(
            agents,
            games_per_session=4,
            stockskis_ratio=0.5,
            stockskis_strength=0.8,
        )

        # Train 1 session (4 games, ~50% against StockŠkis)
        metrics = asyncio.run(trainer.train(num_sessions=1))
        assert metrics.episode == 4
