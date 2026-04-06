"""Smoke test — run a full game with random players to verify the engine works."""

import asyncio
import pytest
from tarok.adapters.ai.random_agent import RandomPlayer
from tarok.use_cases.game_loop import GameLoop


@pytest.mark.asyncio
async def test_full_game_with_random_players():
    players = [RandomPlayer(name=f"Bot-{i}", rng=__import__("random").Random(42 + i)) for i in range(4)]
    game = GameLoop(players, rng=__import__("random").Random(123))

    state, scores = await game.run()

    # Game should be finished
    assert state.phase.value == "finished"
    # All 12 tricks played
    assert state.tricks_played == 12
    # All hands empty
    for hand in state.hands:
        assert len(hand) == 0
    # Scores should balance (zero-sum for 2v2)
    assert sum(scores.values()) == 0, f"Scores don't balance: {scores}"
    # Every player should have a score
    assert len(scores) == 4


@pytest.mark.asyncio
async def test_multiple_games():
    """Run 10 games to check stability."""
    for seed in range(10):
        players = [RandomPlayer(name=f"Bot-{i}", rng=__import__("random").Random(seed * 10 + i)) for i in range(4)]
        game = GameLoop(players, rng=__import__("random").Random(seed))
        state, scores = await game.run()

        assert state.phase.value == "finished"
        assert state.tricks_played == 12
        # Klop and berac are not zero-sum; other contracts are
        if state.contract and not state.contract.is_klop:
            assert sum(scores.values()) == 0, f"Game {seed}: scores={scores}"
