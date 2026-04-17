"""Smoke test — run a full game to verify the engine works end-to-end."""

import asyncio
import pytest

from tarok.use_cases.game_loop import RustGameLoop as GameLoop
from tarok.adapters.players.stockskis_player import StockskisPlayer


@pytest.mark.asyncio
async def test_full_game_with_random_players():
async def test_full_game():
    players = [StockskisPlayer(variant="m6", name=f"Bot-{i}") for i in range(4)]
    game = GameLoop(players)

    state, scores = await game.run()

    # Game should be finished
    assert state.phase.value == "finished"
    # All 12 tricks played
    assert state.tricks_played == 12
    # All hands empty
    for hand in state.hands:
        assert len(hand) == 0
    # Only declarer team should score; opponents get 0
    if state.declarer is not None and not state.contract.is_klop:
        for p in range(4):
            if state.get_team(p) != state.get_team(state.declarer):
                assert scores.get(p, 0) == 0, f"Opponent {p} has non-zero score: {scores}"
    # Every player should have a score
    assert len(scores) == 4


@pytest.mark.asyncio
async def test_multiple_games():
    """Run 10 games to check stability."""
    for seed in range(10):
        players = [StockskisPlayer(variant="m6", name=f"Bot-{i}") for i in range(4)]
        game = GameLoop(players)
        state, scores = await game.run()

        assert state.phase.value == "finished"
        if state.contract and not state.contract.is_berac:
            assert state.tricks_played == 12
        # Only declarer team should score, opponents get 0
        if state.contract and not state.contract.is_klop:
            for p in range(4):
                team = state.get_team(p)
                if team != state.get_team(state.declarer):
                    assert scores.get(p, 0) == 0, f"Game {seed}: opponent {p} scored {scores}"
