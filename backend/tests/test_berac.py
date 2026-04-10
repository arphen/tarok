"""Tests for Berač early termination — declarer must take 0 tricks.

The game should end immediately when the Berač declarer wins any trick,
awarding -70 to the declarer. If the declarer survives all 12 tricks
without winning one, they get +70.
"""

import asyncio
import random

import pytest

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.random_agent import RandomPlayer
from tarok.entities.card import Card, CardType, Suit, SuitRank, DECK
from tarok.entities.game_state import (
    Bid, Contract, GameState, Phase, PlayerRole, Trick,
)
from tarok.entities.scoring import score_game
from tarok.use_cases.game_loop import GameLoop, NullObserver
from tarok.use_cases.deal import deal
from tarok.use_cases.play_trick import start_trick, play_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ScriptedPlayer:
    """Player that returns pre-scripted actions or auto-plays first legal."""

    def __init__(self, name: str, bid=None):
        self.name = name
        self._bid = bid

    async def choose_bid(self, state, player_idx, legal_bids):
        if self._bid is not None and self._bid in legal_bids:
            return self._bid
        return None

    async def choose_king(self, state, player_idx, callable_kings):
        return callable_kings[0]

    async def choose_talon_group(self, state, player_idx, talon_groups):
        return 0

    async def choose_discard(self, state, player_idx, must_discard):
        hand = state.hands[player_idx]
        d = [c for c in hand if not c.is_king and c.card_type != CardType.TAROK]
        d.sort(key=lambda c: c.points)
        return d[:must_discard]

    async def choose_card(self, state, player_idx, legal_plays):
        return legal_plays[0]

    async def choose_announcements(self, state, player_idx):
        return []


class TrackingObserver(NullObserver):
    """Records events to verify early termination."""

    def __init__(self):
        self.tricks_completed = 0
        self.game_ended = False
        self.final_scores = None

    async def on_trick_won(self, trick, winner, state):
        self.tricks_completed += 1

    async def on_game_end(self, scores, state):
        self.game_ended = True
        self.final_scores = scores


# ---------------------------------------------------------------------------
# Unit tests — scoring
# ---------------------------------------------------------------------------

def test_berac_scoring_declarer_wins_trick():
    """If the declarer took any trick, they get -70."""
    state = GameState()
    state.contract = Contract.BERAC
    state.declarer = 0
    state.num_players = 4

    # Simulate: declarer won 1 trick
    trick = Trick(lead_player=1)
    trick.cards = [
        (1, Card(CardType.SUIT, 1, Suit.HEARTS)),
        (2, Card(CardType.SUIT, 2, Suit.HEARTS)),
        (3, Card(CardType.SUIT, 3, Suit.HEARTS)),
        (0, Card(CardType.SUIT, SuitRank.KING.value, Suit.HEARTS)),  # Declarer wins
    ]
    state.tricks = [trick]

    scores = score_game(state)
    assert scores[0] == -70, f"Declarer should lose 70, got {scores[0]}"
    assert scores[1] == 0
    assert scores[2] == 0
    assert scores[3] == 0


def test_berac_scoring_declarer_takes_zero_tricks():
    """If the declarer took 0 tricks across all 12, they get +70."""
    state = GameState()
    state.contract = Contract.BERAC
    state.declarer = 0
    state.num_players = 4

    # 12 tricks all won by player 1 (not the declarer)
    for _ in range(12):
        trick = Trick(lead_player=1)
        trick.cards = [
            (1, Card(CardType.TAROK, 21, None)),  # Mond (highest tarok)
            (2, Card(CardType.TAROK, 1, None)),
            (3, Card(CardType.TAROK, 2, None)),
            (0, Card(CardType.TAROK, 3, None)),
        ]
        state.tricks.append(trick)

    scores = score_game(state)
    assert scores[0] == 70, f"Declarer should win 70, got {scores[0]}"


# ---------------------------------------------------------------------------
# Integration tests — game loop early termination
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_berac_ends_early_when_declarer_wins_trick():
    """Game should stop before 12 tricks if Berač declarer wins a trick."""
    # Force a Berač game where declarer (player 0) bids Berač
    # and is likely to win a trick due to high cards.
    players = [
        ScriptedPlayer("Declarer", bid=Contract.BERAC),
        ScriptedPlayer("Opp-1"),
        ScriptedPlayer("Opp-2"),
        ScriptedPlayer("Opp-3"),
    ]

    observer = TrackingObserver()
    won_early = False

    # Run multiple games with different seeds until we get one
    # where the declarer wins a trick (very likely with random cards).
    for seed in range(50):
        observer = TrackingObserver()
        loop = GameLoop(players, observer=observer, rng=random.Random(seed))
        state, scores = await loop.run(dealer=3)  # player 0 is forehand, bids first

        if state.contract and state.contract.is_berac and state.declarer == 0:
            if state.tricks_played < 12:
                won_early = True
                # Verify: the last trick was won by the declarer
                last_trick = state.tricks[-1]
                assert last_trick.winner() == 0, "Last trick should be won by declarer"
                assert scores[0] == -70, "Declarer should get -70"
                assert observer.game_ended
                assert observer.tricks_completed < 12
                break

    assert won_early, (
        "Could not find a seed where the Berač declarer wins early — "
        "this is statistically almost impossible over 50 seeds"
    )


@pytest.mark.asyncio
async def test_berac_plays_all_12_tricks_if_declarer_never_wins():
    """If the declarer never wins a trick, all 12 tricks play out and they get +70."""
    # Run many games; find one where declarer wins Berač by taking 0 tricks
    players = [
        ScriptedPlayer("Declarer", bid=Contract.BERAC),
        ScriptedPlayer("Opp-1"),
        ScriptedPlayer("Opp-2"),
        ScriptedPlayer("Opp-3"),
    ]

    found_full_game = False
    for seed in range(200):
        observer = TrackingObserver()
        loop = GameLoop(players, observer=observer, rng=random.Random(seed))
        state, scores = await loop.run(dealer=3)

        if (
            state.contract
            and state.contract.is_berac
            and state.declarer == 0
            and state.tricks_played == 12
        ):
            found_full_game = True
            assert scores[0] == 70, f"Declarer should win 70, got {scores[0]}"
            assert observer.tricks_completed == 12
            break

    assert found_full_game, "Could not find a seed where Berač declarer survives 12 tricks"


@pytest.mark.asyncio
async def test_berac_early_termination_correct_phase():
    """Phase should be 'finished' after early Berač termination."""
    players = [
        ScriptedPlayer("Declarer", bid=Contract.BERAC),
        ScriptedPlayer("Opp-1"),
        ScriptedPlayer("Opp-2"),
        ScriptedPlayer("Opp-3"),
    ]

    for seed in range(50):
        loop = GameLoop(players, rng=random.Random(seed))
        state, scores = await loop.run(dealer=3)

        assert state.phase == Phase.FINISHED
        assert len(scores) == 4


@pytest.mark.asyncio
async def test_berac_remaining_cards_stay_in_hands():
    """When Berač ends early, remaining cards should still be in players' hands."""
    players = [
        ScriptedPlayer("Declarer", bid=Contract.BERAC),
        ScriptedPlayer("Opp-1"),
        ScriptedPlayer("Opp-2"),
        ScriptedPlayer("Opp-3"),
    ]

    for seed in range(50):
        loop = GameLoop(players, rng=random.Random(seed))
        state, scores = await loop.run(dealer=3)

        if (
            state.contract
            and state.contract.is_berac
            and state.declarer == 0
            and state.tricks_played < 12
        ):
            # Cards played = 4 per trick
            cards_played = state.tricks_played * 4
            # Cards remaining in hands
            cards_in_hands = sum(len(h) for h in state.hands)
            assert cards_played + cards_in_hands == 48, (
                f"Cards don't add up: {cards_played} played + "
                f"{cards_in_hands} in hands != 48"
            )
            break


@pytest.mark.asyncio
async def test_non_berac_games_unaffected():
    """Regular (non-Berač) games should still play all 12 tricks."""
    for seed in range(10):
        players = [RandomPlayer(name=f"Bot-{i}", rng=random.Random(seed * 10 + i)) for i in range(4)]
        loop = GameLoop(players, rng=random.Random(seed))
        state, scores = await loop.run()

        assert state.phase == Phase.FINISHED
        if state.contract and not state.contract.is_berac:
            assert state.tricks_played == 12


@pytest.mark.asyncio
async def test_berac_with_rl_agents_stable():
    """Berač early termination works with RL agents without crashes."""
    for seed in range(10):
        human = ScriptedPlayer("You", bid=Contract.BERAC)
        agents = [human]
        for i in range(3):
            a = RLAgent(name=f"AI-{i+1}")
            a.set_training(False)
            agents.append(a)

        loop = GameLoop(agents, rng=random.Random(seed))
        state, scores = await loop.run(dealer=3)

        assert state.phase == Phase.FINISHED
        if state.contract and state.contract.is_berac:
            assert scores[state.declarer] in (-70, 70)
