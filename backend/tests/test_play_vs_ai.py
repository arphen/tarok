"""Tests for the Play vs AI flow — verifies the WebSocket game loop works
end-to-end with a simulated human player, covering:
  - Bidding (including illegal bid graceful fallback)
  - King calling (backend sends callable_kings correctly)
  - Talon exchange + discard (must_discard field)
  - Trick play to completion
  - _state_for_player sends correct phase-specific data
"""

import asyncio
import random

import pytest

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.random_agent import RandomPlayer
from tarok.adapters.api.human_player import HumanPlayer
from tarok.adapters.api.ws_observer import _state_for_player
from tarok.entities.card import Card, CardType, Suit, SuitRank
from tarok.entities.game_state import Bid, Contract, GameState, Phase
from tarok.use_cases.game_loop import GameLoop, NullObserver


NAMES = ["You", "AI-1", "AI-2", "AI-3"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AutoHuman:
    """Simulates a human that auto-responds to every game phase."""

    def __init__(self, bid_choice=None):
        self.name = "You"
        self._bid_choice = bid_choice  # None = always pass

    async def choose_bid(self, state, player_idx, legal_bids):
        if self._bid_choice is not None:
            for b in legal_bids:
                if b is not None and b.value == self._bid_choice:
                    return b
        return None

    async def choose_king(self, state, player_idx, callable_kings):
        return callable_kings[0]

    async def choose_talon_group(self, state, player_idx, talon_groups):
        return 0

    async def choose_discard(self, state, player_idx, must_discard):
        hand = state.hands[player_idx]
        discardable = [
            c for c in hand if not c.is_king and c.card_type != CardType.TAROK
        ]
        discardable.sort(key=lambda c: c.points)
        return discardable[:must_discard]

    async def choose_card(self, state, player_idx, legal_plays):
        return legal_plays[0]

    async def choose_announcements(self, state, player_idx):
        return []


class StateTrackingObserver(NullObserver):
    """Records _state_for_player output at each event to detect serialization errors."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.errors: list[str] = []

    async def _record(self, event: str, state: GameState):
        try:
            sd = _state_for_player(state, 0, NAMES)
            self.events.append((event, sd))
        except Exception as e:
            self.errors.append(f"{event}: {e}")

    async def on_game_start(self, state):
        await self._record("game_start", state)

    async def on_deal(self, state):
        await self._record("deal", state)

    async def on_bid(self, player, bid, state):
        await self._record("bid", state)

    async def on_contract_won(self, player, contract, state):
        await self._record("contract_won", state)

    async def on_king_called(self, player, king, state):
        await self._record("king_called", state)

    async def on_talon_revealed(self, groups, state):
        await self._record("talon_revealed", state)

    async def on_talon_exchanged(self, state):
        await self._record("talon_exchanged", state)

    async def on_card_played(self, player, card, state):
        await self._record("card_played", state)

    async def on_trick_won(self, trick, winner, state):
        await self._record("trick_won", state)

    async def on_game_end(self, scores, state):
        await self._record("game_end", state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_game_with_rl_agents_completes():
    """A full game with RLAgent (untrained) + auto-human completes without error."""
    human = AutoHuman()
    agents = [human]
    for i in range(3):
        a = RLAgent(name=f"AI-{i+1}")
        a.set_training(False)
        agents.append(a)

    observer = StateTrackingObserver()
    loop = GameLoop(agents, observer=observer)
    state, scores = await loop.run(dealer=0)

    assert state.phase == Phase.FINISHED
    if state.contract and state.contract.is_berac:
        assert 1 <= state.tricks_played <= 12
    else:
        assert state.tricks_played == 12
    assert len(scores) == 4
    assert not observer.errors, f"Observer errors: {observer.errors}"


@pytest.mark.asyncio
async def test_illegal_bid_falls_back_to_pass():
    """Sending an illegal bid should not crash — it falls back to pass."""

    class AlwaysBidsThree:
        name = "BadHuman"

        async def choose_bid(self, state, player_idx, legal_bids):
            return Contract.THREE  # may be illegal

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

    for seed in range(5):
        agents = [AlwaysBidsThree()]
        for i in range(3):
            a = RLAgent(name=f"AI-{i+1}")
            a.set_training(False)
            agents.append(a)

        loop = GameLoop(agents, observer=NullObserver(), rng=random.Random(seed))
        state, scores = await loop.run(dealer=seed % 4)
        assert state.phase == Phase.FINISHED


@pytest.mark.asyncio
async def test_legal_bids_sent_during_bidding():
    """_state_for_player should include legal_bids when it's the player's turn to bid."""
    human = AutoHuman()
    agents = [human]
    for i in range(3):
        a = RLAgent(name=f"AI-{i+1}")
        a.set_training(False)
        agents.append(a)

    observer = StateTrackingObserver()
    loop = GameLoop(agents, observer=observer)
    await loop.run(dealer=0)

    # Find events during bidding where it was player 0's turn
    bidding_states = [
        sd for event, sd in observer.events
        if sd["phase"] == "bidding" and sd["current_player"] == 0
    ]

    for sd in bidding_states:
        assert sd["legal_bids"] is not None, "legal_bids should be set when it's our turn"
        assert isinstance(sd["legal_bids"], list)
        assert None in sd["legal_bids"], "Pass should always be a legal bid"


@pytest.mark.asyncio
async def test_legal_bids_absent_when_not_our_turn():
    """legal_bids should be null when it's not the player's turn."""
    human = AutoHuman()
    agents = [human]
    for i in range(3):
        a = RLAgent(name=f"AI-{i+1}")
        a.set_training(False)
        agents.append(a)

    observer = StateTrackingObserver()
    loop = GameLoop(agents, observer=observer)
    await loop.run(dealer=0)

    # Find bidding events where it was NOT player 0's turn
    other_turn = [
        sd for event, sd in observer.events
        if sd["phase"] == "bidding" and sd["current_player"] != 0
    ]

    for sd in other_turn:
        assert sd["legal_bids"] is None, "legal_bids should be null when not our turn"


@pytest.mark.asyncio
async def test_legal_bids_match_backend_legality():
    """The legal_bids sent to the frontend should match what the backend considers legal."""
    # Build a state where BERAC has been bid
    state = GameState(dealer=0)
    state.phase = Phase.BIDDING
    state.hands = [[] for _ in range(4)]  # dummy
    state.bids = [
        Bid(player=1, contract=Contract.BERAC),
    ]
    state.current_player = 0
    state.current_bidder = 0

    sd = _state_for_player(state, 0, NAMES)
    legal = sd["legal_bids"]

    # After BERAC (strength 8), only pass and BARVNI_VALAT (strength 9) should be legal
    assert legal is not None
    assert None in legal  # pass
    # THREE, TWO, ONE, SOLO should NOT be in legal
    assert 3 not in legal, "THREE should not be legal after BERAC"
    assert 2 not in legal, "TWO should not be legal after BERAC"
    assert 1 not in legal, "ONE should not be legal after BERAC"
    assert 0 not in legal, "SOLO should not be legal after BERAC"


@pytest.mark.asyncio
async def test_callable_kings_sent_during_king_calling():
    """_state_for_player should send callable_kings (not legal card plays) during king_calling."""
    state = GameState(dealer=0)
    state.phase = Phase.KING_CALLING
    state.declarer = 0
    state.current_player = 0
    state.contract = Contract.THREE
    # Give player 0 a hand without the hearts king
    state.hands = [
        [Card(CardType.TAROK, v, None) for v in range(1, 13)],
        [], [], [],
    ]

    sd = _state_for_player(state, 0, NAMES)

    assert sd["callable_kings"] is not None
    assert isinstance(sd["callable_kings"], list)
    assert len(sd["callable_kings"]) > 0
    # Each callable king should have the correct structure
    for k in sd["callable_kings"]:
        assert k["card_type"] == "suit"
        assert k["suit"] is not None
        assert k["value"] == SuitRank.KING.value


@pytest.mark.asyncio
async def test_must_discard_set_after_talon_pickup():
    """must_discard should be > 0 after picking up talon cards."""
    state = GameState(dealer=0)
    state.phase = Phase.TALON_EXCHANGE
    state.declarer = 0
    state.current_player = 0
    state.contract = Contract.THREE  # pick 3 cards

    # Player 0 has 15 cards (12 original + 3 picked from talon)
    state.hands = [
        [Card(CardType.TAROK, v, None) for v in range(1, 13)]
        + [Card(CardType.SUIT, 1, Suit.HEARTS), Card(CardType.SUIT, 2, Suit.HEARTS), Card(CardType.SUIT, 3, Suit.HEARTS)],
        [], [], [],
    ]

    sd = _state_for_player(state, 0, NAMES)
    assert sd["must_discard"] == 3


@pytest.mark.asyncio
async def test_human_player_submit_action():
    """HumanPlayer.submit_action resolves the pending future correctly."""
    human = HumanPlayer(name="Test")

    async def wait_and_get():
        return await human.choose_bid(None, 0, [None, Contract.THREE])

    # Start waiting for input
    task = asyncio.create_task(wait_and_get())
    await asyncio.sleep(0.01)  # let the task start

    # Submit action (like WebSocket handler would)
    human.submit_action(Contract.THREE)

    result = await task
    assert result == Contract.THREE


@pytest.mark.asyncio
async def test_multiple_games_with_rl_agents_stable():
    """Run 10 games with RL agents to verify stability."""
    for seed in range(10):
        human = AutoHuman()
        agents = [human]
        for i in range(3):
            a = RLAgent(name=f"AI-{i+1}")
            a.set_training(False)
            agents.append(a)

        loop = GameLoop(agents, rng=random.Random(seed))
        state, scores = await loop.run(dealer=seed % 4)
        assert state.phase == Phase.FINISHED
        if state.contract and state.contract.is_berac:
            assert 1 <= state.tricks_played <= 12
        else:
            assert state.tricks_played == 12
