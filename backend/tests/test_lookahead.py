"""Tests for the LookaheadAgent (Monte Carlo lookahead search)."""

import random
import pytest

from tarok.adapters.ai.lookahead_agent import (
    LookaheadAgent,
    _unseen_cards,
    _deal_unknown,
    _evaluate_move,
    _evaluate_move_perfect,
    _hand_strength,
    _choose_bid_heuristic,
    _simulate_random_playout,
)
from tarok.adapters.ai.random_agent import RandomPlayer
from tarok.entities.card import Card, CardType, DECK, Suit, SuitRank
from tarok.entities.game_state import Contract, GameState, Phase
from tarok.use_cases.game_loop import GameLoop


# ---- Helper: quick seeded game state after dealing ----

async def _make_dealt_state(seed: int = 42) -> GameState:
    """Run the game up to trick play with random bids and return the state."""
    players = [RandomPlayer(name=f"Bot-{i}", rng=random.Random(seed + i)) for i in range(4)]
    game = GameLoop(players, rng=random.Random(seed))
    state, scores = await game.run()
    return state


# ---- Unit tests for helpers ----

class TestUnseen:
    def test_unseen_cards_nonoverlap(self):
        """unseen + visible should cover the full deck."""
        rng = random.Random(7)
        # Build a minimal trick-play state
        state = GameState()
        hands = [list(DECK[i * 12 : (i + 1) * 12]) for i in range(4)]
        state.hands = hands
        state.talon = list(DECK[48:])
        state.phase = Phase.TRICK_PLAY

        for p in range(4):
            unseen = _unseen_cards(state, p)
            visible = set(state.hands[p]) | set(state.talon)
            # unseen + visible should cover everything minus any leftover
            assert len(set(unseen) & visible) == 0, "unseen shouldn't overlap with visible"


class TestDealUnknown:
    def test_preserves_hand_sizes(self):
        state = GameState()
        state.hands = [list(DECK[i * 12 : (i + 1) * 12]) for i in range(4)]
        state.talon = list(DECK[48:])
        state.phase = Phase.TRICK_PLAY

        rng = random.Random(99)
        new_state = _deal_unknown(state, 0, rng)

        # Our hand is untouched
        assert new_state.hands[0] == state.hands[0]
        # Other hands have the same sizes
        for p in range(1, 4):
            assert len(new_state.hands[p]) == len(state.hands[p])

    def test_does_not_mutate_original(self):
        state = GameState()
        state.hands = [list(DECK[i * 12 : (i + 1) * 12]) for i in range(4)]
        state.talon = list(DECK[48:])
        state.phase = Phase.TRICK_PLAY

        original_hand_1 = list(state.hands[1])
        _deal_unknown(state, 0, random.Random(1))
        assert state.hands[1] == original_hand_1


class TestHandStrength:
    def test_range(self):
        """Strength should be non-negative for any hand."""
        rng = random.Random(42)
        hand = rng.sample(DECK, 12)
        strength = _hand_strength(hand)
        assert strength >= 0

    def test_tarok_heavy_hand_is_stronger(self):
        """A hand full of taroks should be stronger than one without."""
        taroks_only = [c for c in DECK if c.card_type == CardType.TAROK][:12]
        suits_only = [c for c in DECK if c.card_type == CardType.SUIT][:12]
        assert _hand_strength(taroks_only) > _hand_strength(suits_only)


class TestBidHeuristic:
    def test_weak_hand_passes(self):
        """A very weak hand should pass (return None)."""
        # Pick 12 low-point suit cards
        weak = [c for c in DECK if c.card_type == CardType.SUIT and c.points == 0][:12]
        legal = [None, Contract.THREE, Contract.TWO, Contract.ONE]
        bid = _choose_bid_heuristic(weak, legal)
        # Very weak hands should pass
        assert bid is None

    def test_strong_hand_bids(self):
        """A strong hand should bid something."""
        strong = [c for c in DECK if c.card_type == CardType.TAROK][:12]
        legal = [None, Contract.THREE, Contract.TWO, Contract.ONE]
        bid = _choose_bid_heuristic(strong, legal)
        assert bid is not None
        assert bid in legal


# ---- Integration tests ----

@pytest.mark.asyncio
async def test_full_game_with_lookahead_agents():
    """Run a full game with 4 LookaheadAgents (low sims for speed)."""
    players = [
        LookaheadAgent(n_simulations=1, name=f"LA-{i}", rng=random.Random(42 + i))
        for i in range(4)
    ]
    game = GameLoop(players, rng=random.Random(123))
    state, scores = await game.run()

    assert state.phase == Phase.FINISHED
    assert state.tricks_played == 12
    for hand in state.hands:
        assert len(hand) == 0
    assert len(scores) == 4


@pytest.mark.asyncio
async def test_mixed_lookahead_and_random():
    """One LookaheadAgent + 3 RandomPlayers should complete successfully."""
    players = [
        LookaheadAgent(n_simulations=2, name="LA-0", rng=random.Random(10)),
        RandomPlayer(name="Rand-1", rng=random.Random(11)),
        RandomPlayer(name="Rand-2", rng=random.Random(12)),
        RandomPlayer(name="Rand-3", rng=random.Random(13)),
    ]
    game = GameLoop(players, rng=random.Random(99))
    state, scores = await game.run()

    assert state.phase == Phase.FINISHED
    assert state.tricks_played == 12
    assert len(scores) == 4


@pytest.mark.asyncio
async def test_choose_card_picks_legal_move():
    """choose_card should always return a card from legal_plays."""
    agent = LookaheadAgent(n_simulations=1, name="test", rng=random.Random(42))

    # Build a minimal state where we can test choose_card
    state = GameState()
    state.hands = [list(DECK[i * 12 : (i + 1) * 12]) for i in range(4)]
    state.talon = list(DECK[48:])
    state.phase = Phase.TRICK_PLAY
    state.current_player = 0
    state.contract = Contract.KLOP

    # Must start a trick before choosing a card
    from tarok.use_cases.play_trick import start_trick
    state = start_trick(state)

    legal = state.legal_plays(0)
    assert len(legal) > 0

    chosen = await agent.choose_card(state, 0, legal)
    assert chosen in legal


@pytest.mark.asyncio
async def test_multiple_games_stability():
    """Run 5 games with LookaheadAgents to check stability."""
    for seed in range(5):
        players = [
            LookaheadAgent(n_simulations=1, name=f"LA-{i}", rng=random.Random(seed * 10 + i))
            for i in range(4)
        ]
        game = GameLoop(players, rng=random.Random(seed))
        state, scores = await game.run()

        assert state.phase == Phase.FINISHED
        assert state.tricks_played == 12


@pytest.mark.asyncio
async def test_n_simulations_setter():
    agent = LookaheadAgent(n_simulations=10, name="test")
    assert agent.n_simulations == 10

    agent.n_simulations = 100
    assert agent.n_simulations == 100

    # Cannot go below 1
    agent.n_simulations = 0
    assert agent.n_simulations == 1


# ---- Trainer-compatibility stubs ----

class TestTrainerStubs:
    """LookaheadAgent must expose no-op stubs the trainer calls on all agents."""

    def test_clear_experiences(self):
        agent = LookaheadAgent(n_simulations=1, name="test")
        agent.clear_experiences()  # should not raise

    def test_finalize_game(self):
        agent = LookaheadAgent(n_simulations=1, name="test")
        agent.finalize_game(0.5)  # should not raise

    def test_set_training(self):
        agent = LookaheadAgent(n_simulations=1, name="test")
        agent.set_training(True)   # should not raise
        agent.set_training(False)

    def test_experiences_property(self):
        agent = LookaheadAgent(n_simulations=1, name="test")
        assert agent.experiences == []


# ---- Training integration tests ----

@pytest.mark.asyncio
async def test_training_with_lookahead_opponents():
    """A short training run with lookahead_ratio=1.0 should complete without errors."""
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.training_lab import PPOTrainer

    agents = [RLAgent(name=f"RL-{i}", hidden_size=64) for i in range(4)]
    trainer = PPOTrainer(
        agents,
        games_per_session=2,
        lookahead_ratio=1.0,
        lookahead_sims=1,
        fsp_ratio=0.0,
    )

    metrics = await trainer.train(num_sessions=1)

    # Training completed
    assert metrics.episode == 2
    assert metrics.session == 1
    # Lookahead metrics were recorded
    assert len(metrics.lookahead_score_history) == 1
    assert len(metrics.lookahead_bid_rate_history) == 1


@pytest.mark.asyncio
async def test_training_lookahead_scores_populated():
    """Lookahead score and bid-rate histories should contain valid numbers."""
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.training_lab import PPOTrainer

    agents = [RLAgent(name=f"RL-{i}", hidden_size=64) for i in range(4)]
    trainer = PPOTrainer(
        agents,
        games_per_session=4,
        lookahead_ratio=1.0,
        lookahead_sims=1,
        fsp_ratio=0.0,
    )

    metrics = await trainer.train(num_sessions=2)

    assert len(metrics.lookahead_score_history) == 2
    assert len(metrics.lookahead_bid_rate_history) == 2
    for score in metrics.lookahead_score_history:
        assert isinstance(score, float)
    for bid_rate in metrics.lookahead_bid_rate_history:
        assert 0.0 <= bid_rate <= 1.0


@pytest.mark.asyncio
async def test_training_metrics_to_dict_includes_lookahead():
    """to_dict() must include lookahead histories so the frontend can graph them."""
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.training_lab import PPOTrainer

    agents = [RLAgent(name=f"RL-{i}", hidden_size=64) for i in range(4)]
    trainer = PPOTrainer(
        agents,
        games_per_session=2,
        lookahead_ratio=1.0,
        lookahead_sims=1,
        fsp_ratio=0.0,
    )

    metrics = await trainer.train(num_sessions=1)
    d = metrics.to_dict()

    assert "lookahead_score_history" in d
    assert "lookahead_bid_rate_history" in d
    assert len(d["lookahead_score_history"]) >= 1
    assert len(d["lookahead_bid_rate_history"]) >= 1


# ---- Perfect information mode tests ----

class TestPerfectInformation:
    def test_perfect_info_flag_default_false(self):
        agent = LookaheadAgent(n_simulations=1, name="test")
        assert agent._perfect_information is False

    def test_perfect_info_flag_set(self):
        agent = LookaheadAgent(n_simulations=1, name="test", perfect_information=True)
        assert agent._perfect_information is True

    def test_evaluate_move_perfect_returns_number(self):
        """_evaluate_move_perfect should return a float score."""
        state = GameState()
        state.hands = [list(DECK[i * 12 : (i + 1) * 12]) for i in range(4)]
        state.talon = list(DECK[48:])
        state.phase = Phase.TRICK_PLAY
        state.current_player = 0
        state.contract = Contract.KLOP

        from tarok.use_cases.play_trick import start_trick
        state = start_trick(state)
        legal = state.legal_plays(0)
        assert len(legal) > 0

        rng = random.Random(42)
        score = _evaluate_move_perfect(state, 0, legal[0], 2, rng)
        assert isinstance(score, (int, float))


@pytest.mark.asyncio
async def test_perfect_info_game_completes():
    """Full game with perfect-info LookaheadAgents should finish normally."""
    players = [
        LookaheadAgent(n_simulations=1, name=f"PI-{i}",
                       rng=random.Random(42 + i), perfect_information=True)
        for i in range(4)
    ]
    game = GameLoop(players, rng=random.Random(123))
    state, scores = await game.run()

    assert state.phase == Phase.FINISHED
    assert state.tricks_played == 12
    assert len(scores) == 4


@pytest.mark.asyncio
async def test_perfect_info_choose_card_legal():
    """Perfect-info agent should still return a legal card."""
    agent = LookaheadAgent(n_simulations=1, name="test",
                           rng=random.Random(42), perfect_information=True)
    state = GameState()
    state.hands = [list(DECK[i * 12 : (i + 1) * 12]) for i in range(4)]
    state.talon = list(DECK[48:])
    state.phase = Phase.TRICK_PLAY
    state.current_player = 0
    state.contract = Contract.KLOP

    from tarok.use_cases.play_trick import start_trick
    state = start_trick(state)
    legal = state.legal_plays(0)

    chosen = await agent.choose_card(state, 0, legal)
    assert chosen in legal


@pytest.mark.asyncio
async def test_trainer_uses_perfect_info_lookahead():
    """Trainer should create perfect-info lookahead opponents by default."""
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.training_lab import PPOTrainer

    agents = [RLAgent(name=f"RL-{i}", hidden_size=64) for i in range(4)]
    trainer = PPOTrainer(
        agents,
        games_per_session=2,
        lookahead_ratio=1.0,
        lookahead_sims=1,
        fsp_ratio=0.0,
    )

    assert trainer._lookahead_opponents is not None
    for opp in trainer._lookahead_opponents:
        assert opp._perfect_information is True

    # And it should still complete training
    metrics = await trainer.train(num_sessions=1)
    assert metrics.episode == 2
    assert len(metrics.lookahead_score_history) == 1


@pytest.mark.asyncio
async def test_trainer_imperfect_info_lookahead():
    """Trainer can create imperfect-info (Monte Carlo) lookahead opponents."""
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.training_lab import PPOTrainer

    agents = [RLAgent(name=f"RL-{i}", hidden_size=64) for i in range(4)]
    trainer = PPOTrainer(
        agents,
        games_per_session=2,
        lookahead_ratio=1.0,
        lookahead_sims=1,
        lookahead_perfect_info=False,
        fsp_ratio=0.0,
    )

    assert trainer._lookahead_opponents is not None
    for opp in trainer._lookahead_opponents:
        assert opp._perfect_information is False

    metrics = await trainer.train(num_sessions=1)
    assert metrics.episode == 2
    assert len(metrics.lookahead_score_history) == 1


@pytest.mark.asyncio
async def test_normal_training_without_lookahead():
    """Normal self-play training (no lookahead) must still work."""
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.training_lab import PPOTrainer

    agents = [RLAgent(name=f"RL-{i}", hidden_size=64) for i in range(4)]
    trainer = PPOTrainer(
        agents,
        games_per_session=2,
        lookahead_ratio=0.0,
        stockskis_ratio=0.0,
        fsp_ratio=0.0,
    )

    assert trainer._lookahead_opponents is None
    metrics = await trainer.train(num_sessions=1)
    assert metrics.episode == 2
    assert metrics.session == 1
    # No lookahead metrics when disabled
    assert len(metrics.lookahead_score_history) == 0
    assert len(metrics.lookahead_bid_rate_history) == 0
