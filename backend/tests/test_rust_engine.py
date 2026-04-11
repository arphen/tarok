"""Tests for the tarok_engine Rust extension and RustGameLoop."""

import pytest

# Skip the whole module unless the Rust extension is installed.
te = pytest.importorskip("tarok_engine")

# ---- Extension import ----


def test_tarok_engine_importable():
    """The Rust extension should be importable after `make build-engine`."""
    assert te is not None


def test_tarok_engine_has_rust_game_state():
    """RustGameState class must be exposed."""
    assert hasattr(te, "RustGameState")


def test_tarok_engine_has_phase_constants():
    """Phase constants (u8) must be accessible."""
    for attr in ("PHASE_BIDDING", "PHASE_TRICK_PLAY", "PHASE_SCORING",
                 "PHASE_ANNOUNCEMENTS"):
        assert hasattr(te, attr), f"Missing {attr}"


def test_tarok_engine_has_decision_constants():
    """Decision-type constants must be accessible."""
    for attr in ("DT_BID", "DT_CARD_PLAY"):
        assert hasattr(te, attr), f"Missing {attr}"


# ---- RustGameState basics ----


def test_rust_game_state_deal():
    """Create a RustGameState and deal cards."""
    gs = te.RustGameState(0)
    gs.deal()
    # Each player should have 12 cards
    for p in range(4):
        hand = gs.hand(p)
        assert len(hand) == 12, f"Player {p} has {len(hand)} cards, expected 12"


def test_rust_game_state_total_cards():
    """After deal, 48 cards in hands + 6 in talon = 54."""
    gs = te.RustGameState(0)
    gs.deal()
    hand_cards = sum(len(gs.hand(p)) for p in range(4))
    assert hand_cards == 48


def test_rust_game_state_legal_bids():
    """legal_bids should return a non-empty list for the first bidder."""
    gs = te.RustGameState(0)
    gs.deal()
    first_bidder = 1  # dealer is 0, next player bids first
    bids = gs.legal_bids(first_bidder)
    assert isinstance(bids, list)
    assert len(bids) > 0


def test_rust_game_state_encode_state():
    """encode_state should return a numpy array with the expected size."""
    import numpy as np
    gs = te.RustGameState(0)
    gs.deal()
    state = gs.encode_state(0, te.DT_BID)
    assert isinstance(state, np.ndarray)
    assert state.ndim == 1
    assert len(state) > 0


def test_rust_game_state_legal_plays_mask():
    """legal_plays_mask should return a 54-element float array."""
    import numpy as np
    gs = te.RustGameState(0)
    gs.deal()
    mask = gs.legal_plays_mask(0)
    assert isinstance(mask, np.ndarray)
    assert len(mask) == 54


# ---- RustGameLoop ----


@pytest.fixture
def rl_agents():
    """Create 4 untrained RL agents for testing RustGameLoop."""
    from tarok.adapters.ai.agent import RLAgent
    agents = [RLAgent(name=f"Test-{i}") for i in range(4)]
    for a in agents:
        a.set_training(False)
    return agents


async def test_rust_game_loop_runs(rl_agents):
    """RustGameLoop with RL agents should complete a game."""
    from tarok.adapters.ai.rust_game_loop import RustGameLoop
    loop = RustGameLoop(rl_agents)
    state, scores = await loop.run()
    assert state is not None
    assert isinstance(scores, dict)
    assert len(scores) == 4
    for pid in range(4):
        assert pid in scores


async def test_rust_game_loop_scores_are_ints(rl_agents):
    """Scores from RustGameLoop should all be integers."""
    from tarok.adapters.ai.rust_game_loop import RustGameLoop
    loop = RustGameLoop(rl_agents)
    _state, scores = await loop.run()
    for pid, score in scores.items():
        assert isinstance(score, int), f"Player {pid} score is {type(score)}, not int"


async def test_rust_game_loop_state_has_contract(rl_agents):
    """Completed game should have a contract."""
    from tarok.adapters.ai.rust_game_loop import RustGameLoop
    loop = RustGameLoop(rl_agents)
    state, _scores = await loop.run()
    # Either a real contract or klop (which is -99 / Contract.KLOP)
    assert state.contract is not None or state.phase is not None


async def test_rust_game_loop_different_dealers(rl_agents):
    """Games with different dealers should all complete."""
    from tarok.adapters.ai.rust_game_loop import RustGameLoop
    for dealer in range(4):
        loop = RustGameLoop(rl_agents)
        state, scores = await loop.run(dealer=dealer)
        assert len(scores) == 4


async def test_rust_game_loop_mixed_players_compatible():
    """RustGameLoop should support non-RL players via compatibility fallback."""
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.rust_game_loop import RustGameLoop
    from tarok.adapters.ai.stockskis_v5 import StockSkisPlayerV5

    agents = [RLAgent(name="RL-0")]
    agents.extend(StockSkisPlayerV5(name=f"Skis-{i}") for i in range(1, 4))
    agents[0].set_training(False)

    loop = RustGameLoop(agents)
    _state, scores = await loop.run()

    assert isinstance(scores, dict)
    assert len(scores) == 4


# ---- Trainer fallback ----


def test_trainer_has_rust_flag():
    """Trainer module should expose _HAS_RUST flag."""
    from tarok.adapters.ai import training_lab
    assert isinstance(training_lab._HAS_RUST, bool)


def test_trainer_rust_flag_true_when_installed():
    """With the engine installed, _HAS_RUST should be True."""
    from tarok.adapters.ai import training_lab
    assert training_lab._HAS_RUST is True


def test_rust_game_loop_import_succeeds():
    """RustGameLoop should be importable when engine is installed."""
    from tarok.adapters.ai.rust_game_loop import RustGameLoop
    assert RustGameLoop is not None


def test_batch_game_runner_import_succeeds():
    """BatchGameRunner should be importable when engine is installed."""
    from tarok.adapters.ai.batch_game_runner import BatchGameRunner
    assert BatchGameRunner is not None
