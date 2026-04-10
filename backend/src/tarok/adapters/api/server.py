"""FastAPI server — REST + WebSocket adapter for the Tarok game."""

from __future__ import annotations

import asyncio
import base64
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.random_agent import RandomPlayer
from tarok.adapters.ai.trainer import PPOTrainer, TrainingMetrics
from tarok.adapters.api.human_player import HumanPlayer
from tarok.adapters.api.spectator_observer import SpectatorObserver, list_replays, load_replay
from tarok.adapters.api.ws_observer import WebSocketObserver
from tarok.adapters.api.schemas import (
    NewGameRequest,
    TrainingRequest,
    TrainingMetricsSchema,
    EvoRequest,
    BreedRequest,
)
from tarok.entities.card import Card, CardType, Suit, SuitRank, DECK
from tarok.entities.game_state import Bid, Contract, GameState, Phase, PlayerRole, Trick
from tarok.use_cases.game_loop import GameLoop

# --- Globals managed by lifespan ---
_trainer: PPOTrainer | None = None
_training_task: asyncio.Task | None = None
_latest_metrics: TrainingMetrics | None = None
_active_games: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _trainer, _training_task
    if _trainer:
        _trainer.stop()
    if _training_task and not _training_task.done():
        _training_task.cancel()


app = FastAPI(title="Tarok API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Training endpoints ----

@app.post("/api/training/start")
async def start_training(req: TrainingRequest):
    global _trainer, _training_task, _latest_metrics

    if _training_task and not _training_task.done():
        return {"status": "already_running", "metrics": _latest_metrics.to_dict() if _latest_metrics else None}

    # Resume from latest checkpoint if available
    checkpoint_path = Path("checkpoints/tarok_agent_latest.pt")
    if req.resume_from:
        checkpoint_path = Path("checkpoints") / req.resume_from
        req.resume = True

    if req.resume and checkpoint_path.exists():
        # Infer hidden_size from checkpoint to avoid shape mismatch
        agents = [
            RLAgent.from_checkpoint(checkpoint_path, name="Agent-0"),
            *[RLAgent(name=f"Agent-{i}", hidden_size=req.hidden_size) for i in range(1, 4)],
        ]
    else:
        agents = [RLAgent(name=f"Agent-{i}", hidden_size=req.hidden_size) for i in range(4)]

    _trainer = PPOTrainer(
        agents, lr=req.learning_rate, device="cpu",
        games_per_session=req.games_per_session,
        stockskis_ratio=req.stockskis_ratio,
        stockskis_strength=req.stockskis_strength,
        use_rust_engine=req.use_rust_engine,
        warmup_games=req.warmup_games,
        batch_concurrency=req.batch_concurrency,
    )

    async def on_metrics(metrics: TrainingMetrics):
        global _latest_metrics
        _latest_metrics = metrics

    _trainer.add_metrics_callback(on_metrics)

    async def run_training():
        global _latest_metrics
        try:
            result = await _trainer.train(req.num_sessions)
            _latest_metrics = result
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Training task failed")

    _training_task = asyncio.create_task(run_training())
    return {"status": "started", "num_sessions": req.num_sessions, "games_per_session": req.games_per_session,
            "message": "Run ID will appear in metrics once training begins"}


@app.post("/api/training/stop")
async def stop_training():
    global _trainer, _training_task
    if _trainer:
        _trainer.stop()
    return {"status": "stopped"}


@app.get("/api/training/metrics")
async def get_metrics() -> dict:
    if _latest_metrics:
        return _latest_metrics.to_dict()
    return TrainingMetrics().to_dict()


@app.get("/api/training/status")
async def training_status():
    running = _training_task is not None and not _training_task.done()
    return {"running": running}


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List all saved checkpoint files (training, breeding, and HOF)."""
    ckpt_dir = Path("checkpoints")
    breed_dir = Path("checkpoints/breeding_results")
    hof_dir = Path("checkpoints/hall_of_fame")
    
    result = []
    # Always include "latest" if the file exists
    latest_path = ckpt_dir / "tarok_agent_latest.pt"
    if latest_path.exists():
        import torch as _torch
        try:
            meta = _torch.load(latest_path, map_location="cpu", weights_only=False)
            result.append({
                "filename": "tarok_agent_latest.pt",
                "episode": meta.get("episode", 0),
                "session": meta.get("session", 0),
                "win_rate": meta.get("metrics", {}).get("win_rate", 0),
                "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
                "model_name": meta.get("model_name", None),
                "is_bred": False,
                "is_hof": False,
            })
        except Exception:
            result.append({"filename": "tarok_agent_latest.pt", "episode": 0, "is_bred": False, "is_hof": False})

    if not ckpt_dir.exists():
        return {"checkpoints": result}
    files = [f for f in ckpt_dir.glob("tarok_agent_*.pt") if f.name != "tarok_agent_latest.pt"]
    if breed_dir.exists():
        files.extend([f for f in breed_dir.glob("bred_model_*.pt")])
    if hof_dir.exists():
        files.extend([f for f in hof_dir.glob("hof_*.pt")])

    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    result = []
    for f in files:
        import torch as _torch
        try:
            meta = _torch.load(f, map_location="cpu", weights_only=False)
            model_name = meta.get("model_name", None) or meta.get("display_name", None)
            is_hof = "hof_" in f.name
            
            result.append({
                "filename": str(f.relative_to(ckpt_dir)),
                "episode": meta.get("episode", 0),
                "session": meta.get("session", 0),
                "win_rate": meta.get("metrics", {}).get("win_rate", 0),
                "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
                "model_name": model_name,
                "is_bred": "bred_model" in f.name,
                "is_hof": is_hof,
                "persona": meta.get("persona") if is_hof else None,
            })
        except Exception:
            result.append({
                "filename": str(f.relative_to(ckpt_dir)),
                "episode": 0,
                "is_bred": "bred_model" in f.name,
                "is_hof": "hof_" in f.name,
            })
    return {"checkpoints": result}


# ---- Game endpoints ----

def _load_opponent(choice: str, index: int) -> RLAgent:
    """Load a single AI opponent from a checkpoint choice."""
    if choice == "random":
        agent = RLAgent(name=f"AI-{index + 1}")
        agent.set_training(False)
        return agent

    if choice == "latest":
        path = Path("checkpoints/tarok_agent_latest.pt")
    else:
        path = Path("checkpoints") / choice

    # Try to extract a display name from HOF metadata
    name = f"AI-{index + 1}"
    if path.exists():
        try:
            import torch as _torch
            meta = _torch.load(path, map_location="cpu", weights_only=False)
            if meta.get("display_name"):
                name = meta["display_name"]
            elif meta.get("model_name"):
                name = meta["model_name"]
        except Exception:
            pass
        agent = RLAgent.from_checkpoint(path, name=name)
    else:
        agent = RLAgent(name=name)
    agent.set_training(False)
    return agent


@app.post("/api/game/new")
async def new_game(req: NewGameRequest | None = None):
    """Create a new human-vs-AI game with optional per-opponent model selection."""
    game_id = f"game-{len(_active_games)}"
    opponents = (req.opponents if req else ["latest"] * 3)[:3]
    while len(opponents) < 3:
        opponents.append("latest")

    human = HumanPlayer(name="You")
    agents: list = [human]
    for i, choice in enumerate(opponents):
        agents.append(_load_opponent(choice, i))

    _active_games[game_id] = {
        "human": human,
        "agents": agents,
        "game_loop": None,
        "state": None,
    }
    return {"game_id": game_id}


@app.websocket("/ws/game/{game_id}")
async def game_websocket(ws: WebSocket, game_id: str):
    await ws.accept()

    if game_id not in _active_games:
        await ws.close(code=4004, reason="Game not found")
        return

    game_info = _active_games[game_id]
    human: HumanPlayer = game_info["human"]
    agents = game_info["agents"]
    player_names = [a.name for a in agents]

    observer = WebSocketObserver(ws, player_idx=0, player_names=player_names)
    game_loop = GameLoop(agents, observer=observer)

    # Start game in background
    game_task = asyncio.create_task(game_loop.run())

    try:
        while True:
            data = await ws.receive_json()
            action_type = data.get("action")

            if action_type == "bid":
                contract_val = data.get("contract")
                if contract_val is None:
                    human.submit_action(None)
                else:
                    human.submit_action(Contract(contract_val))

            elif action_type == "call_king":
                suit = Suit(data["suit"])
                king = Card(CardType.SUIT, SuitRank.KING.value, suit)
                human.submit_action(king)

            elif action_type == "choose_talon":
                human.submit_action(data["group_index"])

            elif action_type == "discard":
                cards = []
                for c in data["cards"]:
                    card = Card(
                        CardType(c["card_type"]),
                        c["value"],
                        Suit(c["suit"]) if c.get("suit") else None,
                    )
                    cards.append(card)
                human.submit_action(cards)

            elif action_type == "play_card":
                c = data["card"]
                card = Card(
                    CardType(c["card_type"]),
                    c["value"],
                    Suit(c["suit"]) if c.get("suit") else None,
                )
                human.submit_action(card)

            elif action_type == "set_delay":
                delay = data.get("delay", 1.0)
                observer.ai_delay = max(0.0, min(float(delay), 5.0))

    except WebSocketDisconnect:
        game_task.cancel()
        if game_id in _active_games:
            del _active_games[game_id]


# ---- Spectator endpoints ----

_spectator_games: dict[str, dict] = {}


class SpectateRequest(BaseModel):
    agents: list[dict] = []  # [{name, type: "rl"|"random", checkpoint?}]
    delay: float = 1.5  # seconds between moves


@app.post("/api/spectate/new")
async def new_spectate(req: SpectateRequest):
    """Create a new 4-AI spectator game."""
    game_id = f"spectate-{len(_spectator_games)}"
    replay_name = f"{game_id}-{int(time.time())}.json"

    agents = []
    for i, agent_cfg in enumerate(req.agents[:4]):
        agent_type = agent_cfg.get("type", "rl")
        agent_name = agent_cfg.get("name", f"Agent-{i}")
        checkpoint = agent_cfg.get("checkpoint")

        if agent_type == "random":
            agents.append(RandomPlayer(name=agent_name))
        else:
            ckpt_path = Path("checkpoints") / checkpoint if checkpoint else Path("checkpoints/tarok_agent_latest.pt")
            if ckpt_path.exists():
                agent = RLAgent.from_checkpoint(ckpt_path, name=agent_name)
            else:
                agent = RLAgent(name=agent_name)
            agent.set_training(False)
            agents.append(agent)

    # Fill remaining slots with RL agents
    while len(agents) < 4:
        ckpt_path = Path("checkpoints/tarok_agent_latest.pt")
        if ckpt_path.exists():
            agent = RLAgent.from_checkpoint(ckpt_path, name=f"Agent-{len(agents)}")
        else:
            agent = RLAgent(name=f"Agent-{len(agents)}")
        agent.set_training(False)
        agents.append(agent)

    _spectator_games[game_id] = {
        "agents": agents,
        "delay": req.delay,
        "spectators": [],
        "game_task": None,
        "replay_name": replay_name,
    }
    return {"game_id": game_id, "replay_name": replay_name}


@app.websocket("/ws/spectate/{game_id}")
async def spectate_websocket(ws: WebSocket, game_id: str):
    await ws.accept()

    if game_id not in _spectator_games:
        await ws.close(code=4004, reason="Game not found")
        return

    game_info = _spectator_games[game_id]
    agents = game_info["agents"]
    spectators: list[WebSocket] = game_info["spectators"]
    spectators.append(ws)

    player_names = [a.name for a in agents]

    # Only start the game when the first spectator connects
    if game_info["game_task"] is None:
        observer = SpectatorObserver(
            websockets=spectators,
            player_names=player_names,
            delay=game_info["delay"],
            replay_name=game_info.get("replay_name"),
            replay_metadata={
                "source": "spectate",
                "label": f"Spectate {game_id}",
                "game_id": game_id,
            },
        )
        game_loop = GameLoop(agents, observer=observer)
        game_info["observer"] = observer
        game_info["game_task"] = asyncio.create_task(game_loop.run())

    try:
        # Keep connection open; handle actions like "next_trick"
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            if action == "next_trick" and "observer" in game_info:
                game_info["observer"].next_trick_event.set()
    except WebSocketDisconnect:
        if ws in spectators:
            spectators.remove(ws)
        if not spectators and game_id in _spectator_games:
            task = _spectator_games[game_id].get("game_task")
            if task and not task.done():
                task.cancel()
            del _spectator_games[game_id]


@app.websocket("/ws/training")
async def training_websocket(ws: WebSocket):
    """Stream training metrics to the frontend."""
    await ws.accept()
    try:
        while True:
            if _latest_metrics:
                await ws.send_json({
                    "event": "metrics",
                    "data": _latest_metrics.to_dict(),
                })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


# ---- Health check ----

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/replays")
async def get_replays():
    return {"replays": list_replays()}


@app.get("/api/replays/{replay_name}")
async def get_replay(replay_name: str):
    try:
        return load_replay(replay_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Replay not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---- Camera Agent: analyze hand and recommend play ----

class CardInput(BaseModel):
    card_type: str  # "tarok" or "suit"
    value: int
    suit: str | None = None


class AnalyzeRequest(BaseModel):
    """Cards the user holds + the current trick + game context."""
    hand: list[CardInput]
    trick: list[CardInput] = []  # cards already played in current trick
    contract: str = "three"  # contract name
    position: int = 0  # 0=declarer, 1=partner, 2/3=opponent
    tricks_played: int = 0
    played_cards: list[CardInput] = []  # all previously played cards


def _parse_card(ci: CardInput) -> Card:
    ct = CardType(ci.card_type)
    s = Suit(ci.suit) if ci.suit else None
    return Card(ct, ci.value, s)


def _card_to_dict(card: Card) -> dict:
    return {
        "card_type": card.card_type.value,
        "value": card.value,
        "suit": card.suit.value if card.suit else None,
        "label": card.label,
        "points": card.points,
    }


# Map of known cards in the DECK for quick label-to-card lookup
_LABEL_TO_CARD: dict[str, Card] = {c.label: c for c in DECK}

CONTRACT_NAME_MAP = {
    "three": Contract.THREE,
    "two": Contract.TWO,
    "one": Contract.ONE,
    "solo_three": Contract.SOLO_THREE,
    "solo_two": Contract.SOLO_TWO,
    "solo_one": Contract.SOLO_ONE,
    "solo": Contract.SOLO,
}


@app.post("/api/analyze")
async def analyze_hand(req: AnalyzeRequest):
    """Given a hand of cards and game context, return the AI's recommended play.

    This endpoint lets users photograph a real Tarok hand, input the cards,
    and get the trained agent's recommendation for what to play.
    """
    # Parse cards
    hand = [_parse_card(c) for c in req.hand]
    trick_cards = [_parse_card(c) for c in req.trick]
    played = [_parse_card(c) for c in req.played_cards]

    # Build a synthetic game state for the agent
    state = GameState(phase=Phase.TRICK_PLAY)
    contract = CONTRACT_NAME_MAP.get(req.contract, Contract.THREE)
    state.contract = contract
    state.declarer = 0 if req.position == 0 else 1

    # Set player 0 as the user
    state.hands[0] = list(hand)
    state.current_player = 0
    state.roles = {
        0: PlayerRole.DECLARER if req.position == 0 else (
            PlayerRole.PARTNER if req.position == 1 else PlayerRole.OPPONENT
        ),
        1: PlayerRole.PARTNER if req.position != 1 else PlayerRole.DECLARER,
        2: PlayerRole.OPPONENT,
        3: PlayerRole.OPPONENT,
    }

    # Build current trick if cards have been played
    if trick_cards:
        state.current_trick = Trick(lead_player=(4 - len(trick_cards)) % 4)
        for i, card in enumerate(trick_cards):
            player_idx = (state.current_trick.lead_player + i) % 4
            state.current_trick.cards.append((player_idx, card))
    else:
        state.current_trick = Trick(lead_player=0)

    # Record previously played tricks
    state.tricks = []
    # Approximate: create empty trick records for played tricks count
    for i in range(req.tricks_played):
        t = Trick(lead_player=0)
        # Minimal trick data
        state.tricks.append(t)

    # Compute legal plays
    legal = state.legal_plays(0)

    # Load the trained agent and get its recommendation
    agent = RLAgent(name="Advisor")
    agent.set_training(False)
    checkpoint_path = Path("checkpoints/tarok_agent_latest.pt")
    if checkpoint_path.exists():
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agent.network.load_state_dict(checkpoint["model_state_dict"])

    # Get the agent's card choice
    recommended = await agent.choose_card(state, 0)

    # Also rank all legal plays by the agent's policy
    from tarok.adapters.ai.encoding import encode_state, encode_legal_mask, CARD_TO_IDX
    import torch

    state_tensor = torch.tensor(encode_state(state, 0), dtype=torch.float32).unsqueeze(0)
    legal_mask = torch.tensor(encode_legal_mask(legal), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, value = agent.network(state_tensor)

    # Mask illegal actions
    masked = logits.clone()
    masked[legal_mask == 0] = float('-inf')
    probs = torch.softmax(masked, dim=-1).squeeze(0)

    # Build ranked recommendations
    ranked = []
    for card in legal:
        idx = CARD_TO_IDX.get(card)
        if idx is not None:
            prob = probs[idx].item()
            ranked.append({
                "card": _card_to_dict(card),
                "probability": round(prob, 4),
            })

    ranked.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "recommended": _card_to_dict(recommended),
        "legal_plays": [_card_to_dict(c) for c in legal],
        "ranked_plays": ranked,
        "position_value": round(value.item(), 4) if value is not None else None,
        "has_trained_model": checkpoint_path.exists(),
    }


# ---- Camera Agent: bid recommendation ----

class AnalyzeBidRequest(BaseModel):
    """Hand cards + bidding context → AI bid recommendation."""
    hand: list[CardInput]
    bids: list[dict] = []  # [{player: int, contract: str|null}]
    dealer: int = 0


@app.post("/api/analyze-bid")
async def analyze_bid(req: AnalyzeBidRequest):
    """Given a hand and bidding history, return the AI's recommended bid."""
    from tarok.adapters.ai.encoding import (
        encode_state, encode_bid_mask, BID_ACTIONS, BID_TO_IDX, DecisionType,
    )
    import torch

    hand = [_parse_card(c) for c in req.hand]

    # Build synthetic game state in BIDDING phase
    state = GameState(phase=Phase.BIDDING)
    state.dealer = req.dealer
    state.hands[0] = list(hand)
    state.current_player = 0
    state.current_bidder = 0

    # Replay bid history
    for b in req.bids:
        c = CONTRACT_NAME_MAP.get(b.get("contract", ""), None) if b.get("contract") else None
        state.bids.append(Bid(player=b["player"], contract=c))

    # Legal bids for player 0
    legal = state.legal_bids(0)

    # Load agent
    agent = RLAgent(name="BidAdvisor")
    agent.set_training(False)
    checkpoint_path = Path("checkpoints/tarok_agent_latest.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agent.network.load_state_dict(checkpoint["model_state_dict"])

    # Get bid recommendation
    recommended = await agent.choose_bid(state, 0, legal)

    # Get ranked probabilities
    state_tensor = encode_state(state, 0, DecisionType.BID).unsqueeze(0)
    mask = encode_bid_mask(legal).unsqueeze(0)
    with torch.no_grad():
        logits, value = agent.network(state_tensor, DecisionType.BID)
    masked = logits.clone()
    masked[mask == 0] = float('-inf')
    probs = torch.softmax(masked, dim=-1).squeeze(0)

    ranked = []
    for bid_option in legal:
        if bid_option is None:
            idx = BID_TO_IDX.get(None, 0)
            label = "Pass"
        else:
            idx = BID_TO_IDX.get(bid_option, 0)
            label = bid_option.value if isinstance(bid_option.value, str) else bid_option.name.replace("_", " ").title()
        ranked.append({
            "contract": bid_option.value if bid_option else None,
            "name": label,
            "probability": round(probs[idx].item(), 4),
        })
    ranked.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "recommended": recommended.value if recommended else None,
        "recommended_name": (recommended.name.replace("_", " ").title() if recommended else "Pass"),
        "legal_bids": [{"value": b.value if b else None, "name": b.name.replace("_", " ").title() if b else "Pass"} for b in legal],
        "ranked_bids": ranked,
        "position_value": round(value.item(), 4) if value is not None else None,
        "has_trained_model": checkpoint_path.exists(),
    }


# ---- Camera Agent: king call recommendation ----

class AnalyzeKingRequest(BaseModel):
    hand: list[CardInput]
    contract: str = "three"


@app.post("/api/analyze-king")
async def analyze_king(req: AnalyzeKingRequest):
    """Given a hand and contract, recommend which king to call."""
    from tarok.adapters.ai.encoding import (
        encode_state, encode_king_mask, KING_ACTIONS, SUIT_TO_IDX, DecisionType,
    )
    import torch

    hand = [_parse_card(c) for c in req.hand]
    contract = CONTRACT_NAME_MAP.get(req.contract, Contract.THREE)

    state = GameState(phase=Phase.KING_CALLING)
    state.hands[0] = list(hand)
    state.contract = contract
    state.declarer = 0
    state.current_player = 0
    state.roles = {0: PlayerRole.DECLARER, 1: PlayerRole.OPPONENT, 2: PlayerRole.OPPONENT, 3: PlayerRole.OPPONENT}

    # Find callable kings (kings NOT in hand)
    callable_kings = []
    for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
        king = Card(CardType.SUIT, SuitRank.KING.value, suit)
        if king not in hand:
            callable_kings.append(king)

    if not callable_kings:
        return {"recommended": None, "callable_kings": [], "has_trained_model": False}

    agent = RLAgent(name="KingAdvisor")
    agent.set_training(False)
    checkpoint_path = Path("checkpoints/tarok_agent_latest.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agent.network.load_state_dict(checkpoint["model_state_dict"])

    recommended = await agent.choose_king(state, 0, callable_kings)

    return {
        "recommended": _card_to_dict(recommended),
        "callable_kings": [_card_to_dict(k) for k in callable_kings],
        "has_trained_model": checkpoint_path.exists(),
    }


# ---- Evolution endpoints ----

_evo_task: asyncio.Task | None = None
_evo_progress: dict | None = None


@app.post("/api/evo/start")
async def start_evolution(req: EvoRequest):
    global _evo_task, _evo_progress

    if _evo_task and not _evo_task.done():
        return {"status": "already_running"}

    from tarok.adapters.ai.evo_optimizer import EvoConfig, EvoProgress, run_evolution as _run_evo

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _evo_progress = EvoProgress(
        total_generations=req.num_generations,
        phase="starting",
    ).to_dict()

    config = EvoConfig(
        population_size=req.population_size,
        num_generations=req.num_generations,
        eval_sessions=req.eval_sessions,
        games_per_session=req.games_per_session,
        oracle=req.oracle,
        device=device,
    )

    async def on_progress(progress: EvoProgress):
        global _evo_progress
        _evo_progress = progress.to_dict()

    config.progress_callback = on_progress

    async def run_evo():
        global _evo_progress
        await _run_evo(config)

    _evo_task = asyncio.create_task(run_evo())
    return {"status": "started", "config": {
        "population_size": req.population_size,
        "num_generations": req.num_generations,
        "eval_sessions": req.eval_sessions,
        "games_per_session": req.games_per_session,
    }}


@app.post("/api/evo/stop")
async def stop_evolution():
    global _evo_task
    if _evo_task and not _evo_task.done():
        _evo_task.cancel()
    return {"status": "stopped"}


@app.get("/api/evo/progress")
async def get_evo_progress() -> dict:
    if _evo_progress:
        return _evo_progress
    from tarok.adapters.ai.evo_optimizer import EvoProgress
    return EvoProgress().to_dict()


@app.get("/api/evo/status")
async def evo_status():
    running = _evo_task is not None and not _evo_task.done()
    return {"running": running}


# ---- Breeding endpoints ----

_breed_task: asyncio.Task | None = None
_breed_progress: dict | None = None
_breed_config = None  # Keep reference for clean stop


@app.post("/api/breed/start")
async def start_breeding(req: BreedRequest):
    global _breed_task, _breed_progress, _breed_config

    if _breed_task and not _breed_task.done():
        return {"status": "already_running"}

    from tarok.adapters.ai.breeding import BreedingConfig, BreedingProgress, run_breeding as _run_breed
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _breed_progress = BreedingProgress(
        total_cycles=req.num_cycles,
        total_generations=req.num_generations,
        warmup_total_sessions=req.warmup_sessions,
        refine_total_sessions=req.refine_sessions,
        phase="starting",
    ).to_dict()

    config = BreedingConfig(
        warmup_sessions=req.warmup_sessions,
        warmup_games_per_session=req.warmup_games_per_session,
        population_size=req.population_size,
        num_generations=req.num_generations,
        num_cycles=req.num_cycles,
        eval_games=req.eval_games,
        refine_sessions=req.refine_sessions,
        refine_games_per_session=req.refine_games_per_session,
        oracle=req.oracle,
        device=device,
        resume=req.resume,
        resume_from=req.resume_from,
        model_name=req.model_name,
        stockskis_eval=req.stockskis_eval,
        stockskis_strength=req.stockskis_strength,
    )
    _breed_config = config

    async def on_progress(progress: BreedingProgress):
        global _breed_progress
        _breed_progress = progress.to_dict()

    config.progress_callback = on_progress

    async def run_breed():
        await _run_breed(config)

    _breed_task = asyncio.create_task(run_breed())
    return {"status": "started", "config": {
        "warmup_sessions": req.warmup_sessions,
        "population_size": req.population_size,
        "num_generations": req.num_generations,
        "num_cycles": req.num_cycles,
        "eval_games": req.eval_games,
        "refine_sessions": req.refine_sessions,
    }}


@app.post("/api/breed/stop")
async def stop_breeding():
    global _breed_task, _breed_config
    if _breed_config:
        _breed_config._running = False
    if _breed_task and not _breed_task.done():
        _breed_task.cancel()
    return {"status": "stopped"}


@app.get("/api/breed/progress")
async def get_breed_progress() -> dict:
    if _breed_progress:
        return _breed_progress
    from tarok.adapters.ai.breeding import BreedingProgress
    return BreedingProgress().to_dict()


@app.get("/api/breed/status")
async def breed_status():
    running = _breed_task is not None and not _breed_task.done()
    return {"running": running}


# ---- Training Lab endpoints ----

@app.post("/api/lab/create")
async def lab_create(req: dict = {}):
    """Create a fresh neural network for the training lab."""
    from tarok.adapters.ai.training_lab import create_lab_network
    hidden_size = req.get("hidden_size", 256) if isinstance(req, dict) else 256
    create_lab_network(hidden_size)
    return {"status": "created", "hidden_size": hidden_size}


@app.post("/api/lab/load")
async def lab_load(req: dict = {}):
    """Load an existing checkpoint into the training lab."""
    from tarok.adapters.ai.training_lab import _lab, load_lab_checkpoint

    if _lab.running:
        return {"status": "already_running"}

    choice = req.get("checkpoint") if isinstance(req, dict) else None
    if not choice:
        return {"status": "missing_checkpoint"}

    try:
        info = load_lab_checkpoint(choice)
    except FileNotFoundError:
        return {"status": "not_found", "checkpoint": choice}
    except ValueError as exc:
        return {"status": "invalid", "error": str(exc)}

    return {"status": "loaded", **info}


@app.post("/api/lab/start")
async def lab_start(req: dict = {}):
    """Start the imitation learning pipeline."""
    from tarok.adapters.ai.training_lab import start_lab_training, _lab

    if _lab.running:
        return {"status": "already_running"}

    if _lab.network is None:
        from tarok.adapters.ai.training_lab import create_lab_network
        create_lab_network(req.get("hidden_size", 256))

    await start_lab_training(
        expert_games=req.get("expert_games", 500_000),
        training_epochs=req.get("training_epochs", 3),
        eval_games=req.get("eval_games", 500),
        num_rounds=req.get("num_rounds", 10),
        batch_size=req.get("batch_size", 2048),
        learning_rate=req.get("learning_rate", 1e-3),
        chunk_size=req.get("chunk_size", 50_000),
    )
    return {"status": "started"}


@app.post("/api/lab/self-play")
async def lab_self_play(req: dict = {}):
    """Start self-play PPO training on the lab network."""
    from tarok.adapters.ai.training_lab import start_self_play, _lab

    if _lab.running:
        return {"status": "already_running"}

    if _lab.network is None:
        from tarok.adapters.ai.training_lab import create_lab_network
        create_lab_network(req.get("hidden_size", 256))

    await start_self_play(
        num_sessions=req.get("num_sessions", 50),
        games_per_session=req.get("games_per_session", 20),
        eval_games=req.get("eval_games", 100),
        eval_interval=req.get("eval_interval", 5),
        learning_rate=req.get("learning_rate", 3e-4),
        stockskis_ratio=req.get("stockskis_ratio", 0.0),
        fsp_ratio=req.get("fsp_ratio", 0.3),
        pbt_enabled=req.get("pbt_enabled", False),
        population_size=req.get("population_size", 4),
        exploit_top_ratio=req.get("exploit_top_ratio", 0.25),
        exploit_bottom_ratio=req.get("exploit_bottom_ratio", 0.25),
        mutation_scale=req.get("mutation_scale", 1.0),
        time_limit_minutes=req.get("time_limit_minutes", 5.0),
    )
    return {"status": "started"}


@app.post("/api/lab/stop")
async def lab_stop():
    """Stop the training lab."""
    from tarok.adapters.ai.training_lab import stop_lab
    stop_lab()
    return {"status": "stopped"}


@app.post("/api/lab/overnight")
async def lab_overnight(req: dict = {}):
    """Start a long-running overnight training session.

    Auto-configures PBT self-play for ~8-12 hours of continuous training.
    Evaluates every 10 generations against V1/V2/V3 and saves snapshots.
    Just hit Stop when you wake up.
    """
    from tarok.adapters.ai.training_lab import start_self_play, _lab

    if _lab.running:
        return {"status": "already_running"}

    if _lab.network is None:
        from tarok.adapters.ai.training_lab import create_lab_network
        create_lab_network(req.get("hidden_size", 256))

    # Sensible overnight defaults — large generation count, periodic eval
    await start_self_play(
        num_sessions=req.get("num_sessions", 10000),
        games_per_session=req.get("games_per_session", 20),
        eval_games=req.get("eval_games", 200),
        eval_interval=req.get("eval_interval", 10),
        learning_rate=req.get("learning_rate", 3e-4),
        stockskis_ratio=req.get("stockskis_ratio", 0.0),
        fsp_ratio=req.get("fsp_ratio", 0.3),
        pbt_enabled=req.get("pbt_enabled", True),
        population_size=req.get("population_size", 4),
        exploit_top_ratio=req.get("exploit_top_ratio", 0.2),
        exploit_bottom_ratio=req.get("exploit_bottom_ratio", 0.3),
        mutation_scale=req.get("mutation_scale", 0.8),
        time_limit_minutes=req.get("time_limit_minutes", 480),
    )
    return {"status": "started", "mode": "overnight", "num_sessions": req.get("num_sessions", 10000)}


@app.post("/api/lab/reset")
async def lab_reset():
    """Reset the training lab to initial state."""
    from tarok.adapters.ai.training_lab import reset_lab
    reset_lab()
    return {"status": "reset"}


@app.get("/api/lab/state")
async def lab_state():
    """Get current training lab state."""
    from tarok.adapters.ai.training_lab import get_lab_state
    return get_lab_state()


@app.get("/api/lab/hof")
async def lab_hof():
    """List all Hall of Fame models."""
    from tarok.adapters.ai.training_lab import list_hof
    return {"models": list_hof()}
