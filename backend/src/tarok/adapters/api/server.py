"""FastAPI server — REST + WebSocket adapter for the Tarok game."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tarok.adapters.players.neural_player import NeuralPlayer
from tarok.adapters.players.factory import get_player_factory
from tarok.adapters.api.checkpoint_utils import resolve_checkpoint
from tarok.adapters.players.human_player import HumanPlayer
from tarok.adapters.api.experience_logger import HumanPlayExperienceLogger
from tarok.adapters.api.spectator_observer import SpectatorObserver, list_replays, load_replay
from tarok.adapters.api.ws_observer import WebSocketObserver
from tarok.adapters.api.schemas import (
    NewGameRequest,
    TrainingRequest,
)
from tarok.entities import Card, Suit, SuitRank, DECK, Contract
from tarok.entities.game_types import suit_card
from tarok.use_cases.game_loop import RustGameLoop as GameLoop
from tarok.use_cases.rust_state import _RUST_U8_TO_PY_CONTRACT

from tarok.adapters.api.routers.analyze_router import router as analyze_router
from tarok.adapters.api.routers.tournament_router import router as tournament_router
from tarok.adapters.api.routers.arena_router import router as arena_router


def _card_from_dict(c: dict) -> Card:
    """Reconstruct a Card from a frontend card dict using the Rust index."""
    if c.get("card_type") == "tarok" or c.get("card_type") == 0:
        # tarok: value is 1-22, idx is value-1
        return Card(int(c["value"]) - 1)
    else:
        suit = Suit(c["suit"])
        rank = SuitRank(int(c["value"]))
        return suit_card(suit, rank)


# --- Globals managed by lifespan ---
_training_task: asyncio.Task | None = None
_latest_metrics: dict | None = None
_active_games: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _training_task
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

app.include_router(analyze_router)
app.include_router(tournament_router)
app.include_router(arena_router)


# ---- Training endpoints ----


@app.post("/api/training/start")
async def start_training(req: TrainingRequest):
    del req
    return {
        "status": "disabled",
        "message": "Training lab was removed; /api/training/start is no longer available.",
    }


@app.post("/api/training/stop")
async def stop_training():
    global _training_task
    if _training_task and not _training_task.done():
        _training_task.cancel()
    return {"status": "stopped"}


@app.get("/api/training/metrics")
async def get_metrics() -> dict:
    if _latest_metrics:
        return _latest_metrics
    return {"status": "disabled", "message": "Training lab removed"}


@app.get("/api/training/status")
async def training_status():
    running = _training_task is not None and not _training_task.done()
    return {"running": running, "mode": "disabled"}


# Checkpoint metadata cache: {filepath_str: (mtime, metadata_dict)}
_checkpoint_cache: dict[str, tuple[float, dict]] = {}


def _load_checkpoint_meta(fpath: Path, ckpt_dir: Path) -> dict:
    """Load checkpoint metadata, using cache when file hasn't changed."""
    import torch as _torch

    fstr = str(fpath)
    mtime = fpath.stat().st_mtime
    cached = _checkpoint_cache.get(fstr)
    if cached and cached[0] == mtime:
        return cached[1]

    if fpath.name == "tarok_agent_latest.pt":
        fname = "tarok_agent_latest.pt"
    else:
        fname = fpath.name

    try:
        meta = _torch.load(fpath, map_location="cpu", weights_only=False)
        entry = {
            "filename": fname,
            "episode": meta.get("episode", 0),
            "session": meta.get("session", 0),
            "win_rate": meta.get("metrics", {}).get("win_rate", 0),
            "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
            "model_name": meta.get("model_name", None),
        }
    except Exception:
        entry = {"filename": fname, "episode": 0}

    _checkpoint_cache[fstr] = (mtime, entry)
    return entry


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List checkpoints from the top-level checkpoints/ directory.

    - HOF: checkpoints/hall_of_fame/*.pt  (manually curated, committed to git)
    - Persona models: checkpoints/{PersonaName}/_current.pt
    """
    import torch as _torch

    root_ckpt_dir = Path("../data/checkpoints")
    legacy_ckpt_dirs = [Path("checkpoints"), Path("../checkpoints")]
    hof_dir = root_ckpt_dir / "hall_of_fame"
    result = []
    seen_filenames: set[str] = set()

    if not root_ckpt_dir.exists():
        # Still return legacy checkpoints even when the new canonical dir is absent.
        for legacy_dir in legacy_ckpt_dirs:
            if not legacy_dir.exists():
                continue
            for f in sorted(legacy_dir.glob("*.pt")):
                if f.name in seen_filenames:
                    continue
                seen_filenames.add(f.name)
                result.append(_load_checkpoint_meta(f, legacy_dir))
        return {"checkpoints": result}

    # 0. Legacy flat checkpoint dirs (backward compatibility for older tests/tools)
    for legacy_dir in legacy_ckpt_dirs:
        if not legacy_dir.exists():
            continue
        for f in sorted(legacy_dir.glob("*.pt")):
            if f.name in seen_filenames:
                continue
            seen_filenames.add(f.name)
            result.append(_load_checkpoint_meta(f, legacy_dir))

    # 1. HOF files (manually placed, committed to git)
    hof_files = sorted(hof_dir.glob("*.pt")) if hof_dir.exists() else []
    for f in hof_files:
        try:
            meta = _torch.load(f, map_location="cpu", weights_only=False)
            model_name = meta.get("model_name") or meta.get("display_name") or f.stem
            result.append(
                {
                    "filename": f"hall_of_fame/{f.name}",
                    "persona": meta.get("persona") or f.stem,
                    "model_name": model_name,
                    "episode": meta.get("episode", 0),
                    "session": meta.get("session", 0),
                    "win_rate": meta.get("metrics", {}).get("win_rate", 0),
                    "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
                    "is_hof": True,
                }
            )
            seen_filenames.add(f.name)
        except Exception:
            result.append({"filename": f"hall_of_fame/{f.name}", "is_hof": True, "episode": 0})
            seen_filenames.add(f.name)

    # 2. Persona subdirectories — expose _current.pt for each
    for persona_dir in sorted(root_ckpt_dir.iterdir()):
        if not persona_dir.is_dir() or persona_dir.name == "hall_of_fame":
            continue
        current = persona_dir / "_current.pt"
        if not current.exists():
            continue
        persona_name = persona_dir.name
        try:
            meta = _torch.load(current, map_location="cpu", weights_only=False)
            model_name = meta.get("model_name") or meta.get("display_name") or persona_name
            result.append(
                {
                    "filename": f"{persona_name}/_current.pt",
                    "persona": persona_name,
                    "model_name": model_name,
                    "episode": meta.get("episode", 0),
                    "session": meta.get("session", 0),
                    "win_rate": meta.get("metrics", {}).get("win_rate", 0),
                    "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
                    "is_hof": False,
                }
            )
            seen_filenames.add(f"{persona_name}/_current.pt")
        except Exception:
            result.append(
                {
                    "filename": f"{persona_name}/_current.pt",
                    "persona": persona_name,
                    "is_hof": False,
                    "episode": 0,
                }
            )
            seen_filenames.add(f"{persona_name}/_current.pt")

    return {"checkpoints": result}


def _resolve_checkpoint_path(token: str) -> Path | None:
    return resolve_checkpoint(token)


# ---- Game endpoints ----


def _load_opponent(choice: str, index: int):
    """Load a single AI opponent from a choice string.

    Supports registry players (stockskis_v*, stockskis_m6, nn, human)
    and checkpoint tokens:
      - "PersonaName"             → ../data/checkpoints/PersonaName/_current.pt
      - "PersonaName/_current.pt" → ../data/checkpoints/PersonaName/_current.pt
      - "hall_of_fame/foo.pt"     → ../data/checkpoints/hall_of_fame/foo.pt
    """
    registry = get_player_factory()

    if registry.has(choice):
        return registry.create(choice, name=f"AI-{index + 1}")

    path = _resolve_checkpoint_path(choice)

    name = f"AI-{index + 1}"
    if path and path.exists():
        try:
            import torch as _torch

            meta = _torch.load(path, map_location="cpu", weights_only=False)
            if meta.get("display_name"):
                name = meta["display_name"]
            elif meta.get("model_name"):
                name = meta["model_name"]
        except Exception:
            pass
        agent = NeuralPlayer.from_checkpoint(path, name=name)
    else:
        agent = NeuralPlayer(name=name)
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
        "num_rounds": req.num_rounds if req else 1,
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
    num_rounds = game_info.get("num_rounds", 1)
    player_names = [a.name for a in agents]

    observer = WebSocketObserver(ws, player_idx=0, player_names=player_names)
    experience_logger = HumanPlayExperienceLogger()

    # Match-level state
    cumulative_scores = {i: 0 for i in range(4)}
    caller_counts = {i: 0 for i in range(4)}  # times as declarer
    called_counts = {i: 0 for i in range(4)}  # times as partner
    round_history: list[dict] = []

    async def run_match():
        nonlocal cumulative_scores
        for round_num in range(num_rounds):
            dealer = round_num % 4
            round_decisions: list[dict] = []
            game_loop = GameLoop(
                agents,
                observer=observer,
                decision_recorder=round_decisions.append,
            )

            # Send round_start event
            observer.set_match_info(
                round_num=round_num + 1,
                total_rounds=num_rounds,
                cumulative_scores=cumulative_scores,
                caller_counts=caller_counts,
                called_counts=called_counts,
                round_history=round_history,
            )

            state, scores = await game_loop.run(dealer=dealer)

            # Update match-level stats
            for p, s in scores.items():
                cumulative_scores[p] += s
            if state.declarer is not None:
                caller_counts[state.declarer] += 1
            if state.partner is not None:
                called_counts[state.partner] += 1

            round_history.append(
                {
                    "round": round_num + 1,
                    "scores": dict(scores),
                    "contract": state.contract.value if state.contract else None,
                    "declarer": state.declarer,
                    "partner": state.partner,
                }
            )

            # Persist supervised data from all seats (human and bots).
            try:
                output_path = experience_logger.write_round(
                    game_id=game_id,
                    round_num=round_num + 1,
                    player_names=player_names,
                    decisions=round_decisions,
                    scores=[int(scores.get(i, 0)) for i in range(4)],
                    contract=state.contract.name if state.contract else None,
                    declarer=state.declarer,
                    partner=state.partner,
                )
                round_history[-1]["experience_file"] = str(output_path)
                round_history[-1]["experience_steps"] = len(round_decisions)
            except Exception:
                # Never fail gameplay if experience persistence has an issue.
                round_history[-1]["experience_file"] = None
                round_history[-1]["experience_steps"] = 0

            # Send match progress after each round
            if round_num < num_rounds - 1:
                await observer.send_match_update(
                    cumulative_scores,
                    caller_counts,
                    called_counts,
                    round_history,
                    round_num + 1,
                    num_rounds,
                    state,
                )

        # Final match end
        await observer.send_match_end(
            cumulative_scores,
            caller_counts,
            called_counts,
            round_history,
            num_rounds,
            state,
        )

    game_task = asyncio.create_task(run_match())

    try:
        while True:
            data = await ws.receive_json()
            action_type = data.get("action")

            if action_type == "bid":
                contract_val = data.get("contract")
                if contract_val is None:
                    human.submit_action(None, action_type="bid")
                else:
                    rust_mapped = None
                    if isinstance(contract_val, int):
                        rust_mapped = _RUST_U8_TO_PY_CONTRACT.get(contract_val)
                    if rust_mapped is not None:
                        human.submit_action(rust_mapped, action_type="bid")
                    else:
                        human.submit_action(Contract(contract_val), action_type="bid")

            elif action_type == "call_king":
                suit = Suit(data["suit"])
                king = suit_card(suit, SuitRank.KING)
                human.submit_action(king, action_type="king")

            elif action_type == "choose_talon":
                human.submit_action(data["group_index"], action_type="talon")

            elif action_type == "discard":
                cards = [_card_from_dict(c) for c in data["cards"]]
                human.submit_action(cards, action_type="discard")

            elif action_type == "play_card":
                card = _card_from_dict(data["card"])
                human.submit_action(card, action_type="card")

            elif action_type == "set_delay":
                delay = data.get("delay", 1.0)
                observer.ai_delay = max(0.0, min(float(delay), 5.0))

            elif action_type == "reveal_hands":
                observer.reveal_hands = bool(data.get("reveal", False))

    except WebSocketDisconnect:
        game_task.cancel()
        if game_id in _active_games:
            del _active_games[game_id]


# ---- Spectator endpoints ----

_spectator_games: dict[str, dict] = {}


class SpectateRequest(BaseModel):
    agents: list[dict] = []  # [{name, type: "nn"|"stockskis_*"|"human", checkpoint?}]
    delay: float = 1.5  # seconds between moves


def _available_stockskis_versions() -> list[int]:
    return get_player_factory().stockskis_versions


def _build_stockskis_player(version: int, name: str):
    return get_player_factory().create(f"stockskis_v{version}", name=name)


def _build_spectate_agent(cfg: dict, idx: int):
    agent_type_raw = str(cfg.get("type", "rl"))
    agent_type = agent_type_raw.strip().lower()
    agent_name = str(cfg.get("name", f"Agent-{idx}"))
    checkpoint = cfg.get("checkpoint")

    registry = get_player_factory()

    # Check registry first (stockskis_*, nn, human)
    if registry.has(agent_type):
        return registry.create(agent_type, name=agent_name)

    # Bare "stockskis" → latest version
    if agent_type == "stockskis":
        versions = registry.stockskis_versions
        if not versions:
            raise HTTPException(status_code=400, detail="No StockŠkis versions available on server")
        return registry.create(f"stockskis_v{max(versions)}", name=agent_name)

    # RL agent
    ckpt_path = _resolve_checkpoint_path(checkpoint) if checkpoint else None
    if ckpt_path and ckpt_path.exists():
        agent = NeuralPlayer.from_checkpoint(ckpt_path, name=agent_name)
    else:
        agent = NeuralPlayer(name=agent_name)
    agent.set_training(False)
    return agent


@app.get("/api/agents/stockskis")
async def list_stockskis_versions():
    """List available StockŠkis heuristic versions for UI dropdowns."""
    versions = _available_stockskis_versions()
    registry = get_player_factory()
    all_types = registry.stockskis_types
    return {
        "versions": [f"v{v}" for v in versions],
        "latest": (f"v{max(versions)}" if versions else None),
        "types": all_types if all_types else [],
    }


@app.get("/api/agents")
async def list_agents():
    """List all available agent types for opponent selection.

    Returns categorised entries so the frontend can group them in the lobby
    dropdown (heuristic bots, baseline, search, neural checkpoints).
    """
    registry = get_player_factory()
    bots = registry.list_players()

    # Also include "latest" as a virtual entry for the NN checkpoint
    bots.append(
        {
            "id": "latest",
            "name": "Latest trained model",
            "description": "Most recent PPO-trained neural network checkpoint",
            "category": "neural",
            "version": None,
        }
    )

    return {"agents": bots}


@app.post("/api/spectate/new")
async def new_spectate(req: SpectateRequest):
    """Create a new 4-AI spectator game."""
    game_id = f"spectate-{len(_spectator_games)}"
    replay_name = f"{game_id}-{int(time.time())}.json"

    agents = []
    for i, agent_cfg in enumerate(req.agents[:4]):
        agents.append(_build_spectate_agent(agent_cfg, i))
        # Keep spectate robust.

    # Fill remaining slots with RL agents
    while len(agents) < 4:
        ckpt_path = _resolve_checkpoint_path("training_run") or Path(
            "../data/checkpoints/training_run/_current.pt"
        )
        if ckpt_path.exists():
            agent = NeuralPlayer.from_checkpoint(ckpt_path, name=f"Agent-{len(agents)}")
        else:
            agent = NeuralPlayer(name=f"Agent-{len(agents)}")
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
    spectators: list[WebSocket] = game_info["spectators"]
    spectators.append(ws)

    # Get player names from agents or stored names
    agents = game_info.get("agents")
    if agents:
        player_names = [a.name for a in agents]
    else:
        player_names = game_info.get("agent_names", [f"Player-{i}" for i in range(4)])

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
        game_info["observer"] = observer

        trace = game_info.get("trace")
        if trace is not None:
            # Deterministic trace replay — no agents or model inference
            game_info["game_task"] = asyncio.create_task(
                _replay_from_trace(
                    trace=trace,
                    hands=game_info["preset_hands"],
                    talon=game_info["preset_talon"],
                    dealer=game_info.get("dealer", 0),
                    observer=observer,
                )
            )
        else:
            # Live game with agents
            assert agents is not None
            game_loop = GameLoop(agents, observer=observer, allow_berac=False)
            preset_hands = game_info.get("preset_hands")
            preset_talon = game_info.get("preset_talon")
            dealer = game_info.get("dealer", 0)
            game_info["game_task"] = asyncio.create_task(
                game_loop.run(dealer=dealer, preset_hands=preset_hands, preset_talon=preset_talon)
            )

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


class ArenaReplayRequest(BaseModel):
    hands: list[list[int]]  # [[card_idx...] x 12] x 4
    talon: list[int]  # [card_idx...] x 6
    agents: list[dict]  # [{name, type, checkpoint?}] x 4
    dealer: int = 0
    delay: float = 0.6
    trace: dict | None = None  # Game trace for deterministic replay


async def _replay_from_trace(
    trace: dict,
    hands: list[list[int]],
    talon: list[int],
    dealer: int,
    observer: SpectatorObserver,
) -> None:
    """Replay a game deterministically from a recorded trace.

    No agents or model inference needed — all decisions come from the trace.
    """
    from tarok.use_cases.game_loop import _RustTrickSnapshot, _PyBid
    from tarok.use_cases.rust_state import (
        _BID_IDX_TO_RUST,
        _RUST_U8_TO_PY_CONTRACT,
        _build_py_state_from_rust,
        _is_berac,
        _talon_cards,
        _build_talon_groups,
    )
    import tarok_engine as te  # type: ignore[import-untyped]

    gs = te.RustGameState()
    gs.dealer = dealer
    gs.deal_hands(hands, talon)

    completed_tricks: list[_RustTrickSnapshot] = []
    bid_history: list[_PyBid] = []

    # game_start / deal
    await observer.on_game_start(_build_py_state_from_rust(gs, completed_tricks, bids=bid_history))
    await observer.on_deal(_build_py_state_from_rust(gs, completed_tricks, bids=bid_history))

    # === BIDDING ===
    winning_player = None
    highest = None
    for player, action_idx in trace["bids"]:
        rust_contract = _BID_IDX_TO_RUST[action_idx] if action_idx < len(_BID_IDX_TO_RUST) else None
        if rust_contract is None:
            gs.add_bid(player, None)
            bid_history.append(_PyBid(player, None))
        else:
            gs.add_bid(player, rust_contract)
            py_contract = _RUST_U8_TO_PY_CONTRACT.get(rust_contract)
            bid_history.append(_PyBid(player, py_contract))
            highest = rust_contract
            winning_player = player

        await observer.on_bid(
            bid_history[-1].player,
            bid_history[-1].contract,
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
        )

    # Resolve bidding
    contract = highest
    declarer = winning_player
    if declarer is not None and contract is not None:
        gs.declarer = declarer
        gs.contract = contract
        gs.set_role(declarer, 0)  # Declarer
        for i in range(4):
            if i != declarer:
                gs.set_role(i, 2)  # Opponent

        py_contract = _RUST_U8_TO_PY_CONTRACT.get(contract)
        await observer.on_contract_won(
            declarer,
            py_contract,
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
        )
    else:
        # Klop
        gs.contract = 0
        for i in range(4):
            gs.set_role(i, 2)
        py_contract = _RUST_U8_TO_PY_CONTRACT.get(0)
        await observer.on_contract_won(
            0,
            py_contract,
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
        )

    # === KING CALL ===
    if trace["king_call"] is not None and declarer is not None:
        kc_player, kc_action = trace["king_call"]
        # Find the actual card from callable kings
        callable_idxs = gs.callable_kings()
        # kc_action is suit index: 0=hearts, 1=diamonds, 2=clubs, 3=spades
        suit_map = {0: "hearts", 1: "diamonds", 2: "clubs", 3: "spades"}
        chosen_idx = None
        for idx in callable_idxs:
            card = DECK[idx]
            if card.suit and card.suit.value == suit_map.get(kc_action):
                chosen_idx = idx
                break
        if chosen_idx is None and callable_idxs:
            chosen_idx = callable_idxs[0]
        if chosen_idx is not None:
            gs.set_called_king(chosen_idx)
            # Find partner
            for p in range(4):
                if p != declarer:
                    hand = gs.hand(p)
                    if chosen_idx in hand:
                        gs.partner = p
                        gs.set_role(p, 1)  # Partner
                        break
            await observer.on_king_called(
                declarer,
                DECK[chosen_idx],
                _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
            )

    # === TALON EXCHANGE ===
    if trace["talon_pick"] is not None and declarer is not None and contract is not None:
        tp_player, tp_group_idx = trace["talon_pick"]
        talon_cards_count = _talon_cards(contract)
        talon_idxs = gs.talon()
        groups = _build_talon_groups(talon_idxs, talon_cards_count)
        gs.set_talon_revealed(groups)

        talon_revealed = [[DECK[idx] for idx in g] for g in groups]
        await observer.on_talon_revealed(
            talon_revealed,
            _build_py_state_from_rust(
                gs, completed_tricks, bids=bid_history, talon_revealed=groups
            ),
        )

        # Pick group
        pick_idx = min(tp_group_idx, len(groups) - 1)
        picked = groups[pick_idx]
        for card_idx in picked:
            gs.add_to_hand(declarer, card_idx)
            gs.remove_from_talon(card_idx)

        await observer.on_talon_group_picked(
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
        )

        # Discard from trace
        for card_idx in trace["put_down"]:
            gs.remove_card(declarer, card_idx)
            gs.add_put_down(card_idx)

        await observer.on_talon_exchanged(
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
            picked=[DECK[idx] for idx in picked],
            discarded=[DECK[idx] for idx in trace["put_down"]],
        )

    # === TRICK PLAY ===
    gs.phase = te.PHASE_TRICK_PLAY
    lead_player = (dealer + 1) % 4
    card_cursor = 0
    cards_played_trace = trace["cards_played"]

    for trick_num in range(12):
        lead = lead_player
        trick_cards: list[tuple[int, Card]] = []
        gs.start_trick(lead_player)
        gs.current_player = lead_player
        await observer.on_trick_start(
            _build_py_state_from_rust(
                gs, completed_tricks, bids=bid_history, current_trick=(lead, trick_cards)
            ),
        )

        for offset in range(4):
            player = (lead_player + offset) % 4
            gs.current_player = player

            if card_cursor < len(cards_played_trace):
                trace_player, card_idx = cards_played_trace[card_cursor]
                card_cursor += 1
            else:
                # Fallback
                legal = gs.legal_plays(player)
                card_idx = legal[0] if legal else 0

            gs.play_card(player, card_idx)
            trick_cards.append((player, DECK[card_idx]))
            next_player = (player + 1) % 4
            gs.current_player = next_player
            await observer.on_card_played(
                player,
                DECK[card_idx],
                _build_py_state_from_rust(
                    gs, completed_tricks, bids=bid_history, current_trick=(lead, trick_cards)
                ),
            )

        # Finish trick
        winner, points = gs.finish_trick()
        trick_snapshot = _RustTrickSnapshot(
            lead_player=lead,
            cards=trick_cards,
            winner_player=winner,
            points=points,
        )
        completed_tricks.append(trick_snapshot)
        await observer.on_trick_won(
            trick_snapshot,
            winner,
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
        )
        lead_player = winner

        # Berač early exit
        if (
            contract is not None
            and _is_berac(contract)
            and declarer is not None
            and winner == declarer
        ):
            break

    # === SCORING ===
    gs.phase = te.PHASE_SCORING
    scores_arr = gs.score_game()
    scores = {i: int(scores_arr[i]) for i in range(4)}

    from tarok.use_cases.rust_state import _build_py_state_stub

    py_state = _build_py_state_stub(
        dealer,
        py_contract,
        declarer,
        gs,
        {},
        completed_tricks,
        bid_history,
    )
    py_state.scores = scores

    # Scoring breakdown — computed entirely in Rust
    import json as _json

    breakdown = None
    try:
        raw = _json.loads(gs.score_game_breakdown_json())
        breakdown = {
            "breakdown": {
                "contract": raw.get("contract"),
                "mode": raw.get("mode"),
                "declarer_won": raw.get("declarer_won"),
                "declarer_points": raw.get("declarer_points"),
                "opponent_points": raw.get("opponent_points"),
                "lines": raw.get("lines", []),
            },
            "trick_summary": raw.get("trick_summary", []),
        }
    except Exception:
        pass

    await observer.on_game_end(scores, py_state, breakdown=breakdown)


@app.post("/api/arena/replay")
async def arena_replay(req: ArenaReplayRequest):
    """Create a spectate game replaying a specific deal from the arena."""
    if len(req.hands) != 4 or len(req.talon) != 6:
        raise HTTPException(status_code=400, detail="Need 4 hands and 6 talon cards")
    for h in req.hands:
        if len(h) != 12:
            raise HTTPException(status_code=400, detail="Each hand must have 12 cards")

    game_id = f"arena-replay-{len(_spectator_games)}"
    replay_name = f"{game_id}-{int(time.time())}.json"

    agent_names = [str(cfg.get("name", f"Agent-{i}")) for i, cfg in enumerate(req.agents[:4])]
    while len(agent_names) < 4:
        agent_names.append(f"Agent-{len(agent_names)}")

    if req.trace is not None:
        # Deterministic trace replay — no agents needed
        _spectator_games[game_id] = {
            "agents": None,
            "agent_names": agent_names,
            "delay": req.delay,
            "spectators": [],
            "game_task": None,
            "replay_name": replay_name,
            "preset_hands": req.hands,
            "preset_talon": req.talon,
            "dealer": req.dealer,
            "trace": req.trace,
        }
    else:
        # Legacy: re-simulate with agents (non-deterministic)
        agents = []
        for i, agent_cfg in enumerate(req.agents[:4]):
            agents.append(_build_spectate_agent(agent_cfg, i))

        _spectator_games[game_id] = {
            "agents": agents,
            "agent_names": agent_names,
            "delay": req.delay,
            "spectators": [],
            "game_task": None,
            "replay_name": replay_name,
            "preset_hands": req.hands,
            "preset_talon": req.talon,
            "dealer": req.dealer,
            "trace": None,
        }
    return {"game_id": game_id, "replay_name": replay_name}


@app.websocket("/ws/training")
async def training_websocket(ws: WebSocket):
    """Stream training metrics to the frontend."""
    await ws.accept()
    try:
        while True:
            if _latest_metrics:
                await ws.send_json(
                    {
                        "event": "metrics",
                        "data": _latest_metrics,
                    }
                )
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
