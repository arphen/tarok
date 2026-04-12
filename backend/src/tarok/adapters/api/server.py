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
from tarok.adapters.ai.bot_registry import get_registry
from tarok.adapters.ai.random_agent import RandomPlayer
from tarok.adapters.ai.training_lab import PPOTrainer, TrainingMetrics
from tarok.adapters.api.human_player import HumanPlayer
from tarok.adapters.api.spectator_observer import SpectatorObserver, list_replays, load_replay
from tarok.adapters.api.ws_observer import WebSocketObserver
from tarok.adapters.api.schemas import (
    NewGameRequest,
    TrainingRequest,
    TrainingMetricsSchema,
    LabTrainingRequest,
)
from tarok.entities.card import Card, CardType, Suit, SuitRank, DECK
from tarok.entities.game_state import Bid, Contract, GameState, Phase, PlayerRole, Trick
from tarok.use_cases.game_loop import GameLoop

# --- Globals managed by lifespan ---
_trainer: PPOTrainer | None = None
_training_task: asyncio.Task | None = None
_latest_metrics: TrainingMetrics | None = None
_active_games: dict[str, dict] = {}

# --- Training-lab (GPU lab) mode ---
_lab_runner = None      # RunPPOTraining instance
_lab_thread = None      # background thread running the training loop
_lab_sink = None        # DashboardMetricsSink for metrics polling


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
        lookahead_ratio=req.lookahead_ratio,
        lookahead_sims=req.lookahead_sims,
        lookahead_perfect_info=req.lookahead_perfect_info,
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
        except asyncio.CancelledError:
            import logging
            logging.getLogger(__name__).info("Training task cancelled")
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Training task failed")
            # Preserve the last good metrics so the dashboard shows
            # what happened before the crash instead of going blank.
            if _latest_metrics is None:
                _latest_metrics = _trainer.metrics if _trainer else None

    _training_task = asyncio.create_task(run_training())
    return {"status": "started", "num_sessions": req.num_sessions, "games_per_session": req.games_per_session,
            "message": "Run ID will appear in metrics once training begins"}


@app.post("/api/training/stop")
async def stop_training():
    global _trainer, _training_task, _lab_runner, _lab_thread
    if _trainer:
        _trainer.stop()
    if _lab_runner:
        _lab_runner.stop()
    return {"status": "stopped"}


@app.get("/api/training/metrics")
async def get_metrics() -> dict:
    # Prefer lab metrics if lab mode is active
    if _lab_sink is not None:
        return _lab_sink.snapshot()
    if _latest_metrics:
        return _latest_metrics.to_dict()
    return TrainingMetrics().to_dict()


@app.get("/api/training/status")
async def training_status():
    # Check both old and lab training
    old_running = _training_task is not None and not _training_task.done()
    lab_running = _lab_thread is not None and _lab_thread.is_alive()
    return {"running": old_running or lab_running, "mode": "lab" if lab_running else ("legacy" if old_running else "idle")}


# ---- Training Lab (GPU) endpoints ----

@app.post("/api/training/lab/start")
async def start_lab_training(req: LabTrainingRequest):
    global _lab_runner, _lab_thread, _lab_sink, _trainer, _training_task

    # Don't start if anything is already running
    if _training_task and not _training_task.done():
        return {"status": "already_running", "mode": "legacy"}
    if _lab_thread and _lab_thread.is_alive():
        return {"status": "already_running", "mode": "lab"}

    # Stop old trainer if lingering
    if _trainer:
        _trainer.stop()
        _trainer = None

    from tarok.adapters.ai.lab_bridge import start_lab_training as _start_lab
    import threading

    # Resolve checkpoint path
    checkpoint_path = None
    if req.resume_from:
        from pathlib import Path
        cp = Path("checkpoints") / req.resume_from
        if cp.exists():
            checkpoint_path = str(cp)

    runner, sink = _start_lab(
        checkpoint_path=checkpoint_path,
        config_overrides={
            "num_sessions": req.num_sessions,
            "games_per_session": req.games_per_session,
            "learning_rate": req.learning_rate,
            "hidden_size": req.hidden_size,
            "concurrency": req.concurrency,
            "buffer_capacity": req.buffer_capacity,
            "min_experiences": req.min_experiences,
            "ppo_epochs": req.ppo_epochs,
            "batch_size": req.batch_size,
            "explore_rate": req.explore_rate,
            "checkpoint_interval": req.checkpoint_interval,
            "device": req.device,
        },
    )

    _lab_runner = runner
    _lab_sink = sink

    def _run():
        import logging
        try:
            runner.run()
        except Exception:
            logging.getLogger(__name__).exception("Lab training failed")

    _lab_thread = threading.Thread(target=_run, name="lab-training", daemon=True)
    _lab_thread.start()

    return {
        "status": "started",
        "mode": "lab",
        "num_sessions": req.num_sessions,
        "device": req.device,
    }


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
    """List all saved checkpoint files (training and HOF)."""
    ckpt_dir = Path("checkpoints")
    hof_dir = Path("checkpoints/hall_of_fame")
    root_ckpt_dir = Path("../checkpoints")
    
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
                "is_hof": False,
            })
        except Exception:
            result.append({"filename": "tarok_agent_latest.pt", "episode": 0, "is_hof": False})

    if not ckpt_dir.exists():
        return {"checkpoints": result}
    files = [f for f in ckpt_dir.glob("tarok_agent_*.pt") if f.name != "tarok_agent_latest.pt"]
    if hof_dir.exists():
        files.extend([f for f in hof_dir.glob("hof_*.pt")])

    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

    # Also scan root-level checkpoints/ directory
    if root_ckpt_dir.exists():
        seen_names = {f.name for f in files}
        seen_names.add("tarok_agent_latest.pt")  # already handled above
        root_files = sorted(
            [f for f in root_ckpt_dir.glob("*.pt") if f.name not in seen_names],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        files.extend(root_files)

    for f in files:
        import torch as _torch
        try:
            meta = _torch.load(f, map_location="cpu", weights_only=False)
            model_name = meta.get("model_name", None) or meta.get("display_name", None)
            is_hof = "hof_" in f.name
            
            result.append({
                "filename": str(f.relative_to(ckpt_dir)) if ckpt_dir in f.parents or f.parent == ckpt_dir else f.name,
                "episode": meta.get("episode", 0),
                "session": meta.get("session", 0),
                "win_rate": meta.get("metrics", {}).get("win_rate", 0),
                "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
                "model_name": model_name,
                "is_hof": is_hof,
                "persona": meta.get("persona") if is_hof else None,
            })
        except Exception:
            result.append({
                "filename": str(f.relative_to(ckpt_dir)) if ckpt_dir in f.parents or f.parent == ckpt_dir else f.name,
                "episode": 0,
                "is_hof": "hof_" in f.name,
            })
    return {"checkpoints": result}


# ---- Game endpoints ----

def _load_opponent(choice: str, index: int):
    """Load a single AI opponent from a choice string.

    Supports registry bots (random, stockskis_v1..v5, lookahead, etc.)
    and checkpoint filenames for RL agents.
    """
    registry = get_registry()

    # Check registry first (stockskis_v1..v5, random, lookahead, etc.)
    if registry.has(choice):
        return registry.create(choice, name=f"AI-{index + 1}")

    # RL checkpoint path
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

    # Match-level state
    cumulative_scores = {i: 0 for i in range(4)}
    caller_counts = {i: 0 for i in range(4)}   # times as declarer
    called_counts = {i: 0 for i in range(4)}    # times as partner
    round_history: list[dict] = []

    async def run_match():
        nonlocal cumulative_scores
        for round_num in range(num_rounds):
            dealer = round_num % 4
            game_loop = GameLoop(agents, observer=observer)

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

            round_history.append({
                "round": round_num + 1,
                "scores": dict(scores),
                "contract": state.contract.value if state.contract else None,
                "declarer": state.declarer,
                "partner": state.partner,
            })

            # Send match progress after each round
            if round_num < num_rounds - 1:
                await observer.send_match_update(
                    cumulative_scores, caller_counts, called_counts,
                    round_history, round_num + 1, num_rounds, state,
                )

        # Final match end
        await observer.send_match_end(
            cumulative_scores, caller_counts, called_counts,
            round_history, num_rounds, state,
        )

    game_task = asyncio.create_task(run_match())

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

            elif action_type == "reveal_hands":
                observer.reveal_hands = bool(data.get("reveal", False))

    except WebSocketDisconnect:
        game_task.cancel()
        if game_id in _active_games:
            del _active_games[game_id]


# ---- Spectator endpoints ----

_spectator_games: dict[str, dict] = {}


class SpectateRequest(BaseModel):
    agents: list[dict] = []  # [{name, type: "rl"|"random", checkpoint?}]
    delay: float = 1.5  # seconds between moves


def _available_stockskis_versions() -> list[int]:
    return get_registry().stockskis_versions


def _build_stockskis_player(version: int, name: str):
    return get_registry().create(f"stockskis_v{version}", name=name)


def _build_spectate_agent(cfg: dict, idx: int):
    agent_type_raw = str(cfg.get("type", "rl"))
    agent_type = agent_type_raw.strip().lower()
    agent_name = str(cfg.get("name", f"Agent-{idx}"))
    checkpoint = cfg.get("checkpoint")

    registry = get_registry()

    # Check registry first (random, stockskis_v*, lookahead, etc.)
    if registry.has(agent_type):
        return registry.create(agent_type, name=agent_name)

    # Bare "stockskis" → latest version
    if agent_type == "stockskis":
        versions = registry.stockskis_versions
        if not versions:
            raise HTTPException(status_code=400, detail="No StockŠkis versions available on server")
        return registry.create(f"stockskis_v{max(versions)}", name=agent_name)

    # RL agent
    ckpt_path = Path("checkpoints") / checkpoint if checkpoint else Path("checkpoints/tarok_agent_latest.pt")
    if ckpt_path.exists():
        agent = RLAgent.from_checkpoint(ckpt_path, name=agent_name)
    else:
        agent = RLAgent(name=agent_name)
    agent.set_training(False)
    return agent


@app.get("/api/agents/stockskis")
async def list_stockskis_versions():
    """List available StockŠkis heuristic versions for UI dropdowns."""
    versions = _available_stockskis_versions()
    return {
        "versions": [f"v{v}" for v in versions],
        "latest": (f"v{max(versions)}" if versions else None),
        "types": ([f"stockskis_v{v}" for v in versions] if versions else []),
    }


@app.get("/api/agents")
async def list_agents():
    """List all available agent types for opponent selection.

    Returns categorised entries so the frontend can group them in the lobby
    dropdown (heuristic bots, baseline, search, neural checkpoints).
    """
    registry = get_registry()
    bots = registry.list_bots()

    # Also include "latest" as a virtual entry for the NN checkpoint
    bots.append({
        "id": "latest",
        "name": "Latest trained model",
        "description": "Most recent PPO-trained neural network checkpoint",
        "category": "neural",
        "version": None,
    })

    return {"agents": bots}


@app.post("/api/spectate/new")
async def new_spectate(req: SpectateRequest):
    """Create a new 4-AI spectator game."""
    game_id = f"spectate-{len(_spectator_games)}"
    replay_name = f"{game_id}-{int(time.time())}.json"

    agents = []
    for i, agent_cfg in enumerate(req.agents[:4]):
        agents.append(_build_spectate_agent(agent_cfg, i))
        # Note: Lookahead is intentionally not supported here unless a concrete
        # implementation exists in this codebase. Keep spectate robust.

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


# ---- Tournament batch endpoint ----

class TournamentMatchRequest(BaseModel):
    agents: list[dict]  # [{name, type, checkpoint?}] — exactly 4
    num_games: int = 5


def _build_agent(cfg: dict, idx: int):
    """Instantiate a single agent from a config dict."""
    agent_type = str(cfg.get("type", "rl")).strip().lower()
    agent_name = cfg.get("name", f"Agent-{idx}")
    checkpoint = cfg.get("checkpoint")

    registry = get_registry()

    # Check registry first (random, stockskis_v*, lookahead, etc.)
    if registry.has(agent_type):
        return registry.create(agent_type, name=agent_name)

    # Bare "stockskis" → latest version
    if agent_type == "stockskis":
        versions = registry.stockskis_versions
        if versions:
            return registry.create(f"stockskis_v{max(versions)}", name=agent_name)

    # RL agent (checkpoint)
    ckpt_path = None
    if checkpoint:
        candidate = Path("checkpoints") / checkpoint
        root_candidate = Path("../checkpoints") / checkpoint
        if candidate.exists():
            ckpt_path = candidate
        elif root_candidate.exists():
            ckpt_path = root_candidate
    else:
        default = Path("checkpoints/tarok_agent_latest.pt")
        if default.exists():
            ckpt_path = default

    if ckpt_path:
        agent = RLAgent.from_checkpoint(ckpt_path, name=agent_name)
    else:
        agent = RLAgent(name=agent_name)
    agent.set_training(False)
    return agent


@app.post("/api/tournament/match")
async def tournament_match(req: TournamentMatchRequest):
    """Run N games between 4 agents and return cumulative scores."""
    agents = [_build_agent(cfg, i) for i, cfg in enumerate(req.agents[:4])]
    while len(agents) < 4:
        agents.append(RandomPlayer(name=f"Fill-{len(agents)}"))

    cumulative: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    game_results: list[dict[str, int]] = []

    for _ in range(max(1, min(req.num_games, 100))):
        loop = GameLoop(agents)
        _state, scores = await loop.run()
        for pid, pts in scores.items():
            cumulative[pid] += pts
        game_results.append({str(k): v for k, v in scores.items()})

    ranked = sorted(cumulative.items(), key=lambda x: x[1], reverse=True)
    return {
        "cumulative": {str(k): v for k, v in cumulative.items()},
        "ranked": [{"seat": k, "name": agents[k].name, "score": v} for k, v in ranked],
        "games": game_results,
    }


# ---- Multi-tournament simulation ----

class MultiTournamentRequest(BaseModel):
    agents: list[dict]  # [{name, type, checkpoint?}] — 4-8 entries
    num_tournaments: int = 5
    games_per_round: int = 5


_multi_tournament_task: asyncio.Task | None = None
_multi_tournament_progress: dict | None = None


def _run_single_bracket(entries: list[dict]) -> list[list[dict]]:
    """Build a bracket: return list of match groups (each group = 4 agent configs).

    Shuffle entries, pad to 8, return the 6 match groups in order.
    The bracket advancement is done server-side for the simulation.
    """
    import random as _random

    padded = list(entries)
    while len(padded) < 8:
        padded.append({"name": f"Random-{len(padded)}", "type": "random"})
    _random.shuffle(padded)
    return padded  # type: ignore


async def _simulate_single_tournament(
    agent_configs: list[dict],
    games_per_round: int,
) -> dict[str, dict]:
    """Run one double-elimination tournament, return per-agent placement stats."""
    import random as _random

    padded = list(agent_configs)
    while len(padded) < 8:
        padded.append({"name": f"Random-{len(padded)}", "type": "random"})
    _random.shuffle(padded)

    # Match schedule: same as frontend bracket
    # WB R1-A: slots 0-3, WB R1-B: slots 4-7
    async def _play_match(cfgs: list[dict]) -> list[dict]:
        """Play a match, return configs ranked best→worst."""
        agents = [_build_agent(c, i) for i, c in enumerate(cfgs[:4])]
        while len(agents) < 4:
            agents.append(RandomPlayer(name=f"Fill-{len(agents)}"))

        cumulative: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for _ in range(max(1, min(games_per_round, 100))):
            loop = GameLoop(agents)
            _state, scores = await loop.run()
            for pid, pts in scores.items():
                cumulative[pid] += pts
            await asyncio.sleep(0)

        ranked_seats = sorted(cumulative, key=lambda s: cumulative[s], reverse=True)
        return [cfgs[s] for s in ranked_seats]

    # WB R1
    wb_r1_a = await _play_match(padded[0:4])
    wb_r1_b = await _play_match(padded[4:8])

    wb_r1_top = wb_r1_a[:2] + wb_r1_b[:2]  # advance
    wb_r1_bot = wb_r1_a[2:] + wb_r1_b[2:]  # drop to losers

    # LB R1
    lb_r1 = await _play_match(wb_r1_bot)
    lb_r1_top = lb_r1[:2]
    # lb_r1[2:] eliminated

    # WB Final
    wb_final = await _play_match(wb_r1_top)
    wb_final_top = wb_final[:2]
    wb_final_bot = wb_final[2:]

    # LB Final
    lb_final_entries = lb_r1_top + wb_final_bot
    lb_final = await _play_match(lb_final_entries)
    lb_final_top = lb_final[:2]

    # Grand Final
    gf_entries = wb_final_top + lb_final_top
    gf_ranked = await _play_match(gf_entries)

    # Assign placements (1st=champion, 2nd, 3rd, 4th from GF; 5-6 from LB final losers; 7-8 from LB R1 losers)
    placements: dict[str, int] = {}
    for i, cfg in enumerate(gf_ranked):
        placements[cfg["name"]] = i + 1
    for i, cfg in enumerate(lb_final[2:]):
        placements[cfg["name"]] = 5 + i
    for i, cfg in enumerate(lb_r1[2:]):
        placements[cfg["name"]] = 7 + i

    return placements


@app.post("/api/tournament/simulate")
async def simulate_multi_tournament(req: MultiTournamentRequest):
    """Run multiple tournaments and return aggregate standings."""
    global _multi_tournament_task, _multi_tournament_progress

    if _multi_tournament_task and not _multi_tournament_task.done():
        return {"status": "already_running"}

    num = max(1, min(req.num_tournaments, 100))
    agent_configs = req.agents[:8]

    _multi_tournament_progress = {
        "status": "running",
        "current": 0,
        "total": num,
        "standings": {},
    }

    async def _run():
        global _multi_tournament_progress
        # Per-agent stats: {name: {wins, top2, total_placement, placements: [...]}}
        standings: dict[str, dict] = {}
        for cfg in agent_configs:
            standings[cfg["name"]] = {
                "name": cfg["name"],
                "type": cfg.get("type", "rl"),
                "checkpoint": cfg.get("checkpoint", ""),
                "wins": 0,
                "top2": 0,
                "top4": 0,
                "total_placement": 0,
                "tournaments_played": 0,
                "placements": [],
            }
        # Also init for random fillers
        for i in range(len(agent_configs), 8):
            fill_name = f"Random-{i}"
            standings[fill_name] = {
                "name": fill_name,
                "type": "random",
                "checkpoint": "",
                "wins": 0,
                "top2": 0,
                "top4": 0,
                "total_placement": 0,
                "tournaments_played": 0,
                "placements": [],
            }

        try:
            for t in range(num):
                try:
                    placements = await _simulate_single_tournament(agent_configs, req.games_per_round)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Tournament %d/%d failed, skipping", t + 1, num)
                    _multi_tournament_progress = {
                        "status": "running",
                        "current": t + 1,
                        "total": num,
                        "standings": {
                            name: {**s, "avg_placement": round(s["total_placement"] / max(s["tournaments_played"], 1), 2)}
                            for name, s in standings.items()
                        },
                    }
                    await asyncio.sleep(0)
                    continue

                for name, place in placements.items():
                    if name not in standings:
                        continue
                    s = standings[name]
                    s["tournaments_played"] += 1
                    s["total_placement"] += place
                    s["placements"].append(place)
                    if place == 1:
                        s["wins"] += 1
                    if place <= 2:
                        s["top2"] += 1
                    if place <= 4:
                        s["top4"] += 1

                _multi_tournament_progress = {
                    "status": "running",
                    "current": t + 1,
                    "total": num,
                    "standings": {
                        name: {**s, "avg_placement": round(s["total_placement"] / max(s["tournaments_played"], 1), 2)}
                        for name, s in standings.items()
                    },
                }
                await asyncio.sleep(0)

            # Final result
            final_standings = {
                name: {**s, "avg_placement": round(s["total_placement"] / max(s["tournaments_played"], 1), 2)}
                for name, s in standings.items()
            }
            _multi_tournament_progress = {
                "status": "done",
                "current": num,
                "total": num,
                "standings": final_standings,
            }
        except asyncio.CancelledError:
            import logging
            logging.getLogger(__name__).info("Multi-tournament cancelled")
            _multi_tournament_progress["status"] = "cancelled"
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Multi-tournament failed")
            _multi_tournament_progress["status"] = "error"

    _multi_tournament_task = asyncio.create_task(_run())
    return {"status": "started", "num_tournaments": num}


@app.get("/api/tournament/simulate/progress")
async def multi_tournament_progress():
    if _multi_tournament_progress:
        return _multi_tournament_progress
    return {"status": "idle", "current": 0, "total": 0, "standings": {}}


@app.post("/api/tournament/simulate/stop")
async def stop_multi_tournament():
    global _multi_tournament_task
    if _multi_tournament_task and not _multi_tournament_task.done():
        _multi_tournament_task.cancel()
    return {"status": "stopped"}


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
        expert_source=req.get("expert_source", "v2v3v5"),
        eval_bots=req.get("eval_bots", ["v1", "v2", "v3", "v5"]),
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
        eval_bots=req.get("eval_bots", ["v1", "v2", "v3", "v5"]),
        eval_interval=req.get("eval_interval", 5),
        learning_rate=req.get("learning_rate", 3e-4),
        stockskis_ratio=req.get("stockskis_ratio", 0.0),
        fsp_ratio=req.get("fsp_ratio", 0.3),
        hof_ratio=req.get("hof_ratio", 0.0),
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

    Auto-configures self-play for continuous training.
    Evaluates periodically against V1/V2/V3 and saves snapshots.
    Just hit Stop when you wake up.
    """
    from tarok.adapters.ai.training_lab import start_self_play, _lab

    if _lab.running:
        return {"status": "already_running"}

    if _lab.network is None:
        from tarok.adapters.ai.training_lab import create_lab_network
        create_lab_network(req.get("hidden_size", 256))

    await start_self_play(
        num_sessions=req.get("num_sessions", 10000),
        games_per_session=req.get("games_per_session", 20),
        eval_games=req.get("eval_games", 200),
        eval_interval=req.get("eval_interval", 10),
        learning_rate=req.get("learning_rate", 3e-4),
        stockskis_ratio=req.get("stockskis_ratio", 0.0),
        fsp_ratio=req.get("fsp_ratio", 0.3),
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


@app.delete("/api/lab/hof/{model_hash}")
async def lab_hof_remove(model_hash: str):
    """Remove a model from the Hall of Fame."""
    from tarok.adapters.ai.training_lab import remove_from_hof
    removed = remove_from_hof(model_hash)
    if not removed:
        from starlette.responses import JSONResponse
        return JSONResponse({"error": "Model not found"}, status_code=404)
    return {"ok": True}


@app.post("/api/lab/hof/promote")
async def lab_hof_promote(body: dict):
    """Promote a checkpoint to the Hall of Fame."""
    from tarok.adapters.ai.training_lab import promote_checkpoint_to_hof
    filename = body.get("filename", "")
    if not filename:
        from starlette.responses import JSONResponse
        return JSONResponse({"error": "filename required"}, status_code=400)
    info = promote_checkpoint_to_hof(filename)
    if info is None:
        from starlette.responses import JSONResponse
        return JSONResponse({"error": "Invalid or missing checkpoint"}, status_code=400)
    return {"ok": True, **info}


@app.post("/api/lab/hof/{model_hash}/pin")
async def lab_hof_pin(model_hash: str):
    """Pin a HoF model (exempt from auto-eviction)."""
    from tarok.adapters.ai.training_lab import pin_hof
    if not pin_hof(model_hash):
        from starlette.responses import JSONResponse
        return JSONResponse({"error": "Model not found"}, status_code=404)
    return {"ok": True}


@app.post("/api/lab/hof/{model_hash}/unpin")
async def lab_hof_unpin(model_hash: str):
    """Unpin a HoF model (subject to auto-eviction again)."""
    from tarok.adapters.ai.training_lab import unpin_hof
    if not unpin_hof(model_hash):
        from starlette.responses import JSONResponse
        return JSONResponse({"error": "Model not found"}, status_code=404)
    return {"ok": True}


# ---- Bot Arena: Mass analytics simulation ----

class ArenaRequest(BaseModel):
    agents: list[dict]  # [{name, type, checkpoint?}] — exactly 4
    total_games: int = 100000
    session_size: int = 50  # games per session for progress tracking


_arena_task: asyncio.Task | None = None
_arena_progress: dict | None = None


# Contract name lookup (must match engine-rs Contract enum order)
_ARENA_CONTRACT_NAMES = [
    "KLOP", "THREE", "TWO", "ONE",
    "SOLO_THREE", "SOLO_TWO", "SOLO_ONE", "SOLO",
    "BERAC", "BARVNI_VALAT",
]


def _agent_type_to_seat_label(agent_type: str) -> str | None:
    """Map frontend agent type string to Rust seat_config label.

    Returns None if the type is not supported by the Rust arena engine.
    """
    t = agent_type.strip().lower()
    if t in ("stockskis", "stockskis_v5"):
        return "bot_v5"
    if t == "stockskis_v6":
        return "bot_v6"
    return None


@app.post("/api/arena/start")
async def start_arena(req: ArenaRequest):
    """Run a large-scale bot arena via the Rust engine."""
    global _arena_task, _arena_progress

    if _arena_task and not _arena_task.done():
        return {"status": "already_running"}

    total = max(1, min(req.total_games, 500_000))
    session_size = max(1, min(req.session_size, 1000))
    agent_configs = req.agents[:4]
    while len(agent_configs) < 4:
        agent_configs.append({"name": f"StockŠkis-{len(agent_configs)}", "type": "stockskis"})

    # Validate all agents are bot types supported by Rust arena
    seat_labels = []
    agent_names = []
    agent_types_raw = []
    for i, cfg in enumerate(agent_configs):
        atype = str(cfg.get("type", "stockskis")).strip().lower()
        aname = cfg.get("name", f"Agent-{i}")
        label = _agent_type_to_seat_label(atype)
        if label is None:
            return {
                "status": "error",
                "detail": f"Seat {i} type '{atype}' not supported. Use stockskis, stockskis_v5, or stockskis_v6.",
            }
        seat_labels.append(label)
        agent_names.append(aname)
        agent_types_raw.append(atype)

    seat_config = ",".join(seat_labels)

    _arena_progress = {
        "status": "running",
        "games_done": 0,
        "total_games": total,
        "analytics": None,
    }

    async def _run_arena():
        global _arena_progress
        import logging
        log = logging.getLogger(__name__)

        try:
            import tarok_engine as te
        except ImportError:
            log.error("tarok_engine not available — cannot run arena")
            _arena_progress["status"] = "error"
            return

        import numpy as np

        # Per-player accumulators
        player_stats = []
        for i in range(4):
            player_stats.append({
                "name": agent_names[i],
                "type": agent_types_raw[i],
                "total_score": 0,
                "games_played": 0,
                "placements": {1: 0, 2: 0, 3: 0, 4: 0},
                "placement_sum": 0.0,
                "sessions_played": 0,
                "wins": 0.0,
                "positive_games": 0,
                "declared_count": 0,
                "declared_won": 0,
                "defended_count": 0,
                "defended_won": 0,
                "best_game_score": None,
                "worst_game_score": None,
                "best_game_idx": None,
                "worst_game_idx": None,
                "score_history": [],
            })
        contract_stats: dict = {}
        games_done = 0

        # Run in batches for progress updates
        batch_size = min(10_000, total)

        try:
            while games_done < total:
                n_batch = min(batch_size, total - games_done)

                # Run entire batch in Rust (GIL released, Rayon parallel)
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda n=n_batch: te.run_arena_games(
                        n_games=n,
                        seat_config=seat_config,
                    ),
                )

                scores = np.asarray(result["scores"])      # (n_batch, 4) int32
                contracts = np.asarray(result["contracts"]) # (n_batch,) uint8
                declarers = np.asarray(result["declarers"]) # (n_batch,) int8
                partners = np.asarray(result["partners"])   # (n_batch,) int8

                # Process: extract per-game stats
                for g in range(n_batch):
                    game_idx = games_done + g
                    sc = scores[g]
                    contract_u8 = int(contracts[g])
                    decl = int(declarers[g])
                    part = int(partners[g])
                    contract_name = _ARENA_CONTRACT_NAMES[contract_u8] if contract_u8 < len(_ARENA_CONTRACT_NAMES) else "UNKNOWN"

                    for pid in range(4):
                        s = int(sc[pid])
                        ps = player_stats[pid]
                        ps["total_score"] += s
                        ps["games_played"] += 1
                        if s > 0:
                            ps["positive_games"] += 1
                        if ps["best_game_score"] is None or s > ps["best_game_score"]:
                            ps["best_game_score"] = s
                            ps["best_game_idx"] = game_idx
                        if ps["worst_game_score"] is None or s < ps["worst_game_score"]:
                            ps["worst_game_score"] = s
                            ps["worst_game_idx"] = game_idx

                    # Declarer/defender tracking
                    if decl >= 0 and contract_name != "KLOP":
                        decl_team = {decl}
                        if part >= 0:
                            decl_team.add(part)
                        decl_won = int(sc[decl]) > 0

                        for pid in range(4):
                            if pid in decl_team:
                                player_stats[pid]["declared_count"] += 1
                                if decl_won:
                                    player_stats[pid]["declared_won"] += 1
                            else:
                                player_stats[pid]["defended_count"] += 1
                                if not decl_won:
                                    player_stats[pid]["defended_won"] += 1

                    # Contract stats
                    if contract_name not in contract_stats:
                        contract_stats[contract_name] = {
                            "played": 0, "decl_won": 0,
                            "total_decl_score": 0, "total_def_score": 0,
                        }
                    cs = contract_stats[contract_name]
                    cs["played"] += 1
                    if decl >= 0 and contract_name != "KLOP":
                        if int(sc[decl]) > 0:
                            cs["decl_won"] += 1
                        cs["total_decl_score"] += int(sc[decl])
                        decl_team = {decl}
                        if part >= 0:
                            decl_team.add(part)
                        def_sc = sum(int(sc[p]) for p in range(4) if p not in decl_team)
                        cs["total_def_score"] += def_sc

                # Session-level placement from this batch
                num_sessions_in_batch = max(1, n_batch // session_size)
                for s_idx in range(num_sessions_in_batch):
                    start = s_idx * session_size
                    end = min(start + session_size, n_batch)
                    if start >= n_batch:
                        break
                    cumulative = [0, 0, 0, 0]
                    for g in range(start, end):
                        for pid in range(4):
                            cumulative[pid] += int(scores[g, pid])

                    # Rank with proper tie handling
                    ranked = sorted(range(4), key=lambda p: cumulative[p], reverse=True)
                    places = [0.0] * 4
                    i = 0
                    while i < 4:
                        j = i
                        while j < 4 and cumulative[ranked[j]] == cumulative[ranked[i]]:
                            j += 1
                        avg_rank = (i + 1 + j) / 2
                        for k in range(i, j):
                            places[k] = avg_rank
                        i = j

                    for rank_idx, pid in enumerate(ranked):
                        ps = player_stats[pid]
                        place_f = places[rank_idx]
                        place_int = max(1, min(4, round(place_f)))
                        ps["placements"][place_int] += 1
                        ps["placement_sum"] += place_f
                        ps["sessions_played"] += 1
                        # Credit fractional wins for ties at 1st place
                        if place_f <= 1.0:
                            ps["wins"] += 1.0
                        elif place_f <= 1.5:
                            # Two-way tie for 1st: each gets 0.5
                            num_tied = sum(1 for p2 in places if p2 == place_f)
                            ps["wins"] += 1.0 / num_tied

                    for pid in range(4):
                        player_stats[pid]["score_history"].append(cumulative[pid])

                games_done += n_batch

                analytics = _build_arena_analytics(player_stats, contract_stats, games_done, total)
                _arena_progress = {
                    "status": "running",
                    "games_done": games_done,
                    "total_games": total,
                    "analytics": analytics,
                }
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            log.info("Arena cancelled at game %d/%d", games_done, total)
            _arena_progress["status"] = "cancelled"
            _arena_progress["analytics"] = _build_arena_analytics(player_stats, contract_stats, games_done, total)
            return
        except Exception:
            log.exception("Arena failed at game %d/%d", games_done, total)
            _arena_progress["status"] = "error"
            _arena_progress["analytics"] = _build_arena_analytics(player_stats, contract_stats, games_done, total)
            return

        _arena_progress = {
            "status": "done",
            "games_done": games_done,
            "total_games": total,
            "analytics": _build_arena_analytics(player_stats, contract_stats, games_done, total),
        }

    _arena_task = asyncio.create_task(_run_arena())
    return {"status": "started", "total_games": total, "session_size": session_size}


def _build_arena_analytics(player_stats, contract_stats, games_done, total_games):
    """Build the analytics payload from accumulated stats."""
    players = []
    for ps in player_stats:
        gp = max(ps["games_played"], 1)
        sp = max(ps["sessions_played"], 1)
        players.append({
            "name": ps["name"],
            "type": ps["type"],
            "games_played": ps["games_played"],
            "sessions_played": ps["sessions_played"],
            "total_score": ps["total_score"],
            "avg_score": round(ps["total_score"] / gp, 2),
            "placements": ps["placements"],
            "avg_placement": round(ps["placement_sum"] / sp, 2),
            "win_rate": round(ps["wins"] / sp * 100, 2),
            "positive_rate": round(ps["positive_games"] / gp * 100, 2),
            "bids_made": {},
            "declared_count": ps["declared_count"],
            "declared_won": ps["declared_won"],
            "declared_win_rate": round(ps["declared_won"] / max(ps["declared_count"], 1) * 100, 2),
            "defended_count": ps["defended_count"],
            "defended_won": ps["defended_won"],
            "defended_win_rate": round(ps["defended_won"] / max(ps["defended_count"], 1) * 100, 2),
            "announcements_made": {},
            "kontra_count": 0,
            "best_game": {"score": ps["best_game_score"], "game_idx": ps["best_game_idx"]},
            "worst_game": {"score": ps["worst_game_score"], "game_idx": ps["worst_game_idx"]},
            "score_history": ps["score_history"],
        })

    contracts = {}
    for name, cs in contract_stats.items():
        played = max(cs["played"], 1)
        contracts[name] = {
            "played": cs["played"],
            "decl_win_rate": round(cs["decl_won"] / played * 100, 2),
            "avg_decl_score": round(cs["total_decl_score"] / played, 2),
            "avg_def_score": round(cs["total_def_score"] / played, 2),
        }

    return {
        "games_done": games_done,
        "total_games": total_games,
        "players": players,
        "contracts": contracts,
    }


@app.get("/api/arena/progress")
async def arena_progress():
    if _arena_progress:
        return _arena_progress
    return {"status": "idle", "games_done": 0, "total_games": 0, "analytics": None}


@app.post("/api/arena/stop")
async def stop_arena():
    global _arena_task
    if _arena_task and not _arena_task.done():
        _arena_task.cancel()
    return {"status": "stopped"}
