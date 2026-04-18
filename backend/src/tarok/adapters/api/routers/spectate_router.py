"""Spectator game, agent listing, arena replay, and replay endpoints."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tarok.adapters.api.checkpoint_utils import resolve_checkpoint
from tarok.adapters.api.spectator_observer import SpectatorObserver, list_replays, load_replay
from tarok.adapters.players.factory import get_player_factory
from tarok.adapters.players.neural_player import NeuralPlayer
from tarok.entities import DECK
from tarok.use_cases.game_loop import RustGameLoop as GameLoop

router = APIRouter(tags=["spectate"])

_spectator_games: dict[str, dict] = {}


class SpectateRequest(BaseModel):
    agents: list[dict] = []
    delay: float = 1.5


class ArenaReplayRequest(BaseModel):
    hands: list[list[int]]  # [[card_idx...] x 12] x 4
    talon: list[int]  # [card_idx...] x 6
    agents: list[dict]  # [{name, type, checkpoint?}] x 4
    dealer: int = 0
    delay: float = 0.6
    trace: dict | None = None


def _build_spectate_agent(cfg: dict, idx: int):
    agent_type_raw = str(cfg.get("type", "rl"))
    agent_type = agent_type_raw.strip().lower()
    agent_name = str(cfg.get("name", f"Agent-{idx}"))
    checkpoint = cfg.get("checkpoint")

    registry = get_player_factory()

    if registry.has(agent_type):
        return registry.create(agent_type, name=agent_name)

    if agent_type == "stockskis":
        versions = registry.stockskis_versions
        if not versions:
            raise HTTPException(status_code=400, detail="No StockŠkis versions available on server")
        return registry.create(f"stockskis_v{max(versions)}", name=agent_name)

    ckpt_path = resolve_checkpoint(checkpoint) if checkpoint else None
    if ckpt_path and ckpt_path.exists():
        agent = NeuralPlayer.from_checkpoint(ckpt_path, name=agent_name)
    else:
        agent = NeuralPlayer(name=agent_name)
    agent.set_training(False)
    return agent


@router.get("/api/agents/stockskis")
async def list_stockskis_versions():
    """List available StockŠkis heuristic versions for UI dropdowns."""
    registry = get_player_factory()
    versions = registry.stockskis_versions
    all_types = registry.stockskis_types
    return {
        "versions": [f"v{v}" for v in versions],
        "latest": (f"v{max(versions)}" if versions else None),
        "types": all_types if all_types else [],
    }


@router.get("/api/agents")
async def list_agents():
    """List all available agent types for opponent selection."""
    registry = get_player_factory()
    bots = registry.list_players()
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


@router.post("/api/spectate/new")
async def new_spectate(req: SpectateRequest):
    """Create a new 4-AI spectator game."""
    game_id = f"spectate-{len(_spectator_games)}"
    replay_name = f"{game_id}-{int(time.time())}.json"

    agents = []
    for i, agent_cfg in enumerate(req.agents[:4]):
        agents.append(_build_spectate_agent(agent_cfg, i))

    while len(agents) < 4:
        ckpt_path = resolve_checkpoint("training_run") or Path(
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


@router.websocket("/ws/spectate/{game_id}")
async def spectate_websocket(ws: WebSocket, game_id: str):
    await ws.accept()

    if game_id not in _spectator_games:
        await ws.close(code=4004, reason="Game not found")
        return

    game_info = _spectator_games[game_id]
    spectators: list[WebSocket] = game_info["spectators"]
    spectators.append(ws)

    agents = game_info.get("agents")
    if agents:
        player_names = [a.name for a in agents]
    else:
        player_names = game_info.get("agent_names", [f"Player-{i}" for i in range(4)])

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
            assert agents is not None
            game_loop = GameLoop(agents, observer=observer, allow_berac=False)
            preset_hands = game_info.get("preset_hands")
            preset_talon = game_info.get("preset_talon")
            dealer = game_info.get("dealer", 0)
            game_info["game_task"] = asyncio.create_task(
                game_loop.run(dealer=dealer, preset_hands=preset_hands, preset_talon=preset_talon)
            )

    try:
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


@router.post("/api/arena/replay")
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


@router.get("/api/replays")
async def get_replays():
    return {"replays": list_replays()}


@router.get("/api/replays/{replay_name}")
async def get_replay(replay_name: str):
    try:
        return load_replay(replay_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Replay not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _replay_from_trace(
    trace: dict,
    hands: list[list[int]],
    talon: list[int],
    dealer: int,
    observer: SpectatorObserver,
) -> None:
    """Replay a game deterministically from a recorded trace."""
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
        gs.set_role(declarer, 0)
        for i in range(4):
            if i != declarer:
                gs.set_role(i, 2)
        py_contract = _RUST_U8_TO_PY_CONTRACT.get(contract)
        await observer.on_contract_won(
            declarer,
            py_contract,
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
        )
    else:
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
        callable_idxs = gs.callable_kings()
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
            for p in range(4):
                if p != declarer:
                    hand = gs.hand(p)
                    if chosen_idx in hand:
                        gs.partner = p
                        gs.set_role(p, 1)
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
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history, talon_revealed=groups),
        )

        pick_idx = min(tp_group_idx, len(groups) - 1)
        picked = groups[pick_idx]
        for card_idx in picked:
            gs.add_to_hand(declarer, card_idx)
            gs.remove_from_talon(card_idx)

        await observer.on_talon_group_picked(
            _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
        )

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
        trick_cards: list[tuple[int, object]] = []
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
                legal = gs.legal_plays(player)
                card_idx = legal[0] if legal else 0

            gs.play_card(player, card_idx)
            trick_cards.append((player, DECK[card_idx]))
            gs.current_player = (player + 1) % 4
            await observer.on_card_played(
                player,
                DECK[card_idx],
                _build_py_state_from_rust(
                    gs, completed_tricks, bids=bid_history, current_trick=(lead, trick_cards)
                ),
            )

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
    import json as _json

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
