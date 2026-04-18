"""Human-vs-AI game endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tarok.adapters.api.checkpoint_utils import resolve_checkpoint
from tarok.adapters.players.factory import get_player_factory
from tarok.adapters.players.human_player import HumanPlayer
from tarok.adapters.players.neural_player import NeuralPlayer
from tarok.adapters.experience_logger import HumanPlayExperienceLogger
from tarok.adapters.score_breakdown_parser import JsonScoreBreakdownParser
from tarok.adapters.api.ws_observer import WebSocketObserver
from tarok.adapters.api.schemas import NewGameRequest
from tarok.entities import Card, Suit, SuitRank, Contract
from tarok.entities.game_types import suit_card
from tarok.use_cases.game_loop import RustGameLoop as GameLoop
from tarok.use_cases.rust_state import _RUST_U8_TO_PY_CONTRACT

router = APIRouter(tags=["game"])

_active_games: dict[str, dict] = {}


def _card_from_dict(c: dict) -> Card:
    """Reconstruct a Card from a frontend card dict using the Rust index."""
    if c.get("card_type") == "tarok" or c.get("card_type") == 0:
        return Card(int(c["value"]) - 1)
    else:
        suit = Suit(c["suit"])
        rank = SuitRank(int(c["value"]))
        return suit_card(suit, rank)


def _load_opponent(choice: str, index: int):
    """Load a single AI opponent.

    Supports registry players (stockskis_v*, stockskis_m6, nn, human)
    and checkpoint tokens:
      - "PersonaName"             → ../data/checkpoints/PersonaName/_current.pt
      - "PersonaName/_current.pt" → ../data/checkpoints/PersonaName/_current.pt
      - "hall_of_fame/foo.pt"     → ../data/checkpoints/hall_of_fame/foo.pt
    """
    registry = get_player_factory()

    if registry.has(choice):
        return registry.create(choice, name=f"AI-{index + 1}")

    path = resolve_checkpoint(choice)
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


@router.post("/api/game/new")
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


@router.websocket("/ws/game/{game_id}")
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

    cumulative_scores = {i: 0 for i in range(4)}
    caller_counts = {i: 0 for i in range(4)}
    called_counts = {i: 0 for i in range(4)}
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
                score_breakdown_parser=JsonScoreBreakdownParser(),
            )

            observer.set_match_info(
                round_num=round_num + 1,
                total_rounds=num_rounds,
                cumulative_scores=cumulative_scores,
                caller_counts=caller_counts,
                called_counts=called_counts,
                round_history=round_history,
            )

            state, scores = await game_loop.run(dealer=dealer)

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
                round_history[-1]["experience_file"] = None
                round_history[-1]["experience_steps"] = 0

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
