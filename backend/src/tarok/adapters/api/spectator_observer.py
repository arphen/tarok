"""Spectator observer — broadcasts full game state (all hands visible) to connected clients."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from tarok.entities.card import Card
from tarok.entities.game_state import Contract, GameState, Trick
from tarok.entities.scoring import score_game_breakdown


BACKEND_ROOT = Path(__file__).resolve().parents[4]
REPLAYS_DIR = BACKEND_ROOT / "checkpoints" / "replays"


def _ensure_replays_dir() -> None:
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_replay_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return cleaned or f"replay-{int(time.time())}"


def _resolve_replay_path(name: str) -> Path:
    _ensure_replays_dir()
    filename = _sanitize_replay_name(name)
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    path = (REPLAYS_DIR / filename).resolve()
    if not path.is_relative_to(REPLAYS_DIR.resolve()):
        raise ValueError("replay must stay inside checkpoints/replays")
    return path


def list_replays() -> list[dict[str, Any]]:
    _ensure_replays_dir()
    items: list[dict[str, Any]] = []
    for replay_file in sorted(REPLAYS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(replay_file.read_text())
            timeline = payload.get("timeline") or []
            items.append({
                "filename": replay_file.name,
                "created_at": payload.get("created_at", replay_file.stat().st_mtime),
                "source": payload.get("source", "spectate"),
                "label": payload.get("label", replay_file.stem),
                "player_names": payload.get("player_names", []),
                "events": len(timeline),
            })
        except Exception:
            items.append({
                "filename": replay_file.name,
                "created_at": replay_file.stat().st_mtime,
                "source": "unknown",
                "label": replay_file.stem,
                "player_names": [],
                "events": 0,
            })
    return items


def load_replay(name: str) -> dict[str, Any]:
    path = _resolve_replay_path(name)
    return json.loads(path.read_text())


def _card_to_dict(card: Card) -> dict:
    return {
        "card_type": card.card_type.value,
        "value": card.value,
        "suit": card.suit.value if card.suit else None,
        "label": card.label,
        "points": card.points,
    }


def _full_state(state: GameState, player_names: list[str]) -> dict:
    """Return the complete game state with all hands visible."""
    return {
        "phase": state.phase.value,
        "hands": [
            [_card_to_dict(c) for c in state.hands[i]]
            for i in range(4)
        ],
        "hand_sizes": [len(h) for h in state.hands],
        "talon_groups": (
            [[_card_to_dict(c) for c in g] for g in state.talon_revealed]
            if state.talon_revealed
            else None
        ),
        "bids": [
            {"player": b.player, "contract": b.contract.value if b.contract else None}
            for b in state.bids
        ],
        "contract": state.contract.value if state.contract else None,
        "declarer": state.declarer,
        "called_king": _card_to_dict(state.called_king) if state.called_king else None,
        "partner_revealed": state.is_partner_revealed,
        "partner": state.partner if state.is_partner_revealed else None,
        "current_trick": (
            [(p, _card_to_dict(c)) for p, c in state.current_trick.cards]
            if state.current_trick
            else []
        ),
        "tricks_played": state.tricks_played,
        "current_player": state.current_player,
        "scores": state.scores if state.scores else None,
        "player_names": player_names,
        "completed_tricks": [
            {
                "lead_player": t.lead_player,
                "cards": [(p, _card_to_dict(c)) for p, c in t.cards],
                "winner": t.winner(),
            }
            for t in state.tricks
        ],
        "roles": {
            str(k): v.value if hasattr(v, "value") else str(v)
            for k, v in (state.roles or {}).items()
        },
        "announcements": {
            str(k): [a.value if hasattr(a, "value") else str(a) for a in v]
            for k, v in (state.announcements or {}).items()
        },
        "kontra_levels": {
            (k.value if hasattr(k, "value") else str(k)): (v.value if hasattr(v, "value") else str(v))
            for k, v in (state.kontra_levels or {}).items()
        },
        "put_down": [_card_to_dict(c) for c in state.put_down] if state.put_down else [],
    }


class SpectatorObserver:
    """Broadcasts game events to spectator WebSocket clients with a configurable delay."""

    def __init__(
        self,
        websockets: list[WebSocket],
        player_names: list[str],
        delay: float = 1.0,
        replay_name: str | None = None,
        replay_metadata: dict[str, Any] | None = None,
    ):
        self._websockets = websockets
        self._player_names = player_names
        self._delay = delay
        self.next_trick_event = asyncio.Event()
        self._replay_name = replay_name
        self._replay_metadata = replay_metadata or {}
        self._timeline: list[dict[str, Any]] = []

    def _save_replay(self) -> None:
        if not self._replay_name or not self._timeline:
            return
        payload = {
            "filename": _resolve_replay_path(self._replay_name).name,
            "created_at": time.time(),
            "player_names": self._player_names,
            "timeline": self._timeline,
            **self._replay_metadata,
        }
        _resolve_replay_path(self._replay_name).write_text(json.dumps(payload))

    async def _broadcast(self, event: str, data: Any, state: GameState) -> None:
        msg = {
            "event": event,
            "data": data,
            "state": _full_state(state, self._player_names),
        }
        self._timeline.append(msg)
        dead = []
        for ws in self._websockets:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._websockets.remove(ws)
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        else:
            await asyncio.sleep(0)

    async def on_game_start(self, state: GameState) -> None:
        await self._broadcast("game_start", {}, state)

    async def on_deal(self, state: GameState) -> None:
        await self._broadcast("deal", {}, state)

    async def on_bid(self, player: int, bid: Contract | None, state: GameState) -> None:
        await self._broadcast("bid", {
            "player": player,
            "contract": bid.value if bid else None,
        }, state)

    async def on_contract_won(self, player: int, contract: Contract, state: GameState) -> None:
        await self._broadcast("contract_won", {
            "player": player,
            "contract": contract.value,
        }, state)

    async def on_king_called(self, player: int, king: Card, state: GameState) -> None:
        await self._broadcast("king_called", {
            "player": player,
            "king": _card_to_dict(king),
        }, state)

    async def on_talon_revealed(self, groups: list[list[Card]], state: GameState) -> None:
        await self._broadcast("talon_revealed", {
            "groups": [[_card_to_dict(c) for c in g] for g in groups],
        }, state)

    async def on_talon_exchanged(self, state: GameState, picked: list | None = None, discarded: list | None = None) -> None:
        data: dict = {}
        if picked:
            data["picked"] = [_card_to_dict(c) for c in picked]
        if discarded:
            data["discarded"] = [_card_to_dict(c) for c in discarded]
        await self._broadcast("talon_exchanged", data, state)

    async def on_card_played(self, player: int, card: Card, state: GameState) -> None:
        await self._broadcast("card_played", {
            "player": player,
            "card": _card_to_dict(card),
        }, state)

    async def on_rule_verified(self, player: int, rule: str, state: GameState) -> None:
        await self._broadcast("rule_verified", {
            "player": player,
            "rule": rule,
        }, state)

    async def on_trick_won(self, trick: Trick, winner: int, state: GameState) -> None:
        await self._broadcast("trick_won", {
            "winner": winner,
            "cards": [(p, _card_to_dict(c)) for p, c in trick.cards],
        }, state)
        
        if self._delay < 0:
            # Manual mode: Wait for spectator action
            self.next_trick_event.clear()
            await self.next_trick_event.wait()
        elif self._delay > 0:
            await asyncio.sleep(self._delay)
        else:
            await asyncio.sleep(0)
            # Auto mode but extra delay for end of trick
            await asyncio.sleep(self._delay)

    async def on_game_end(self, scores: dict[int, int], state: GameState) -> None:
        breakdown_data = score_game_breakdown(state)
        await self._broadcast("game_end", {
            "scores": {str(k): v for k, v in scores.items()},
            "breakdown": breakdown_data["breakdown"],
            "trick_summary": breakdown_data["trick_summary"],
        }, state)
        self._save_replay()
