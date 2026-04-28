"""Tests that the bot-arena HTTP router correctly routes 3-player variant
requests through to ``run_self_play``.

These tests do **not** invoke the Rust engine; they patch
``tarok_engine.run_self_play`` to a stub and inspect the resolved variant,
seat count, and engine kwargs.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from tarok.adapters import arena_history
from tarok.adapters.api.routers import arena_router as ar
from tarok.adapters.api.server import app


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setattr(arena_history, "_HISTORY_PATH", tmp_path / "arena_results.json")
    return tmp_path


@pytest.fixture
def stub_engine(monkeypatch):
    """Patch tarok_engine.run_self_play with a stub that records its kwargs."""

    calls: list[dict] = []

    def _fake_run_self_play(**kwargs):
        calls.append(kwargs)
        n = int(kwargs["n_games"])
        # Minimum return shape: scores [N,4] (rust always returns 4 cols),
        # contracts/declarers/partners [N], bid_contracts/taroks_in_hand [N,4].
        return {
            "scores": np.zeros((n, 4), dtype=np.int32),
            "contracts": np.zeros(n, dtype=np.uint8),
            "declarers": np.full(n, -1, dtype=np.int8),
            "partners": np.full(n, -1, dtype=np.int8),
            "bid_contracts": np.full((n, 4), -1, dtype=np.int8),
            "taroks_in_hand": np.zeros((n, 4), dtype=np.uint8),
        }

    fake_te = SimpleNamespace(run_self_play=_fake_run_self_play)
    monkeypatch.setitem(__import__("sys").modules, "tarok_engine", fake_te)
    return calls


@pytest.fixture(autouse=True)
def reset_arena_state():
    # Make sure no prior task is hanging around.
    ar._arena_task = None
    ar._arena_progress = None
    yield
    if ar._arena_task and not ar._arena_task.done():
        ar._arena_task.cancel()
    ar._arena_task = None
    ar._arena_progress = None


@pytest.mark.asyncio
async def test_start_arena_three_player_pads_to_three_seats(client, stub_engine):
    """Variant=three_player should populate exactly 3 seats with
    bot_v3_3p, and forward variant=1 to the engine."""
    payload = {
        "variant": "three_player",
        "agents": [],
        "total_games": 4,
        "session_size": 4,
    }
    async with client as c:
        resp = await c.post("/api/arena/start", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "started"
        assert body["variant"] == "three_player"

        # Wait for the background task to finish.
        for _ in range(50):
            if ar._arena_task is None or ar._arena_task.done():
                break
            await asyncio.sleep(0.05)

    assert stub_engine, "engine was never called"
    last = stub_engine[-1]
    assert last["variant"] == 1
    assert last["seat_config"] == "bot_v3_3p,bot_v3_3p,bot_v3_3p"


@pytest.mark.asyncio
async def test_start_arena_four_player_default_pads_to_four(client, stub_engine):
    payload = {
        "agents": [],
        "total_games": 4,
        "session_size": 4,
    }
    async with client as c:
        resp = await c.post("/api/arena/start", json=payload)
        assert resp.json()["status"] == "started"
        for _ in range(50):
            if ar._arena_task is None or ar._arena_task.done():
                break
            await asyncio.sleep(0.05)

    last = stub_engine[-1]
    assert last["variant"] == 0
    seats = last["seat_config"].split(",")
    assert len(seats) == 4
    assert all(s == "bot_v5" for s in seats)


@pytest.mark.asyncio
async def test_start_arena_rejects_invalid_variant_override_without_checkpoints(
    client, stub_engine
):
    """When request-level variant is provided, n_seats follows it even
    without checkpoints. Sending 5 agents under variant=three_player should
    truncate to 3 and still succeed."""
    payload = {
        "variant": "three_player",
        "agents": [{"name": f"Bot-{i}", "type": "stockskis_v3_3p"} for i in range(5)],
        "total_games": 4,
        "session_size": 4,
    }
    async with client as c:
        resp = await c.post("/api/arena/start", json=payload)
        assert resp.json()["status"] == "started"
        for _ in range(50):
            if ar._arena_task is None or ar._arena_task.done():
                break
            await asyncio.sleep(0.05)

    last = stub_engine[-1]
    assert last["seat_config"].count(",") == 2  # exactly 3 seats
