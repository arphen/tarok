"""Tests for per-opponent model selection in game creation."""

import pytest
from httpx import ASGITransport, AsyncClient

from tarok.adapters.api.server import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_new_game_default_opponents(client):
    """POST /api/game/new with no body still works (all latest)."""
    resp = await client.post("/api/game/new")
    assert resp.status_code == 200
    assert "game_id" in resp.json()


async def test_new_game_with_random_opponents(client):
    """Select 'random' (untrained) models for all opponents."""
    resp = await client.post(
        "/api/game/new",
        json={
            "opponents": ["random", "random", "random"],
        },
    )
    assert resp.status_code == 200
    assert "game_id" in resp.json()


async def test_new_game_with_latest_opponents(client):
    resp = await client.post(
        "/api/game/new",
        json={
            "opponents": ["latest", "latest", "latest"],
        },
    )
    assert resp.status_code == 200
    assert "game_id" in resp.json()


async def test_new_game_mixed_opponents(client):
    """Mix of random and latest models."""
    resp = await client.post(
        "/api/game/new",
        json={
            "opponents": ["latest", "random", "latest"],
        },
    )
    assert resp.status_code == 200
    assert "game_id" in resp.json()


async def test_new_game_nonexistent_checkpoint(client):
    """A checkpoint filename that doesn't exist falls back to untrained."""
    resp = await client.post(
        "/api/game/new",
        json={
            "opponents": ["nonexistent.pt", "random", "latest"],
        },
    )
    assert resp.status_code == 200
    assert "game_id" in resp.json()


async def test_new_game_fewer_than_three_opponents(client):
    """Fewer than 3 entries should be padded with 'latest'."""
    resp = await client.post(
        "/api/game/new",
        json={
            "opponents": ["random"],
        },
    )
    assert resp.status_code == 200
    assert "game_id" in resp.json()


async def test_checkpoints_endpoint(client):
    resp = await client.get("/api/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert "checkpoints" in data
    assert isinstance(data["checkpoints"], list)
