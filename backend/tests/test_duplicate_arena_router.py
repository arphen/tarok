"""Tests for duplicate-arena HTTP endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from tarok.adapters import duplicate_arena_history
from tarok.adapters.api.server import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    """Redirect persistence to a tmp file."""
    path = tmp_path / "duplicate_arena_results.json"
    monkeypatch.setattr(duplicate_arena_history, "_HISTORY_PATH", path)
    return path


@pytest.mark.asyncio
async def test_duplicate_history_empty_returns_empty_list(client, isolated_history):
    async with client as c:
        resp = await c.get("/api/arena/duplicate/history")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


@pytest.mark.asyncio
async def test_duplicate_progress_reports_last_run_when_idle(client, isolated_history):
    duplicate_arena_history.persist_run(
        challenger="A.pt",
        defender="B.pt",
        boards=10,
        seed=42,
        pairing="rotation_8game",
        status="done",
        result={
            "boards_played": 8,
            "challenger_mean_score": 1.5,
            "defender_mean_score": -0.5,
            "mean_duplicate_advantage": 2.0,
            "duplicate_advantage_std": 0.1,
            "ci_low_95": 1.9,
            "ci_high_95": 2.1,
            "imps_per_board": 0.2,
        },
    )
    async with client as c:
        resp = await c.get("/api/arena/duplicate/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["challenger"] == "A.pt"
    assert data["result"]["boards_played"] == 8


@pytest.mark.asyncio
async def test_duplicate_start_rejects_unknown_checkpoints(client, isolated_history):
    async with client as c:
        resp = await c.post(
            "/api/arena/duplicate/start",
            json={
                "challenger": "this_ckpt_does_not_exist_xyz.pt",
                "defender": "another_missing_xyz.pt",
                "boards": 10,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "not found" in data["message"]


@pytest.mark.asyncio
async def test_duplicate_start_rejects_invalid_pairing(client, isolated_history):
    async with client as c:
        resp = await c.post(
            "/api/arena/duplicate/start",
            json={
                "challenger": "A.pt",
                "defender": "B.pt",
                "boards": 10,
                "pairing": "not_a_real_pairing",
            },
        )
    data = resp.json()
    assert data["status"] == "error"
    assert "pairing" in data["message"]


@pytest.mark.asyncio
async def test_duplicate_start_rejects_zero_boards(client, isolated_history):
    async with client as c:
        resp = await c.post(
            "/api/arena/duplicate/start",
            json={
                "challenger": "A.pt",
                "defender": "B.pt",
                "boards": 0,
            },
        )
    data = resp.json()
    assert data["status"] == "error"
    assert "boards" in data["message"]


@pytest.mark.asyncio
async def test_duplicate_start_accepts_rotation_6game_pairing(client, isolated_history):
    """`rotation_6game` is a valid pairing token (3-player); router should
    surface a different validation error (missing checkpoint) rather than
    'unsupported pairing'."""
    async with client as c:
        resp = await c.post(
            "/api/arena/duplicate/start",
            json={
                "challenger": "this_ckpt_does_not_exist_xyz.pt",
                "defender": "another_missing_xyz.pt",
                "boards": 10,
                "pairing": "rotation_6game",
            },
        )
    data = resp.json()
    assert data["status"] == "error"
    # Must not be the "unsupported pairing" error.
    assert "pairing" not in data["message"].lower() or "not found" in data["message"].lower()
