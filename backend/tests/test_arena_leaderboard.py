"""Tests for persistent arena history and checkpoint leaderboard."""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from tarok.adapters.api.routers import arena_router
from tarok.adapters.api.server import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def _player(
    name: str,
    games_played: int,
    avg_placement: float,
    bid_won_count: int,
    avg_taroks_in_hand: float,
    declared_count: int,
    declared_won: int,
    avg_declared_win_score: float,
    avg_declared_loss_score: float,
    times_called: int,
) -> dict:
    return {
        "name": name,
        "games_played": games_played,
        "avg_placement": avg_placement,
        "bid_won_count": bid_won_count,
        "avg_taroks_in_hand": avg_taroks_in_hand,
        "declared_count": declared_count,
        "declared_won": declared_won,
        "avg_declared_win_score": avg_declared_win_score,
        "avg_declared_loss_score": avg_declared_loss_score,
        "times_called": times_called,
    }


async def test_arena_history_persists_and_filters_by_checkpoint(tmp_path: Path, monkeypatch, client):
    arena_results = tmp_path / "arena_results.json"
    monkeypatch.setattr(arena_router, "_arena_history_path", arena_results)

    req_agents = [
        {"name": "RL-A", "type": "rl", "checkpoint": "cp_a.pt"},
        {"name": "Bot-B", "type": "stockskis", "checkpoint": ""},
        {"name": "Bot-C", "type": "stockskis", "checkpoint": ""},
        {"name": "Bot-D", "type": "stockskis", "checkpoint": ""},
    ]

    payload = {
        "status": "done",
        "games_done": 100,
        "analytics": {
            "players": [
                _player("RL-A", 100, 1.8, 22, 5.2, 18, 11, 42.0, -19.0, 6),
                _player("Bot-B", 100, 2.3, 0, 4.6, 0, 0, 0.0, 0.0, 0),
                _player("Bot-C", 100, 2.9, 0, 4.7, 0, 0, 0.0, 0.0, 0),
                _player("Bot-D", 100, 3.0, 0, 4.4, 0, 0, 0.0, 0.0, 0),
            ]
        },
    }

    arena_router._persist_arena_run(req_agents, total_games=100, session_size=20, payload=payload)

    assert arena_results.exists()

    all_runs = await client.get("/api/arena/history")
    assert all_runs.status_code == 200
    data = all_runs.json()
    assert len(data["runs"]) == 1
    assert data["runs"][0]["checkpoints"] == ["cp_a.pt"]

    filtered = await client.get("/api/arena/history", params={"checkpoint": "cp_a.pt"})
    assert filtered.status_code == 200
    fdata = filtered.json()
    assert len(fdata["runs"]) == 1

    empty = await client.get("/api/arena/history", params={"checkpoint": "missing.pt"})
    assert empty.status_code == 200
    assert empty.json()["runs"] == []


async def test_arena_checkpoint_leaderboard_aggregates_across_runs(tmp_path: Path, monkeypatch, client):
    arena_results = tmp_path / "arena_results.json"
    monkeypatch.setattr(arena_router, "_arena_history_path", arena_results)

    run1_agents = [
        {"name": "RL-A", "type": "rl", "checkpoint": "cp_a.pt"},
        {"name": "RL-B", "type": "rl", "checkpoint": "cp_b.pt"},
        {"name": "Bot-C", "type": "stockskis", "checkpoint": ""},
        {"name": "Bot-D", "type": "stockskis", "checkpoint": ""},
    ]
    run1_payload = {
        "status": "done",
        "games_done": 200,
        "analytics": {
            "players": [
                _player("RL-A", 200, 1.9, 40, 5.1, 30, 18, 45.0, -18.0, 10),
                _player("RL-B", 200, 2.5, 25, 4.8, 24, 12, 36.0, -22.0, 8),
                _player("Bot-C", 200, 2.7, 0, 4.7, 0, 0, 0.0, 0.0, 0),
                _player("Bot-D", 200, 2.9, 0, 4.5, 0, 0, 0.0, 0.0, 0),
            ]
        },
    }

    run2_agents = [
        {"name": "RL-A", "type": "rl", "checkpoint": "cp_a.pt"},
        {"name": "Bot-B", "type": "stockskis", "checkpoint": ""},
        {"name": "Bot-C", "type": "stockskis", "checkpoint": ""},
        {"name": "Bot-D", "type": "stockskis", "checkpoint": ""},
    ]
    run2_payload = {
        "status": "done",
        "games_done": 100,
        "analytics": {
            "players": [
                _player("RL-A", 100, 1.7, 24, 5.3, 16, 11, 41.0, -17.0, 5),
                _player("Bot-B", 100, 2.4, 0, 4.6, 0, 0, 0.0, 0.0, 0),
                _player("Bot-C", 100, 2.9, 0, 4.8, 0, 0, 0.0, 0.0, 0),
                _player("Bot-D", 100, 3.0, 0, 4.4, 0, 0, 0.0, 0.0, 0),
            ]
        },
    }

    arena_router._persist_arena_run(run1_agents, total_games=200, session_size=20, payload=run1_payload)
    arena_router._persist_arena_run(run2_agents, total_games=100, session_size=20, payload=run2_payload)

    resp = await client.get("/api/arena/leaderboard/checkpoints")
    assert resp.status_code == 200

    leaderboard = resp.json()["leaderboard"]
    checkpoints = [row["checkpoint"] for row in leaderboard]
    assert "cp_a.pt" in checkpoints
    assert "cp_b.pt" in checkpoints

    cp_a = next(row for row in leaderboard if row["checkpoint"] == "cp_a.pt")
    assert cp_a["runs"] >= 2
    assert cp_a["games"] == 300
    assert cp_a["bid_wins"] == 64
    assert cp_a["avg_placement"] < 2.0

    cp_b = next(row for row in leaderboard if row["checkpoint"] == "cp_b.pt")
    assert cp_b["runs"] >= 1
    assert cp_b["games"] == 200

    # cp_a should rank above cp_b because it has better (lower) avg placement
    assert leaderboard[0]["checkpoint"] == "cp_a.pt"
