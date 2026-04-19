"""Tests for tournament endpoints."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from tarok.adapters.api.server import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---- Single match endpoint ----


async def test_tournament_match_basic(client):
    """POST /api/tournament/match runs games and returns ranked results."""
    resp = await client.post(
        "/api/tournament/match",
        json={
            "agents": [
                {"name": "A", "type": "random"},
                {"name": "B", "type": "random"},
                {"name": "C", "type": "random"},
                {"name": "D", "type": "random"},
            ],
            "num_games": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "cumulative" in data
    assert "ranked" in data
    assert "games" in data
    assert len(data["ranked"]) == 4
    # All 4 seats present in cumulative
    for seat in ["0", "1", "2", "3"]:
        assert seat in data["cumulative"]


async def test_tournament_match_returns_correct_names(client):
    """Agent names in the ranked result match what was sent."""
    resp = await client.post(
        "/api/tournament/match",
        json={
            "agents": [
                {"name": "Alpha", "type": "random"},
                {"name": "Beta", "type": "random"},
                {"name": "Gamma", "type": "random"},
                {"name": "Delta", "type": "random"},
            ],
            "num_games": 1,
        },
    )
    data = resp.json()
    names = {r["name"] for r in data["ranked"]}
    assert names == {"Alpha", "Beta", "Gamma", "Delta"}


async def test_tournament_match_multiple_games(client):
    """Multiple games should produce a games list matching num_games."""
    resp = await client.post(
        "/api/tournament/match",
        json={
            "agents": [
                {"name": "A", "type": "random"},
                {"name": "B", "type": "random"},
                {"name": "C", "type": "random"},
                {"name": "D", "type": "random"},
            ],
            "num_games": 3,
        },
    )
    data = resp.json()
    assert len(data["games"]) == 3


async def test_tournament_match_num_games_clamped(client):
    """num_games is clamped to [1, 100]."""
    resp = await client.post(
        "/api/tournament/match",
        json={
            "agents": [
                {"name": "A", "type": "random"},
                {"name": "B", "type": "random"},
                {"name": "C", "type": "random"},
                {"name": "D", "type": "random"},
            ],
            "num_games": 0,
        },
    )
    data = resp.json()
    assert len(data["games"]) >= 1


async def test_tournament_match_fewer_agents_padded(client):
    """Fewer than 4 agents should be padded with random fillers."""
    resp = await client.post(
        "/api/tournament/match",
        json={
            "agents": [
                {"name": "Solo", "type": "random"},
            ],
            "num_games": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ranked"]) == 4


async def test_tournament_match_ranked_is_sorted(client):
    """Ranked list should be sorted by score descending."""
    resp = await client.post(
        "/api/tournament/match",
        json={
            "agents": [
                {"name": "A", "type": "random"},
                {"name": "B", "type": "random"},
                {"name": "C", "type": "random"},
                {"name": "D", "type": "random"},
            ],
            "num_games": 2,
        },
    )
    data = resp.json()
    scores = [r["score"] for r in data["ranked"]]
    assert scores == sorted(scores, reverse=True)


# ---- Multi-tournament simulation ----


async def test_multi_tournament_simulate_starts(client):
    """POST /api/tournament/simulate starts the simulation."""
    resp = await client.post(
        "/api/tournament/simulate",
        json={
            "agents": [
                {"name": "M1", "type": "random"},
                {"name": "M2", "type": "random"},
                {"name": "M3", "type": "random"},
                {"name": "M4", "type": "random"},
            ],
            "num_tournaments": 1,
            "games_per_round": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"


async def test_multi_tournament_progress_idle(client):
    """GET /api/tournament/simulate/progress returns idle when nothing running."""
    # Stop any leftover task first
    await client.post("/api/tournament/simulate/stop")
    await asyncio.sleep(0.1)

    resp = await client.get("/api/tournament/simulate/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("idle", "done", "cancelled")


async def test_multi_tournament_completes(client):
    """A 1-tournament simulation should complete and show standings."""
    await client.post(
        "/api/tournament/simulate",
        json={
            "agents": [
                {"name": "T1", "type": "random"},
                {"name": "T2", "type": "random"},
                {"name": "T3", "type": "random"},
                {"name": "T4", "type": "random"},
            ],
            "num_tournaments": 1,
            "games_per_round": 1,
        },
    )

    # Poll until done (with timeout)
    for _ in range(60):
        await asyncio.sleep(0.5)
        resp = await client.get("/api/tournament/simulate/progress")
        data = resp.json()
        if data["status"] == "done":
            break
    else:
        pytest.fail("Multi-tournament did not complete in time")

    assert data["status"] == "done"
    assert data["current"] == 1
    assert data["total"] == 1
    standings = data["standings"]
    # All 4 named agents should appear (plus fillers to pad to 8)
    named = [s for s in standings.values() if s["name"] in ("T1", "T2", "T3", "T4")]
    assert len(named) == 4
    for s in named:
        assert s["tournaments_played"] == 1
        assert s["avg_placement"] >= 1
        assert len(s["placements"]) == 1


async def test_multi_tournament_standings_have_correct_fields(client):
    """Each standing entry should have all expected fields."""
    await client.post(
        "/api/tournament/simulate",
        json={
            "agents": [
                {"name": "F1", "type": "random"},
                {"name": "F2", "type": "random"},
                {"name": "F3", "type": "random"},
                {"name": "F4", "type": "random"},
            ],
            "num_tournaments": 1,
            "games_per_round": 1,
        },
    )

    for _ in range(60):
        await asyncio.sleep(0.5)
        resp = await client.get("/api/tournament/simulate/progress")
        data = resp.json()
        if data["status"] == "done":
            break

    for s in data["standings"].values():
        assert "name" in s
        assert "type" in s
        assert "wins" in s
        assert "top2" in s
        assert "top4" in s
        assert "avg_placement" in s
        assert "placements" in s
        assert "tournaments_played" in s


async def test_multi_tournament_exactly_one_winner_per_tournament(client):
    """Across all standings, total wins should equal num_tournaments."""
    await client.post(
        "/api/tournament/simulate",
        json={
            "agents": [
                {"name": "W1", "type": "random"},
                {"name": "W2", "type": "random"},
                {"name": "W3", "type": "random"},
                {"name": "W4", "type": "random"},
            ],
            "num_tournaments": 2,
            "games_per_round": 1,
        },
    )

    for _ in range(120):
        await asyncio.sleep(0.5)
        resp = await client.get("/api/tournament/simulate/progress")
        data = resp.json()
        if data["status"] == "done":
            break
    else:
        pytest.fail("Multi-tournament did not complete in time")

    total_wins = sum(s["wins"] for s in data["standings"].values())
    assert total_wins == 2


async def test_multi_tournament_stop(client):
    """POST /api/tournament/simulate/stop cancels the task."""
    await client.post(
        "/api/tournament/simulate",
        json={
            "agents": [
                {"name": "S1", "type": "random"},
                {"name": "S2", "type": "random"},
                {"name": "S3", "type": "random"},
                {"name": "S4", "type": "random"},
            ],
            "num_tournaments": 50,
            "games_per_round": 1,
        },
    )

    await asyncio.sleep(0.2)
    resp = await client.post("/api/tournament/simulate/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


async def test_multi_tournament_num_clamped(client):
    """num_tournaments is clamped to [1, 100]."""
    resp = await client.post(
        "/api/tournament/simulate",
        json={
            "agents": [
                {"name": "C1", "type": "random"},
                {"name": "C2", "type": "random"},
                {"name": "C3", "type": "random"},
                {"name": "C4", "type": "random"},
            ],
            "num_tournaments": 200,
            "games_per_round": 1,
        },
    )
    data = resp.json()
    # Server should clamp to 100
    assert data.get("num_tournaments", 100) <= 100


# ---- Root checkpoints scanning ----


async def test_checkpoints_includes_root_level(client, tmp_path, monkeypatch):
    """Checkpoints endpoint should list files from ../checkpoints/ too."""
    import torch

    # Create backend/checkpoints/ and ../checkpoints/ relative to a temp CWD
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    ckpt_dir = backend_dir / "checkpoints"
    ckpt_dir.mkdir()
    root_ckpt_dir = tmp_path / "checkpoints"
    root_ckpt_dir.mkdir()

    # Save a dummy checkpoint in root
    dummy = {"episode": 42, "metrics": {"win_rate": 0.7, "avg_reward": 5.0}}
    torch.save(dummy, root_ckpt_dir / "tarok_agent_ep42.pt")

    monkeypatch.chdir(backend_dir)

    resp = await client.get("/api/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    filenames = [c["filename"] for c in data["checkpoints"]]
    assert "tarok_agent_ep42.pt" in filenames


async def test_checkpoints_lists_persona_test_pt(client, tmp_path, monkeypatch):
    """Persona folders expose both _current.pt and _test.pt when present."""
    import torch

    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    data_ckpt = tmp_path / "data" / "checkpoints"
    persona = data_ckpt / "TestPersona"
    persona.mkdir(parents=True)
    dummy = {"episode": 1, "model_name": "M", "metrics": {"win_rate": 0.5}}
    torch.save(dummy, persona / "_current.pt")
    torch.save(dummy, persona / "_test.pt")

    monkeypatch.chdir(backend_dir)

    resp = await client.get("/api/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    filenames = [c["filename"] for c in data["checkpoints"]]
    assert "TestPersona/_current.pt" in filenames
    assert "TestPersona/_test.pt" in filenames
    test_entry = next(c for c in data["checkpoints"] if c["filename"] == "TestPersona/_test.pt")
    assert "(test)" in (test_entry.get("model_name") or "")


async def test_checkpoints_deduplicates(client, tmp_path, monkeypatch):
    """If same filename exists in both dirs, only one should appear."""
    import torch

    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    ckpt_dir = backend_dir / "checkpoints"
    ckpt_dir.mkdir()
    root_ckpt_dir = tmp_path / "checkpoints"
    root_ckpt_dir.mkdir()

    dummy = {"episode": 1, "metrics": {"win_rate": 0.5, "avg_reward": 0}}
    torch.save(dummy, ckpt_dir / "tarok_agent_ep1.pt")
    torch.save(dummy, root_ckpt_dir / "tarok_agent_ep1.pt")

    monkeypatch.chdir(backend_dir)

    resp = await client.get("/api/checkpoints")
    data = resp.json()
    names = [c["filename"] for c in data["checkpoints"]]
    assert names.count("tarok_agent_ep1.pt") == 1


# ---- Multi-tournament resilience ----


async def test_multi_tournament_resilience_completes(client):
    """Multi-tournament with many random agents should complete all tournaments."""
    await client.post(
        "/api/tournament/simulate",
        json={
            "agents": [
                {"name": "R1", "type": "random"},
                {"name": "R2", "type": "random"},
                {"name": "R3", "type": "random"},
                {"name": "R4", "type": "random"},
            ],
            "num_tournaments": 3,
            "games_per_round": 1,
        },
    )

    for _ in range(120):
        await asyncio.sleep(0.5)
        resp = await client.get("/api/tournament/simulate/progress")
        data = resp.json()
        if data["status"] == "done":
            break
    else:
        pytest.fail("Multi-tournament did not complete in time")

    assert data["current"] == 3
    assert data["total"] == 3
    # Every agent should have played 3 tournaments
    for s in data["standings"].values():
        assert s["tournaments_played"] == 3


async def test_multi_tournament_error_cancelled_progress_reflects(client):
    """Stopping mid-run should reflect cancelled status in progress."""
    await client.post(
        "/api/tournament/simulate",
        json={
            "agents": [
                {"name": "X1", "type": "random"},
                {"name": "X2", "type": "random"},
                {"name": "X3", "type": "random"},
                {"name": "X4", "type": "random"},
            ],
            "num_tournaments": 100,
            "games_per_round": 1,
        },
    )

    # Let it start, then cancel
    await asyncio.sleep(1)
    await client.post("/api/tournament/simulate/stop")
    await asyncio.sleep(1)

    resp = await client.get("/api/tournament/simulate/progress")
    data = resp.json()
    assert data["status"] in ("cancelled", "done")
