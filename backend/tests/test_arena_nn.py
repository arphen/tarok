"""Test that the arena can run with NN agents and produces analytics."""
import pytest
import numpy as np
from httpx import ASGITransport, AsyncClient
from tarok.adapters.api.server import app
import tempfile
import os

@pytest.mark.asyncio
async def test_arena_runs_with_nn_agent(tmp_path, monkeypatch):
    # Patch arena_results path to temp
    from tarok.adapters.api.routers import arena_router
    arena_results = tmp_path / "arena_results.json"
    monkeypatch.setattr(arena_router, "_arena_history_path", arena_results)

    # Create a real checkpoint file (TorchScript export needs valid weights)
    import torch
    from tarok.core.network import TarokNet
    ckpt_path = tmp_path / "dummy.pt"
    torch.save(TarokNet().state_dict(), str(ckpt_path))

    # Compose request: 1 NN agent, 3 bots
    req_agents = [
        {"name": "NN-1", "type": "rl", "checkpoint": str(ckpt_path.name)},
        {"name": "Bot-2", "type": "stockskis", "checkpoint": ""},
        {"name": "Bot-3", "type": "stockskis", "checkpoint": ""},
        {"name": "Bot-4", "type": "stockskis", "checkpoint": ""},
    ]

    # Patch checkpoint search to look in tmp_path
    monkeypatch.setattr(arena_router, "_ARENA_CHECKPOINT_DIRS", [tmp_path])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/arena/start", json={
            "agents": req_agents,
            "total_games": 20,
            "session_size": 5,
        })
        data = resp.json()
        assert data["status"] == "started"

        # Poll progress until done
        for _ in range(30):
            resp2 = await client.get("/api/arena/progress")
            prog = resp2.json()
            if prog["status"] == "done":
                break
            import asyncio; await asyncio.sleep(0.2)
        else:
            raise AssertionError("Arena did not finish in time")

        # Check analytics
        analytics = prog["analytics"]
        assert analytics is not None
        assert "players" in analytics
        assert any(p["type"] == "rl" for p in analytics["players"])
        assert analytics["games_done"] == 20
        # Should have taroks_per_contract and contracts
        assert "taroks_per_contract" in analytics
        assert "contracts" in analytics
        # Per-player taroks_per_contract
        for p in analytics["players"]:
            assert "taroks_per_contract" in p
            assert isinstance(p["taroks_per_contract"], dict)
