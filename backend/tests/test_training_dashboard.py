"""Tests for training dashboard API endpoints and trainer integration."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.trainer import PPOTrainer, TrainingMetrics, _HAS_RUST
from tarok.adapters.api.server import app, _latest_metrics


# ── Trainer unit tests ──────────────────────────────────────────────

@pytest.fixture
def trainer():
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    return PPOTrainer(
        agents, lr=3e-4, device="cpu",
        games_per_session=5,
    )


async def test_trainer_completes_sessions(trainer):
    result = await trainer.train(2)
    assert result.episode == 10  # 2 sessions * 5 games
    assert result.session == 2
    assert result.total_sessions == 2


async def test_trainer_metrics_populated(trainer):
    result = await trainer.train(2)
    assert len(result.reward_history) == 2
    assert len(result.win_rate_history) == 2
    assert len(result.loss_history) == 2
    assert len(result.session_avg_score_history) == 2
    assert result.games_per_second > 0


async def test_trainer_contract_stats(trainer):
    result = await trainer.train(3)
    total = sum(cs.played for cs in result.contract_stats.values())
    assert total == 15  # 3 sessions * 5 games


async def test_trainer_stop():
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    t = PPOTrainer(agents, lr=3e-4, device="cpu", games_per_session=5)

    async def stop_after_delay():
        await asyncio.sleep(0.1)
        t.stop()

    asyncio.create_task(stop_after_delay())
    result = await t.train(1000)  # Would take forever without stop
    assert result.session < 1000


async def test_trainer_metrics_callback():
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    t = PPOTrainer(agents, lr=3e-4, device="cpu", games_per_session=5)
    received = []

    async def on_metrics(m):
        received.append(m.episode)

    t.add_metrics_callback(on_metrics)
    await t.train(1)
    assert len(received) > 0
    assert received[-1] == 5


async def test_trainer_to_dict():
    m = TrainingMetrics(episode=10, session=2, total_sessions=5, win_rate=0.5)
    d = m.to_dict()
    assert d["episode"] == 10
    assert d["session"] == 2
    assert d["win_rate"] == 0.5
    assert "contract_stats" in d
    assert "reward_history" in d
    assert "tarok_count_bids" in d


async def test_rust_engine_fallback(trainer):
    """When Rust engine is unavailable, use_rust_engine should be False."""
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    t = PPOTrainer(agents, lr=3e-4, device="cpu", games_per_session=5, use_rust_engine=True)
    # _HAS_RUST is False because tarok_engine isn't installed
    assert t.use_rust_engine is False
    result = await t.train(1)
    assert result.episode == 5


async def test_trainer_tarok_count_bids(trainer):
    result = await trainer.train(2)
    # After 10 games, at least some tarok counts should have data
    total_bids = sum(
        sum(bids.values())
        for bids in result.tarok_count_bids.values()
    )
    assert total_bids == 10  # one entry per game for player 0


# ── API endpoint tests ──────────────────────────────────────────────

@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_api_metrics_default(client):
    resp = await client.get("/api/training/metrics")
    assert resp.status_code == 200
    d = resp.json()
    assert d["episode"] == 0 or isinstance(d["episode"], int)
    assert "contract_stats" in d
    assert "reward_history" in d


async def test_api_status_not_running(client):
    resp = await client.get("/api/training/status")
    assert resp.status_code == 200
    d = resp.json()
    assert "running" in d


async def test_api_checkpoints(client):
    resp = await client.get("/api/checkpoints")
    assert resp.status_code == 200
    d = resp.json()
    assert "checkpoints" in d
    assert isinstance(d["checkpoints"], list)


async def test_api_start_and_stop(client):
    resp = await client.post("/api/training/start", json={
        "num_sessions": 1,
        "games_per_session": 5,
    })
    assert resp.status_code == 200
    d = resp.json()
    assert d["status"] == "started"

    # Wait for training to complete
    for _ in range(30):
        await asyncio.sleep(0.5)
        status = (await client.get("/api/training/status")).json()
        if not status["running"]:
            break

    metrics = (await client.get("/api/training/metrics")).json()
    assert metrics["episode"] >= 5
    assert metrics["session"] >= 1


async def test_api_stop_training(client):
    resp = await client.post("/api/training/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


async def test_api_start_uses_default_schema(client):
    """use_rust_engine should default to False so training doesn't crash."""
    resp = await client.post("/api/training/start", json={
        "num_sessions": 1,
        "games_per_session": 5,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] in ("started", "already_running")

    # Wait for it to finish
    for _ in range(30):
        await asyncio.sleep(0.5)
        status = (await client.get("/api/training/status")).json()
        if not status["running"]:
            break

    metrics = (await client.get("/api/training/metrics")).json()
    assert metrics["episode"] >= 5
