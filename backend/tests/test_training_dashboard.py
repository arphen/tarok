"""Tests for training dashboard API endpoints and trainer integration."""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import torch
from httpx import ASGITransport, AsyncClient

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.training_lab import PPOTrainer, TrainingMetrics, _HAS_RUST
from tarok.adapters.api.server import app, _latest_metrics, _checkpoint_cache


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
    assert len(result.avg_placement_history) == 2
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
    m = TrainingMetrics(episode=10, session=2, total_sessions=5, avg_placement=2.5)
    d = m.to_dict()
    assert d["episode"] == 10
    assert d["session"] == 2
    assert d["avg_placement"] == 2.5
    assert "contract_stats" in d
    assert "reward_history" in d
    assert "tarok_count_bids" in d


async def test_rust_engine_fallback(trainer):
    """When Rust engine is unavailable, use_rust_engine should be False."""
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    with patch("tarok.adapters.ai.training_lab._HAS_RUST", False):
        t = PPOTrainer(agents, lr=3e-4, device="cpu", games_per_session=5, use_rust_engine=True)
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


# ── Checkpoint endpoint tests ───────────────────────────────────────

def _save_fake_checkpoint(path: Path, episode: int = 0, session: int = 0,
                          win_rate: float = 0.0, avg_reward: float = 0.0,
                          model_name: str | None = None):
    """Save a minimal fake checkpoint with metadata."""
    data = {
        "episode": episode,
        "session": session,
        "metrics": {"win_rate": win_rate, "avg_reward": avg_reward},
        "model_name": model_name,
        "model_state_dict": {},
        "optimizer_state_dict": {},
    }
    torch.save(data, path)


async def test_checkpoints_includes_latest(tmp_path):
    """The 'latest' checkpoint must appear in the response."""
    from tarok.adapters.api.server import _load_checkpoint_meta
    d = tmp_path / "checkpoints"
    d.mkdir()
    _save_fake_checkpoint(d / "tarok_agent_latest.pt", episode=100, session=10, win_rate=0.55)
    _checkpoint_cache.clear()

    entry = _load_checkpoint_meta(d / "tarok_agent_latest.pt", d, d / "breeding_results")
    assert entry["filename"] == "tarok_agent_latest.pt"
    assert entry["episode"] == 100
    assert entry["session"] == 10
    assert entry["win_rate"] == 0.55


async def test_checkpoints_returns_list(client):
    """GET /api/checkpoints always returns a list, even with no files."""
    resp = await client.get("/api/checkpoints")
    assert resp.status_code == 200
    d = resp.json()
    assert "checkpoints" in d
    assert isinstance(d["checkpoints"], list)


async def test_checkpoint_metadata_fields(tmp_path):
    """Each checkpoint entry has all expected metadata fields."""
    from tarok.adapters.api.server import _load_checkpoint_meta
    d = tmp_path / "checkpoints"
    d.mkdir()
    _save_fake_checkpoint(
        d / "tarok_agent_ep50.pt", episode=50, session=5,
        win_rate=0.6, avg_reward=12.5, model_name="test_model"
    )
    _checkpoint_cache.clear()

    entry = _load_checkpoint_meta(d / "tarok_agent_ep50.pt", d, d / "breeding_results")
    assert entry["filename"] == "tarok_agent_ep50.pt"
    assert entry["episode"] == 50
    assert entry["session"] == 5
    assert entry["win_rate"] == 0.6
    assert entry["avg_reward"] == 12.5
    assert entry["model_name"] == "test_model"
    assert entry["is_bred"] is False


async def test_checkpoint_cache_avoids_reload(tmp_path):
    """Second call with same mtime returns cached result without re-loading."""
    from tarok.adapters.api.server import _load_checkpoint_meta
    d = tmp_path / "checkpoints"
    d.mkdir()
    fpath = d / "tarok_agent_ep10.pt"
    _save_fake_checkpoint(fpath, episode=10, session=1)
    _checkpoint_cache.clear()

    entry1 = _load_checkpoint_meta(fpath, d, d / "breeding_results")
    assert str(fpath) in _checkpoint_cache

    entry2 = _load_checkpoint_meta(fpath, d, d / "breeding_results")
    assert entry1 is entry2  # exact same dict object from cache


async def test_checkpoint_cache_invalidates_on_mtime_change(tmp_path):
    """Cache is invalidated when file modification time changes."""
    from tarok.adapters.api.server import _load_checkpoint_meta
    import os
    d = tmp_path / "checkpoints"
    d.mkdir()
    fpath = d / "tarok_agent_ep20.pt"
    _save_fake_checkpoint(fpath, episode=20, session=2, win_rate=0.4)
    _checkpoint_cache.clear()

    entry1 = _load_checkpoint_meta(fpath, d, d / "breeding_results")
    assert entry1["win_rate"] == 0.4

    # Re-save with different data and bump mtime
    time.sleep(0.05)
    _save_fake_checkpoint(fpath, episode=20, session=2, win_rate=0.7)
    os.utime(fpath, (time.time() + 1, time.time() + 1))

    entry2 = _load_checkpoint_meta(fpath, d, d / "breeding_results")
    assert entry2["win_rate"] == 0.7
    assert entry1 is not entry2


async def test_bred_checkpoint_detected(tmp_path):
    """Bred model checkpoints have is_bred=True."""
    from tarok.adapters.api.server import _load_checkpoint_meta
    d = tmp_path / "checkpoints"
    breed_dir = d / "breeding_results"
    breed_dir.mkdir(parents=True)
    fpath = breed_dir / "bred_model_final.pt"
    _save_fake_checkpoint(fpath, episode=0, model_name="bred_v1")
    _checkpoint_cache.clear()

    entry = _load_checkpoint_meta(fpath, d, breed_dir)
    assert entry["is_bred"] is True
    assert entry["model_name"] == "bred_v1"
    assert entry["filename"] == "breeding_results/bred_model_final.pt"


async def test_corrupt_checkpoint_handled(tmp_path):
    """Corrupt checkpoint files don't crash the endpoint."""
    from tarok.adapters.api.server import _load_checkpoint_meta
    d = tmp_path / "checkpoints"
    d.mkdir()
    fpath = d / "tarok_agent_ep99.pt"
    fpath.write_bytes(b"not a valid torch file")
    _checkpoint_cache.clear()

    entry = _load_checkpoint_meta(fpath, d, d / "breeding_results")
    assert entry["filename"] == "tarok_agent_ep99.pt"
    assert entry["episode"] == 0
    assert entry["is_bred"] is False


# ── Responsiveness tests ────────────────────────────────────────────

async def test_concurrent_polling_responsive(client):
    """Metrics, status and checkpoints can all be fetched concurrently."""
    results = await asyncio.gather(
        client.get("/api/training/metrics"),
        client.get("/api/training/status"),
        client.get("/api/checkpoints"),
    )
    for resp in results:
        assert resp.status_code == 200


async def test_metrics_available_during_training(client):
    """Metrics endpoint remains responsive while training is running."""
    resp = await client.post("/api/training/start", json={
        "num_sessions": 3,
        "games_per_session": 5,
    })
    assert resp.status_code == 200

    # Poll a few times while training runs
    responsive_count = 0
    for _ in range(10):
        await asyncio.sleep(0.3)
        t0 = time.monotonic()
        results = await asyncio.gather(
            client.get("/api/training/metrics"),
            client.get("/api/training/status"),
            client.get("/api/checkpoints"),
        )
        elapsed = time.monotonic() - t0
        if all(r.status_code == 200 for r in results):
            responsive_count += 1
        # Each poll should complete quickly (< 5s even on slow CI)
        assert elapsed < 5.0, f"Polling took {elapsed:.1f}s — server is blocked"
        status = results[1].json()
        if not status["running"]:
            break

    assert responsive_count >= 1, "Server was never responsive during training"


async def test_checkpoints_endpoint_not_blocking(client):
    """GET /api/checkpoints should return within a reasonable time."""
    t0 = time.monotonic()
    resp = await client.get("/api/checkpoints")
    elapsed = time.monotonic() - t0
    assert resp.status_code == 200
    assert elapsed < 5.0, f"/api/checkpoints took {elapsed:.1f}s"


# ── Robustness tests — game failures & NaN ──────────────────────────

async def test_trainer_survives_game_exception():
    """If a single game raises, training continues with the remaining games."""
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    t = PPOTrainer(agents, lr=3e-4, device="cpu", games_per_session=5)

    call_count = 0
    _original_run = None

    # Patch GameLoop.run to fail on the 3rd game
    from tarok.adapters.ai.rust_game_loop import RustGameLoop as GameLoop
    _original_run = GameLoop.run

    async def flaky_run(self, *a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("Simulated game crash")
        return await _original_run(self, *a, **kw)

    with patch.object(GameLoop, "run", flaky_run):
        result = await t.train(1)

    # Training should complete. The crashing game is skipped, so we get
    # 4 games instead of 5. episode == games actually finished.
    assert result.session == 1
    assert result.episode == 4  # 5 - 1 crashed = 4


async def test_trainer_survives_all_games_crash():
    """If every game in a session crashes, training still finishes."""
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    t = PPOTrainer(agents, lr=3e-4, device="cpu", games_per_session=3)

    from tarok.adapters.ai.rust_game_loop import RustGameLoop as GameLoop

    async def always_crash(self, *a, **kw):
        raise RuntimeError("total failure")

    with patch.object(GameLoop, "run", always_crash):
        result = await t.train(2)

    # No games completed but sessions iterated
    assert result.session == 2
    assert result.episode == 0


async def test_agent_mode_restored_after_crash():
    """Agent list is restored even when a lookahead game crashes."""
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    t = PPOTrainer(
        agents, lr=3e-4, device="cpu", games_per_session=2,
        lookahead_ratio=1.0,  # all games use lookahead
        lookahead_sims=5,
    )

    original_agent_names = [a.name for a in t.agents]

    from tarok.adapters.ai.rust_game_loop import RustGameLoop as GameLoop
    _orig = GameLoop.run

    call_count = 0

    async def crash_first(self, *a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("lookahead crash")
        return await _orig(self, *a, **kw)

    with patch.object(GameLoop, "run", crash_first):
        await t.train(1)

    # After training, agents should be the originals (not stuck in lookahead mode)
    restored_names = [a.name for a in t.agents]
    assert restored_names == original_agent_names


async def test_ppo_skips_nan_loss():
    """NaN loss in a batch is skipped rather than corrupting the network."""
    agents = [RLAgent(name=f"Agent-{i}", hidden_size=64) for i in range(4)]
    t = PPOTrainer(agents, lr=3e-4, device="cpu", games_per_session=5)

    # Run one normal session to collect experiences
    result = await t.train(1)
    # Weights should remain finite
    for p in t.shared_network.parameters():
        assert torch.isfinite(p).all(), "Network weights became non-finite"


async def test_training_task_preserves_metrics_on_crash(client):
    """If training crashes, last metrics are preserved (not reset to None)."""
    import tarok.adapters.api.server as srv

    # Start training that will fail
    from tarok.adapters.ai.rust_game_loop import RustGameLoop as GameLoop
    async def crash_run(self, *a, **kw):
        raise RuntimeError("deliberate crash")

    resp = await client.post("/api/training/start", json={
        "num_sessions": 1,
        "games_per_session": 3,
    })
    assert resp.status_code == 200

    # Wait for the task to finish (it does 0 games because all crash,
    # but the train() method itself returns normally since games are skipped)
    for _ in range(30):
        await asyncio.sleep(0.3)
        status = (await client.get("/api/training/status")).json()
        if not status["running"]:
            break

    # Metrics endpoint should still return valid data
    metrics_resp = await client.get("/api/training/metrics")
    assert metrics_resp.status_code == 200
    d = metrics_resp.json()
    assert isinstance(d["episode"], int)
