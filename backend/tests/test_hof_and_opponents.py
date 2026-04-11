"""Tests for HoFManager and OpponentPool."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from tarok.adapters.ai.hof_manager import HoFManager
from tarok.adapters.ai.network import TarokNet
from tarok.adapters.ai.opponent_pool import (
    FSPOpponent,
    HoFOpponent,
    OpponentGameResult,
    OpponentPool,
    PureSelfPlayOpponent,
    StockSkisOpponent,
)


def _make_network(hidden_size: int = 256) -> TarokNet:
    net = TarokNet(hidden_size=hidden_size)
    return net


def _dummy_persona(name: str = "Test", age: int = 1) -> dict:
    return {"first_name": name, "last_name": "Model", "age": age}


def _dummy_eval(vs_v1: float = 0.5, vs_v5: float = 0.4) -> list[dict]:
    return [{"vs_v1": vs_v1, "vs_v5": vs_v5, "vs_v3": 0.45}]


# ── HoFManager ──────────────────────────────────────────────────────────


class TestHoFManager:
    def test_save_and_list(self, tmp_path: Path):
        mgr = HoFManager(tmp_path / "hof", max_auto=10)
        net = _make_network()
        info = mgr.save(net, _dummy_persona("Ana"), _dummy_eval())

        assert "filename" in info
        assert info["pinned"] is False

        entries = mgr.list()
        assert len(entries) == 1
        assert entries[0]["persona"]["first_name"] == "Ana"
        assert entries[0].get("pinned") is False

    def test_auto_eviction(self, tmp_path: Path):
        mgr = HoFManager(tmp_path / "hof", max_auto=3)

        # Save 5 models with different scores
        networks = []
        for i in range(5):
            net = _make_network()
            # Nudge weights so hashes differ
            with torch.no_grad():
                net.shared[0].weight.add_(torch.randn_like(net.shared[0].weight) * 0.01)
            networks.append(net)
            mgr.save(
                net,
                _dummy_persona(f"M{i}", age=i),
                _dummy_eval(vs_v1=0.1 * i, vs_v5=0.05 * i),
            )

        auto = mgr.list_auto()
        # Should have evicted down to 3
        assert len(auto) == 3

    def test_pinned_exempt_from_eviction(self, tmp_path: Path):
        mgr = HoFManager(tmp_path / "hof", max_auto=2)

        # Save 3 auto + 2 pinned
        for i in range(3):
            net = _make_network()
            with torch.no_grad():
                net.shared[0].weight.add_(torch.randn_like(net.shared[0].weight) * 0.01)
            mgr.save(net, _dummy_persona(f"Auto{i}", age=i), _dummy_eval(vs_v1=0.1 * i))

        for i in range(2):
            net = _make_network()
            with torch.no_grad():
                net.shared[0].weight.add_(torch.randn_like(net.shared[0].weight) * 0.01)
            mgr.save(
                net, _dummy_persona(f"Pinned{i}", age=10 + i),
                _dummy_eval(), pinned=True,
            )

        assert mgr.auto_count == 2  # evicted 1
        assert mgr.pinned_count == 2
        assert len(mgr.list()) == 4  # 2 pinned + 2 auto

    def test_pin_and_unpin(self, tmp_path: Path):
        mgr = HoFManager(tmp_path / "hof", max_auto=10)
        net = _make_network()
        info = mgr.save(net, _dummy_persona("PinMe"), _dummy_eval())
        h = info["model_hash"]

        assert mgr.pin(h) is True
        assert mgr.list_pinned()[0]["model_hash"] == h
        assert mgr.auto_count == 0

        # Pinned file should be in pinned/ dir
        assert (mgr.pinned_dir / info["filename"]).exists()

        # Unpin moves back
        assert mgr.unpin(h) is True
        assert mgr.pinned_count == 0
        assert mgr.auto_count == 1

    def test_remove(self, tmp_path: Path):
        mgr = HoFManager(tmp_path / "hof", max_auto=10)
        net = _make_network()
        info = mgr.save(net, _dummy_persona("Remove"), _dummy_eval())
        h = info["model_hash"]

        assert mgr.remove(h) is True
        assert len(mgr.list()) == 0
        assert mgr.remove(h) is False  # already gone

    def test_load_model(self, tmp_path: Path):
        mgr = HoFManager(tmp_path / "hof", max_auto=10)
        net = _make_network()
        info = mgr.save(net, _dummy_persona("Load"), _dummy_eval())
        h = info["model_hash"]

        result = mgr.load_model(h)
        assert result is not None
        loaded_net, metadata = result
        assert isinstance(loaded_net, TarokNet)

    def test_migrate_existing(self, tmp_path: Path):
        hof_dir = tmp_path / "hof"
        hof_dir.mkdir()
        # Write a manifest without 'pinned' field
        manifest = [
            {"filename": "hof_test.pt", "display_name": "test", "model_hash": "abcd1234"},
        ]
        (hof_dir / "manifest.json").write_text(json.dumps(manifest))

        mgr = HoFManager(hof_dir, max_auto=10)
        count = mgr.migrate_existing()
        assert count == 1
        entries = mgr.list()
        assert entries[0].get("pinned") is False

    def test_set_max_auto(self, tmp_path: Path):
        mgr = HoFManager(tmp_path / "hof", max_auto=10)
        for i in range(5):
            net = _make_network()
            with torch.no_grad():
                net.shared[0].weight.add_(torch.randn_like(net.shared[0].weight) * 0.01)
            mgr.save(net, _dummy_persona(f"N{i}", age=i), _dummy_eval(vs_v1=0.1 * i))

        assert mgr.auto_count == 5
        mgr.set_max_auto(3)
        assert mgr.auto_count == 3


# ── OpponentPool ─────────────────────────────────────────────────────────


class TestOpponentPool:
    def test_pure_selfplay(self):
        pool = OpponentPool()
        sp = PureSelfPlayOpponent(weight=1.0)
        pool.add(sp)

        chosen = pool.choose()
        assert chosen.name == "self-play"
        assert chosen.requires_external_experience() is False
        assert chosen.make_players() == []

    def test_weighted_selection(self):
        pool = OpponentPool(rng=__import__("random").Random(42))
        sp = PureSelfPlayOpponent(weight=0.7)
        sk = StockSkisOpponent(version=1, weight=0.3)
        pool.add(sp)
        pool.add(sk)

        counts = {"self-play": 0, "stockskis-v1": 0}
        for _ in range(1000):
            opp = pool.choose()
            counts[opp.name] += 1

        # Should roughly follow 70/30 split
        assert counts["self-play"] > 500
        assert counts["stockskis-v1"] > 150

    def test_stats_tracking(self):
        sp = PureSelfPlayOpponent()
        sp.record_result(OpponentGameResult(raw_score=50, won=True))
        sp.record_result(OpponentGameResult(raw_score=-30, won=False))

        assert sp.stats.games == 2
        assert sp.stats.wins == 1
        assert sp.stats.win_rate == 0.5

    def test_stockskis_opponent(self):
        sk = StockSkisOpponent(version=1, strength=1.0, weight=0.5)
        assert sk.name == "stockskis-v1"
        assert sk.is_available() is True
        assert sk.requires_external_experience() is True
        players = sk.make_players()
        assert len(players) == 3

    def test_hof_opponent_no_models(self, tmp_path: Path):
        hof = HoFOpponent(hof_dir=tmp_path / "empty_hof", weight=0.2)
        assert hof.is_available() is False
        assert hof.make_players() == []

    def test_hof_opponent_with_models(self, tmp_path: Path):
        hof_dir = tmp_path / "hof"
        hof_dir.mkdir()
        # Create a dummy HoF model file
        net = _make_network()
        torch.save({
            "model_state_dict": net.state_dict(),
            "hidden_size": 256,
        }, hof_dir / "hof_Test_Model_age1_abcd1234.pt")

        hof = HoFOpponent(hof_dir=hof_dir, weight=0.2)
        assert hof.is_available() is True
        players = hof.make_players(shared_network=net)
        assert len(players) == 3

    def test_hof_opponent_includes_pinned(self, tmp_path: Path):
        hof_dir = tmp_path / "hof"
        pinned_dir = hof_dir / "pinned"
        pinned_dir.mkdir(parents=True)
        # Only a pinned model exists
        net = _make_network()
        torch.save({
            "model_state_dict": net.state_dict(),
            "hidden_size": 256,
        }, pinned_dir / "hof_Pinned_Model_age1_pin12345.pt")

        hof = HoFOpponent(hof_dir=hof_dir, weight=0.2)
        assert hof.is_available() is True
        players = hof.make_players(shared_network=net)
        assert len(players) == 3

    def test_pool_stats_dict(self):
        pool = OpponentPool()
        sp = PureSelfPlayOpponent(weight=0.5)
        sk = StockSkisOpponent(version=1, weight=0.5)
        pool.add(sp)
        pool.add(sk)

        stats = pool.stats_dict()
        assert "self-play" in stats
        assert "stockskis-v1" in stats

    def test_pool_empty_fallback(self):
        pool = OpponentPool()
        chosen = pool.choose()
        # Should return a fallback pure self-play
        assert chosen.name == "self-play"

    def test_remove_opponent(self):
        pool = OpponentPool()
        pool.add(PureSelfPlayOpponent(weight=0.5))
        pool.add(StockSkisOpponent(version=1, weight=0.5))
        assert len(pool.opponents) == 2
        pool.remove("stockskis-v1")
        assert len(pool.opponents) == 1

    def test_fsp_opponent_not_ready(self):
        # FSP with an empty bank should not be available
        from tarok.adapters.ai.network_bank import NetworkBank
        bank = NetworkBank(max_size=5)
        fsp = FSPOpponent(bank, weight=0.3)
        assert fsp.is_available() is False

    def test_fsp_opponent_ready(self):
        from tarok.adapters.ai.network_bank import NetworkBank
        bank = NetworkBank(max_size=5)
        net = _make_network()
        bank.push(net.state_dict())
        fsp = FSPOpponent(bank, weight=0.3)
        assert fsp.is_available() is True
        players = fsp.make_players(shared_network=net)
        assert len(players) == 3
