"""Tests for ResolveConfig defaults around outplace/session Elo weighting."""

from __future__ import annotations

from training.use_cases.resolve_config import ResolveConfig


class _Loader:
    def __init__(self, payload: dict):
        self._payload = payload

    def load(self, _path: str) -> dict:
        return self._payload


def test_resolve_config_defaults_league_elo_weight_to_session_size() -> None:
    payload = {
        "games": 10000,
        "outplace_session_size": 50,
        "league": {
            "enabled": True,
            "opponents": [
                {"name": "V5", "type": "bot_v5", "initial_elo": 1600},
            ],
        },
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="dummy.yaml")

    assert cfg.league is not None
    assert cfg.league.elo_outplace_unit_weight == 50.0


def test_resolve_config_league_elo_weight_can_be_overridden_explicitly() -> None:
    payload = {
        "outplace_session_size": 50,
        "league": {
            "enabled": True,
            "elo_outplace_unit_weight": 12.5,
            "opponents": [
                {"name": "V5", "type": "bot_v5", "initial_elo": 1600},
            ],
        },
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="dummy.yaml")

    assert cfg.league is not None
    assert cfg.league.elo_outplace_unit_weight == 12.5


def test_resolve_config_reads_policy_coef() -> None:
    payload = {
        "policy_coef": 0.0,
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="dummy.yaml")

    assert cfg.policy_coef == 0.0


def test_resolve_config_reads_behavioral_cloning_fields() -> None:
    payload = {
        "behavioral_clone_coef": 1.25,
        "behavioral_clone_teacher": "bot_v5",
        "behavioral_clone_games_per_iteration": 321,
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="dummy.yaml")

    assert cfg.behavioral_clone_coef == 1.25
    assert cfg.behavioral_clone_teacher == "bot_v5"
    assert cfg.behavioral_clone_games_per_iteration == 321
