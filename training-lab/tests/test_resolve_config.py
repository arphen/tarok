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
        "behavioral_clone_schedule": "linear",
        "behavioral_clone_coef_min": 0.1,
        "behavioral_clone_teacher": "bot_v5",
        "behavioral_clone_games_per_iteration": 321,
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="dummy.yaml")

    assert cfg.behavioral_clone_coef == 1.25
    assert cfg.behavioral_clone_schedule == "linear"
    assert cfg.behavioral_clone_coef_min == 0.1
    assert cfg.behavioral_clone_teacher == "bot_v5"
    assert cfg.behavioral_clone_games_per_iteration == 321


def test_resolve_config_reads_initial_league_calibration_fields() -> None:
    payload = {
        "league": {
            "enabled": True,
            "initial_calibration_enabled": True,
            "initial_calibration_games_per_pair": 3000,
            "initial_calibration_anchor": "V3",
            "initial_calibration_anchor_elo": 1500.0,
            "opponents": [
                {"name": "V3", "type": "bot_v3", "initial_elo": 1300},
                {"name": "V5", "type": "bot_v5", "initial_elo": 1600},
            ],
        },
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="dummy.yaml")

    assert cfg.league is not None
    assert cfg.league.initial_calibration_enabled is True
    assert cfg.league.initial_calibration_games_per_pair == 3000
    assert cfg.league.initial_calibration_anchor == "V3"
    assert cfg.league.initial_calibration_anchor_elo == 1500.0


def test_resolve_config_sets_profile_name_from_config_path_stem() -> None:
    cfg = ResolveConfig(_Loader({})).resolve(cli={}, config_path="configs/self-play.yaml")
    assert cfg.profile_name == "self-play"


def test_resolve_config_uses_custom_profile_name_without_config_path() -> None:
    cfg = ResolveConfig(_Loader({})).resolve(cli={}, config_path=None)
    assert cfg.profile_name == "custom"


def test_resolve_config_parses_centaur_knobs() -> None:
    payload = {
        "centaur_handoff_trick": 8,
        "centaur_pimc_worlds": 100,
        "centaur_endgame_solver": "pimc",
        "centaur_alpha_mu_depth": 3,
        "centaur_deterministic_seed": 42,
    }
    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="x.yaml")
    assert cfg.centaur_handoff_trick == 8
    assert cfg.centaur_pimc_worlds == 100
    assert cfg.centaur_endgame_solver == "pimc"
    assert cfg.centaur_alpha_mu_depth == 3
    assert cfg.centaur_deterministic_seed == 42


def test_resolve_config_centaur_knobs_default_to_none() -> None:
    cfg = ResolveConfig(_Loader({})).resolve(cli={}, config_path="x.yaml")
    assert cfg.centaur_handoff_trick is None
    assert cfg.centaur_pimc_worlds is None
    assert cfg.centaur_endgame_solver is None
    assert cfg.centaur_alpha_mu_depth is None
    assert cfg.centaur_deterministic_seed is None


def test_resolve_config_parses_duplicate_learner_seat_token() -> None:
    payload = {
        "duplicate": {
            "enabled": True,
            "learner_seat_token": "centaur",
        },
    }
    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="x.yaml")
    assert cfg.duplicate is not None
    assert cfg.duplicate.enabled is True
    assert cfg.duplicate.learner_seat_token == "centaur"


def test_resolve_config_reads_bid_entropy_coef() -> None:
    payload = {
        "entropy_coef": 0.01,
        "bid_entropy_coef": 0.03,
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="x.yaml")

    assert cfg.entropy_coef == 0.01
    assert cfg.bid_entropy_coef == 0.03


def test_resolve_config_parses_duplicate_negative_reward_multiplier() -> None:
    payload = {
        "duplicate": {
            "enabled": True,
            "negative_reward_multiplier": 3.5,
        },
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="x.yaml")

    assert cfg.duplicate.enabled is True
    assert cfg.duplicate.negative_reward_multiplier == 3.5


def test_resolve_config_parses_duplicate_berac_bid_penalty() -> None:
    payload = {
        "duplicate": {
            "enabled": True,
            "berac_bid_penalty": -10.0,
        },
    }

    cfg = ResolveConfig(_Loader(payload)).resolve(cli={}, config_path="x.yaml")

    assert cfg.duplicate.enabled is True
    assert cfg.duplicate.berac_bid_penalty == -10.0
