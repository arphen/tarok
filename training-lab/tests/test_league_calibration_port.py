"""Tests for the league calibration port + its two default adapters."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from training.adapters.league_calibration import (
    DuplicateTournamentLeagueCalibrationAdapter,
    SelfPlayLeagueCalibrationAdapter,
)
from training.entities.league import LeagueConfig, LeagueOpponent, LeaguePool


class _FakeSelfPlay:
    def __init__(self, scores_by_seat0: dict[str, list[list[int]]]) -> None:
        self._scores = scores_by_seat0
        self.calls: list[str] = []

    def run(
        self,
        *,
        model_path: str,
        n_games: int,
        seat_config: str,
        explore_rate: float,
        concurrency: int,
        include_replay_data: bool = False,
        include_oracle_states: bool = False,
        lapajne_mc_worlds=None,
        lapajne_mc_sims=None,
        variant: int = 0,
    ):
        del model_path, n_games, explore_rate, concurrency
        del include_replay_data, include_oracle_states, lapajne_mc_worlds, lapajne_mc_sims
        del variant
        self.calls.append(seat_config)
        seat0 = seat_config.split(",", 1)[0]
        return {"scores": self._scores.get(seat0, self._scores["__default__"])}

    def compute_run_stats(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


# ---------- self-play adapter ----------


def _pool_with_bots() -> LeaguePool:
    return LeaguePool(
        config=LeagueConfig(
            enabled=True,
            elo_eval_games=4,
            opponents=(
                LeagueOpponent(name="V3", type="bot_v3", initial_elo=1300.0),
                LeagueOpponent(name="V5", type="bot_v5", initial_elo=1500.0),
                LeagueOpponent(name="M6", type="bot_m6", initial_elo=1700.0),
            ),
        ),
    )


def test_self_play_adapter_calibrate_candidate_returns_median():
    pool = _pool_with_bots()
    # For each run the candidate ("model.pt") sits at seat 0; learner and
    # target seat-1 scores drive the implied Elo via _ELO_PER_PLACEMENT.
    # Give every session equal scores so placements=(2.5, 2.5, 2.5, 2.5) and
    # implied = target.elo exactly.
    default = [[10, 10, 10, 10]] * 4
    sp = _FakeSelfPlay({"__default__": default})
    adapter = SelfPlayLeagueCalibrationAdapter(sp)

    result = adapter.calibrate_candidate(
        pool=pool,
        model_path="model.pt",
        concurrency=1,
        session_size=1,
        n_games_per_opponent=4,
    )
    # Three implied values = (1300, 1500, 1700); median = 1500
    assert result == pytest.approx(1500.0)


def test_self_play_adapter_calibrate_initial_empty_pool_returns_anchor():
    pool = LeaguePool(config=LeagueConfig(enabled=True, opponents=()))
    sp = _FakeSelfPlay({"__default__": [[0, 0, 0, 0]]})
    adapter = SelfPlayLeagueCalibrationAdapter(sp)
    ok = adapter.calibrate_initial(
        pool=pool,
        model_path="m.pt",
        concurrency=1,
        session_size=1,
        n_games_per_pair=4,
        anchor_name=None,
        anchor_elo=1234.0,
    )
    assert ok is True
    assert pool.learner_elo == pytest.approx(1234.0)


# ---------- duplicate-tournament adapter ----------


def _fake_te_module(score_fn):
    """Build a stub ``tarok_engine`` module whose run_self_play returns
    deterministic scores computed from ``score_fn(seat_config, deck_seeds)``.
    """

    class _TE:
        def __init__(self):
            self.calls: list[dict] = []

        def run_self_play(self, **kwargs):
            self.calls.append(kwargs)
            seat_cfg = kwargs["seat_config"]
            seeds = kwargs["deck_seeds"]
            n = len(seeds)
            scores = score_fn(seat_cfg, seeds)
            assert len(scores) == n
            return {"scores": scores}

    return _TE()


def test_duplicate_calibrate_candidate_median_from_paired_scores():
    pool = _pool_with_bots()

    # Deterministic rule: seat-0 score depends only on seat0 token.
    # Candidate (nn) beats V3 2/4, ties V5 2/4, loses to M6 2/4.
    def score(seat_cfg: str, seeds):
        seat0 = seat_cfg.split(",", 1)[0]
        rows = []
        for _ in seeds:
            if seat0 == "nn":
                rows.append([50, 0, 0, 0])
            elif seat0 == "bot_v3":
                rows.append([10, 0, 0, 0])
            elif seat0 == "bot_v5":
                rows.append([50, 0, 0, 0])
            else:  # bot_m6
                rows.append([100, 0, 0, 0])
        return rows

    fake_te = _fake_te_module(score)

    with patch(
        "training.adapters.league_calibration.duplicate_tournament_calibration.te",
        fake_te,
    ):
        adapter = DuplicateTournamentLeagueCalibrationAdapter(rng_seed=42)
        result = adapter.calibrate_candidate(
            pool=pool,
            model_path="learner.pt",
            concurrency=1,
            session_size=1,
            n_games_per_opponent=4,
        )

    # wr vs V3: 1.0  -> implied = 1300 + 250 * ( 1.0) = 1550
    # wr vs V5: 0.5  -> implied = 1500 + 250 * ( 0.0) = 1500
    # wr vs M6: 0.0  -> implied = 1700 + 250 * (-1.0) = 1450
    # median = 1500
    assert result == pytest.approx(1500.0)


def test_duplicate_calibrate_candidate_empty_pool_returns_none():
    pool = LeaguePool(config=LeagueConfig(enabled=True, opponents=()))
    adapter = DuplicateTournamentLeagueCalibrationAdapter(rng_seed=0)
    assert adapter.calibrate_candidate(
        pool=pool,
        model_path="m.pt",
        concurrency=1,
        session_size=1,
        n_games_per_opponent=4,
    ) is None


def test_duplicate_calibrate_initial_pins_anchor_and_derives_others():
    pool = _pool_with_bots()

    # Anchor=V3 at 1500. Rule: V5 beats V3 in 3/4 pairs; M6 ties V3.
    # (We only inspect per-opponent wr so exact score logic is simple.)
    def score(seat_cfg: str, seeds):
        seat0 = seat_cfg.split(",", 1)[0]
        rows = []
        for i, _ in enumerate(seeds):
            if seat0 == "bot_v3":
                rows.append([10, 0, 0, 0])
            elif seat0 == "bot_v5":
                # Win 3 of 4 vs v3 → seat0 score strictly greater on 3 out of 4
                rows.append([50 if i < 3 else 5, 0, 0, 0])
            elif seat0 == "bot_m6":
                # Tie v3 → same seat-0 score
                rows.append([10, 0, 0, 0])
            else:  # nn learner
                rows.append([20, 0, 0, 0])  # always beats v3 -> wr=1.0
        return rows

    fake_te = _fake_te_module(score)

    with patch(
        "training.adapters.league_calibration.duplicate_tournament_calibration.te",
        fake_te,
    ):
        adapter = DuplicateTournamentLeagueCalibrationAdapter(rng_seed=0)
        ok = adapter.calibrate_initial(
            pool=pool,
            model_path="learner.pt",
            concurrency=1,
            session_size=1,
            n_games_per_pair=4,
            anchor_name="V3",
            anchor_elo=1500.0,
        )

    assert ok is True
    by_name = {e.opponent.name: e.elo for e in pool.entries}
    assert by_name["V3"] == pytest.approx(1500.0)  # anchor pinned
    # V5 wr=0.75 → elo = 1500 + 250*(2*0.75-1) = 1500 + 125 = 1625
    assert by_name["V5"] == pytest.approx(1625.0)
    # M6 ties → wr=0.5 → elo = 1500
    assert by_name["M6"] == pytest.approx(1500.0)
    # Learner always wins vs V3 → wr=1.0 → elo = 1500 + 250 = 1750
    assert pool.learner_elo == pytest.approx(1750.0)
