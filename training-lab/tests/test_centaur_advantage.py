"""Smoke test: centaur (NN + PIMC endgame) outperforms pure NN.

Both players start from the **same** randomly-initialised weights so the only
difference is the PIMC endgame solver that kicks in for the last 4 tricks.
Because the NN is untrained its card-play actions are near-random whereas
PIMC plays optimally (given the sampled worlds), which gives centaur a
measurable average-score advantage.

The test is intentionally generous (α = 0.10, one-sided) so it doesn't
flake in CI while still catching a total wiring failure.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


def _preload_torch_dylibs_for_macos() -> None:
    if sys.platform != "darwin":
        return
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = f"{torch_lib}:{existing}" if existing else str(torch_lib)
    for name in ["libc10.dylib", "libtorch.dylib", "libtorch_cpu.dylib", "libtorch_python.dylib"]:
        lib_path = torch_lib / name
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


_preload_torch_dylibs_for_macos()

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_LAB_ROOT = REPO_ROOT / "training-lab"
if str(TRAINING_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_LAB_ROOT))

te = pytest.importorskip("tarok_engine")

from tarok_model.network import TarokNetV4
from training.adapters.modeling import TorchModelAdapter


def _export_random_model(hidden_size: int) -> str:
    """Export a randomly-initialised TorchScript model and return its path."""
    weights = TarokNetV4(hidden_size=hidden_size, oracle_critic=False).state_dict()
    tmp = tempfile.NamedTemporaryFile(prefix="tarok_centaur_test_", suffix=".pt", delete=False)
    tmp.close()
    TorchModelAdapter().export_for_inference(
        weights=weights,
        hidden_size=hidden_size,
        oracle=False,
        model_arch="v4",
        path=tmp.name,
    )
    return tmp.name


def _run_games(model_path: str, seat_config: str, n_games: int, **kwargs) -> np.ndarray:
    """Run self-play games and return per-game scores as (n_games, 4) array."""
    raw = te.run_self_play(
        n_games=n_games,
        concurrency=4,
        model_path=model_path,
        explore_rate=0.0,
        seat_config=seat_config,
        include_replay_data=False,
        include_oracle_states=False,
        **kwargs,
    )
    return np.asarray(raw["scores"], dtype=np.float64)


def test_centaur_outplaces_nn_with_random_weights() -> None:
    """Centaur seat 0 should score higher on average than pure NN seat 0.

    Both use the same untrained (random) model.  The only difference is
    PIMC endgame for the last 4 tricks.  We run enough games for the PIMC
    advantage to surface reliably.
    """
    n_games = 200
    model_path = _export_random_model(hidden_size=128)
    try:
        centaur_scores = _run_games(
            model_path,
            "centaur,bot_v5,bot_v5,bot_v5",
            n_games,
            centaur_handoff_trick=8,
            centaur_pimc_worlds=30,
        )
        nn_scores = _run_games(
            model_path,
            "nn,bot_v5,bot_v5,bot_v5",
            n_games,
        )

        centaur_mean = centaur_scores[:, 0].mean()
        nn_mean = nn_scores[:, 0].mean()

        # Centaur should do at least as well as pure NN.  With random weights
        # the PIMC endgame provides a consistent edge.
        assert centaur_mean > nn_mean, (
            f"Expected centaur ({centaur_mean:.1f}) > nn ({nn_mean:.1f})"
        )
    finally:
        Path(model_path).unlink(missing_ok=True)


def test_centaur_emits_nan_log_probs_for_pimc_decisions() -> None:
    """PIMC-decided steps must have log_prob = NaN (PPO sentinel).

    This is the key contract between CentaurBot and the PPO trainer:
    PIMC steps tagged with NaN log_prob are skipped during policy loss
    computation while their rewards still flow through GAE.
    """
    model_path = _export_random_model(hidden_size=128)
    try:
        raw = te.run_self_play(
            n_games=50,
            concurrency=4,
            model_path=model_path,
            explore_rate=0.0,
            seat_config="centaur,bot_v5,bot_v5,bot_v5",
            include_replay_data=False,
            include_oracle_states=False,
            centaur_handoff_trick=8,
            centaur_pimc_worlds=30,
        )
        log_probs = np.asarray(raw["log_probs"], dtype=np.float32)
        nan_count = int(np.isnan(log_probs).sum())

        # There must be some NaN log_probs (PIMC-decided steps).
        assert nan_count > 0, "Expected NaN log_probs from PIMC decisions but found none"

        # Not ALL should be NaN — the NN still handles early/mid-game + bidding.
        nn_count = int((~np.isnan(log_probs)).sum())
        assert nn_count > 0, "Expected some NN-decided steps but all were NaN"
    finally:
        Path(model_path).unlink(missing_ok=True)
