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


def _export_random_model(hidden_size: int, torch_seed: int = 0) -> str:
    """Export a deterministically-initialised TorchScript model and return its path."""
    gen = torch.Generator().manual_seed(torch_seed)
    with torch.no_grad():
        net = TarokNetV4(hidden_size=hidden_size, oracle_critic=False)
        for p in net.parameters():
            # Re-sample each parameter with the seeded generator so the
            # test's model weights are reproducible regardless of global
            # torch RNG state.
            p.copy_(torch.empty_like(p).normal_(generator=gen) * 0.1)
    weights = net.state_dict()
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


def test_lustrek_outplaces_m6_on_matched_hands() -> None:
    """Duplicate-style comparison: Lustrek should beat M6 on the same deals.

    We run the two seat configurations against identical opponents and with
    identical ``deck_seeds`` so the only difference is the seat-0 bot policy.
    This removes dealing variance and makes the test robust.
    """
    n_games = 400
    rng = np.random.default_rng(12345)
    deck_seeds = rng.integers(0, 2**63 - 1, size=n_games, dtype=np.uint64).tolist()

    lustrek_scores = _run_games(
        model_path="",
        seat_config="bot_lustrek,bot_v5,bot_v5,bot_v5",
        n_games=n_games,
        deck_seeds=deck_seeds,
    )
    m6_scores = _run_games(
        model_path="",
        seat_config="bot_m6,bot_v5,bot_v5,bot_v5",
        n_games=n_games,
        deck_seeds=deck_seeds,
    )

    diff = lustrek_scores[:, 0] - m6_scores[:, 0]
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    mean_diff = float(diff.mean())

    assert wins > losses, (
        f"Expected Lustrek to win more matched hands than M6; "
        f"wins={wins}, losses={losses}, mean_diff={mean_diff:.2f}"
    )
    assert mean_diff > 0.0, (
        f"Expected Lustrek to have positive mean score delta vs M6; "
        f"mean_diff={mean_diff:.2f}, wins={wins}, losses={losses}"
    )


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
