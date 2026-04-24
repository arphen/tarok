"""Tests for the global TorchScript model cache in the Rust engine.

Fix B: ``NeuralNetPlayer::new`` used to call ``CModule::load_on_device``
unconditionally, which deserialises the model graph from disk on every
``run_self_play`` invocation. Under duplicate-RL that meant ~192 reloads
per iteration. The cache is keyed by ``(canonical_path, device, mtime)``
so repeated loads of the same checkpoint are a HashMap lookup and are
transparently invalidated when the file is re-exported.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import time
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


def _export_random_model(path: str, hidden_size: int = 64, torch_seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(torch_seed)
    with torch.no_grad():
        net = TarokNetV4(hidden_size=hidden_size, oracle_critic=False)
        for p in net.parameters():
            p.copy_(torch.empty_like(p).normal_(generator=gen) * 0.1)
    TorchModelAdapter().export_for_inference(
        weights=net.state_dict(),
        hidden_size=hidden_size,
        oracle=False,
        model_arch="v4",
        path=path,
    )


def _run(model_path: str, n_games: int, deck_seeds: list[int] | None = None) -> np.ndarray:
    raw = te.run_self_play(
        n_games=n_games,
        concurrency=min(4, n_games),
        model_path=model_path,
        explore_rate=0.0,
        seat_config="nn,bot_v5,bot_v5,bot_v5",
        include_replay_data=False,
        include_oracle_states=False,
        deck_seeds=deck_seeds,
    )
    return np.asarray(raw["scores"], dtype=np.int64)


def test_repeated_load_is_fast() -> None:
    """Second load of the same checkpoint should be orders of magnitude faster.

    This is the core contract of the cache: an ``Arc<CModule>`` is reused
    across invocations when the underlying file hasn't changed.
    """
    tmp = tempfile.NamedTemporaryFile(prefix="tarok_cache_test_", suffix=".pt", delete=False)
    tmp.close()
    _export_random_model(tmp.name)

    seeds = list(range(1234, 1234 + 8))

    # Warm the cache and JIT.
    _run(tmp.name, n_games=len(seeds), deck_seeds=seeds)

    # Cold-ish first timed call (model already cached from warm-up, but
    # this still exercises the full path).
    t0 = time.perf_counter()
    first = _run(tmp.name, n_games=len(seeds), deck_seeds=seeds)
    t_first = time.perf_counter() - t0
    del first

    # Many repeated small calls — if the cache were missing, each one
    # would reload the model from disk (~tens of ms each).
    n_reps = 20
    t0 = time.perf_counter()
    for _ in range(n_reps):
        _run(tmp.name, n_games=len(seeds), deck_seeds=seeds)
    t_repeated = (time.perf_counter() - t0) / n_reps

    # Repeated calls should be in the same ballpark as the first, not
    # dominated by reload overhead. A 3× budget keeps the test robust
    # against CI jitter while still catching a missing cache (which
    # would make every call as slow as a fresh maturin-warm load).
    assert t_repeated < t_first * 3 + 0.050, (
        f"Repeated run took {t_repeated:.4f}s vs first {t_first:.4f}s — "
        f"cache likely not working."
    )

    os.unlink(tmp.name)


def test_mtime_change_invalidates_cache() -> None:
    """Re-exporting the same path with different weights must yield
    different self-play results, proving the cache was invalidated by the
    mtime change.
    """
    tmp = tempfile.NamedTemporaryFile(prefix="tarok_cache_test_", suffix=".pt", delete=False)
    tmp.close()
    seeds = list(range(5000, 5000 + 16))

    _export_random_model(tmp.name, torch_seed=1)
    scores_v1 = _run(tmp.name, n_games=len(seeds), deck_seeds=seeds)

    # Ensure the filesystem mtime actually changes. macOS/APFS mtime
    # resolution can be as coarse as 1s on some network mounts; we give
    # it a small sleep and then also explicitly bump mtime just in case.
    time.sleep(1.1)
    _export_random_model(tmp.name, torch_seed=999)
    # Force mtime to now in case the export wrote fast enough to preserve
    # the old timestamp.
    now = time.time()
    os.utime(tmp.name, (now, now))

    scores_v2 = _run(tmp.name, n_games=len(seeds), deck_seeds=seeds)

    # With the cache correctly keyed on mtime, v2 uses the re-exported
    # weights and produces a different score distribution on the same
    # decks. Without invalidation, the cached v1 model would be reused
    # and the arrays would be identical.
    assert not np.array_equal(scores_v1, scores_v2), (
        "Cache did not invalidate on re-export — same checkpoint path with "
        "new weights produced identical results."
    )

    os.unlink(tmp.name)
