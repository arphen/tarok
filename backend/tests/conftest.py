"""Pytest configuration for backend tests.

On macOS, DYLD_FALLBACK_LIBRARY_PATH is stripped by SIP when crossing process
trust boundaries (e.g. uv run → python subprocess). This conftest preloads the
torch shared libraries via ctypes so that tarok_engine can dlopen libtorch_cpu.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys


def _preload_torch_libs() -> None:
    """Load all .dylib files from torch/lib before tarok_engine is imported."""
    if sys.platform != "darwin":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            return
        torch_lib_dir = pathlib.Path(spec.origin).resolve().parent / "lib"
        if not torch_lib_dir.is_dir():
            return
        for dylib in sorted(torch_lib_dir.glob("*.dylib")):
            try:
                ctypes.CDLL(str(dylib))
            except OSError:
                pass
    except Exception:
        pass


_preload_torch_libs()
