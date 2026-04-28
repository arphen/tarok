"""Tarok backend package initialization.

Import torch early so its bundled libtorch dylibs are loaded before the
Rust extension module (tarok_engine) is imported on macOS.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except Exception:
    # Keep package importable for tooling; tarok_engine import will still
    # surface a clear runtime error if torch is really unavailable.
    pass
