"""Shared checkpoint path resolution for all routers.

The backend runs from ``backend/`` CWD; the canonical checkpoint store is the
top-level ``data/checkpoints/`` directory.

Supported token forms
---------------------
- ``"PersonaName"``              →  ../data/checkpoints/PersonaName/_current.pt
- ``"PersonaName/_current.pt"``  →  ../data/checkpoints/PersonaName/_current.pt
- ``"hall_of_fame/foo.pt"``      →  ../data/checkpoints/hall_of_fame/foo.pt
- other relative path            →  tried under ../data/checkpoints/, then as literal
"""
from __future__ import annotations

from pathlib import Path

_ROOT_CKPT_DIR = Path("../data/checkpoints")


def resolve_checkpoint(token: str) -> Path | None:
    """Return the resolved Path for *token*, or ``None`` if not found."""
    if not token:
        return None

    root = _ROOT_CKPT_DIR

    # Bare persona name (no extension, no slash) → PersonaName/_current.pt
    if "/" not in token and not token.endswith(".pt"):
        candidate = root / token / "_current.pt"
        return candidate if candidate.exists() else None

    # Relative path rooted under top-level checkpoints/
    candidate = root / token
    if candidate.exists():
        return candidate

    # Literal/absolute path fallback
    literal = Path(token)
    return literal if literal.exists() else None


def resolve_checkpoint_or_default(token: str | None, default_persona: str = "training_run") -> Path | None:
    """Resolve *token*, or fall back to *default_persona*/_current.pt."""
    if token:
        path = resolve_checkpoint(token)
        if path:
            return path
    return resolve_checkpoint(default_persona)
