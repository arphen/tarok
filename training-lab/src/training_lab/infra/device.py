"""Device detection and logging helpers."""

from __future__ import annotations

import logging

import torch


def detect_device() -> str:
    """Detect the best available compute device.

    Returns 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for training-lab."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
