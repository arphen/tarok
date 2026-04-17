"""Factory for compute backends — resolves device string to implementation.

Usage::

    from training.adapters.compute import create_backend

    backend = create_backend("auto")   # best available (CUDA > MPS > CPU)
    backend = create_backend("cpu")    # explicit CPU
    backend = create_backend("cuda")   # explicit CUDA
    backend = create_backend("cuda:1") # specific GPU
    backend = create_backend("mps")    # Apple Silicon GPU
"""

from __future__ import annotations

import torch

from training.adapters.compute.backend import ComputeBackend
from training.adapters.compute.cpu_backend import CpuBackend
from training.adapters.compute.gpu_backend import GpuBackend


def create(device: str = "auto") -> ComputeBackend:
    """Instantiate the appropriate ``ComputeBackend`` for *device*.

    ``"auto"`` probes for CUDA → MPS → CPU in that order.
    """
    resolved = _resolve(device)

    if resolved == "cpu":
        return CpuBackend()

    # cuda, cuda:N, mps
    return GpuBackend(device=resolved)


def _resolve(device: str) -> str:
    """Resolve ``"auto"`` and validate explicit device strings."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    kind = device.split(":")[0]
    if kind == "cpu":
        return "cpu"

    if kind == "cuda":
        if not torch.cuda.is_available():
            print(f"[compute] CUDA requested but unavailable — falling back to CPU")
            return "cpu"
        return device

    if kind == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print(f"[compute] MPS requested but unavailable — falling back to CPU")
            return "cpu"
        return device

    raise ValueError(
        f"Unknown compute device {device!r}. "
        f"Expected 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'."
    )
