"""Compute backend plugin system.

Provides a ``ComputeBackend`` interface that abstracts device placement,
network preparation, and batched inference for training.
"""

from training.adapters.compute.backend import ComputeBackend
from training.adapters.compute.factory import create as create_backend

__all__ = ["ComputeBackend", "create_backend"]
