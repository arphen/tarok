"""Compute backend plugin system.

Provides a ``ComputeBackend`` interface that abstracts device placement,
network preparation, and batched inference.
"""

from tarok.core.compute.backend import ComputeBackend
from tarok.core.compute.factory import create as create_backend

__all__ = ["ComputeBackend", "create_backend"]
