"""Compute backend plugin system — clean architecture dependency injection.

Provides a ``ComputeBackend`` interface that abstracts device placement,
network preparation, and batched inference.  Implementations live in
sibling modules (``cpu_backend``, ``gpu_backend``).  Use ``factory.create``
to get the right backend at runtime.
"""

from tarok.adapters.ai.compute.backend import ComputeBackend
from tarok.adapters.ai.compute.factory import create as create_backend

__all__ = ["ComputeBackend", "create_backend"]
