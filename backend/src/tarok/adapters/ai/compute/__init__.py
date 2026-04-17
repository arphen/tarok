"""Backward-compat re-exports — use tarok_model.compute instead."""
from tarok_model.compute import ComputeBackend, create_backend

__all__ = ["ComputeBackend", "create_backend"]
