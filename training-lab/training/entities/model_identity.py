"""ModelIdentity dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    hidden_size: int
    oracle_critic: bool
    is_new: bool
