"""Network Bank — historical checkpoint pool for Fictitious Self-Play.

Instead of always training against the current policy (pure self-play),
the agent plays a fraction of games against randomly sampled historical
checkpoints.  This prevents distribution collapse and cyclical non-transitive
learning pathologies.
"""

from __future__ import annotations

import copy
import random
from collections import deque

import torch


class NetworkBank:
    """Pool of historical network state_dicts for fictitious self-play."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._snapshots: deque[dict[str, torch.Tensor]] = deque(maxlen=max_size)

    def push(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Store a deep copy of the current network weights."""
        self._snapshots.append(copy.deepcopy(state_dict))

    def sample(self) -> dict[str, torch.Tensor] | None:
        """Sample a random historical snapshot. Returns None if empty."""
        if not self._snapshots:
            return None
        return random.choice(self._snapshots)

    @property
    def size(self) -> int:
        return len(self._snapshots)

    @property
    def is_ready(self) -> bool:
        """Has at least one snapshot for sampling."""
        return len(self._snapshots) > 0
