"""Network Bank — historical checkpoint pool for Fictitious Self-Play.

Instead of always training against the current policy (pure self-play),
the agent plays a fraction of games against randomly sampled historical
checkpoints.  This prevents distribution collapse and cyclical non-transitive
learning pathologies.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Snapshot:
    """A stored network snapshot with identity metadata."""
    state_dict: dict[str, torch.Tensor]
    snapshot_id: str  # e.g. "snap-3#a1b2c3d4"
    index: int        # sequential push index


class NetworkBank:
    """Pool of historical network state_dicts for fictitious self-play."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._snapshots: deque[Snapshot] = deque(maxlen=max_size)
        self._push_count: int = 0

    @staticmethod
    def _hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
        """Compute a short hash from state_dict for identification."""
        h = hashlib.sha256()
        for key in sorted(state_dict.keys())[:3]:
            h.update(state_dict[key].cpu().numpy().tobytes()[:64])
        return h.hexdigest()[:8]

    def push(self, state_dict: dict[str, torch.Tensor]) -> str:
        """Store a deep copy of the current network weights.

        Returns the snapshot_id assigned to this snapshot.
        """
        self._push_count += 1
        sd = copy.deepcopy(state_dict)
        short_hash = self._hash_state_dict(sd)
        snapshot_id = f"snap-{self._push_count}#{short_hash}"
        self._snapshots.append(Snapshot(
            state_dict=sd,
            snapshot_id=snapshot_id,
            index=self._push_count,
        ))
        return snapshot_id

    def sample(self) -> tuple[dict[str, torch.Tensor], str] | tuple[None, str]:
        """Sample a random historical snapshot.

        Returns (state_dict, snapshot_id) or (None, "") if empty.
        """
        if not self._snapshots:
            return None, ""
        snap = random.choice(self._snapshots)
        return snap.state_dict, snap.snapshot_id

    @property
    def size(self) -> int:
        return len(self._snapshots)

    @property
    def is_ready(self) -> bool:
        """Has at least one snapshot for sampling."""
        return len(self._snapshots) > 0
