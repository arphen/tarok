"""Thread-safe experience ring buffer for async producer-consumer PPO."""

from __future__ import annotations

import threading
import time
from collections import deque

from training_lab.entities.experience import Experience, TaggedExperience


class ExperienceBuffer:
    """Thread-safe ring buffer for completed game experiences.

    The producer thread pushes completed game experiences with a policy
    version tag.  The consumer thread pulls batches when enough fresh
    experiences are available.  Stale experiences (from old policy versions)
    are discarded automatically.
    """

    def __init__(self, capacity: int = 50_000, max_staleness: int = 3):
        self.capacity = capacity
        self.max_staleness = max_staleness
        self._buffer: deque[TaggedExperience] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def push_game(self, experiences: list[Experience], policy_version: int) -> None:
        """Push all experiences from a completed game into the buffer."""
        now = time.monotonic()
        tagged = [
            TaggedExperience(
                experience=exp,
                policy_version=policy_version,
                collection_time=now,
            )
            for exp in experiences
        ]
        with self._lock:
            self._buffer.extend(tagged)
            self._not_empty.notify_all()

    def pull_batch(self, min_size: int, timeout: float = 30.0) -> list[TaggedExperience] | None:
        """Pull at least min_size experiences, blocking until available.

        Returns None if timeout expires before enough experiences accumulate.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            while len(self._buffer) < min_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

            batch = list(self._buffer)
            self._buffer.clear()
            return batch

    def discard_stale(self, current_version: int) -> int:
        """Remove experiences older than max_staleness updates.

        Returns the number of discarded experiences.
        """
        cutoff = current_version - self.max_staleness
        with self._lock:
            before = len(self._buffer)
            self._buffer = deque(
                (e for e in self._buffer if e.policy_version >= cutoff),
                maxlen=self.capacity,
            )
            return before - len(self._buffer)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0
