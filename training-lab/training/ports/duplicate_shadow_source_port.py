"""Port: duplicate shadow-source — which frozen policy plays the shadow seat.

The duplicate-RL loop replays each deal with the learner replaced by a
frozen "shadow" policy. The choice of shadow policy is a research axis —
separable from the pairing strategy and the reward model — so it is
exposed as its own port.

Adapters live under :mod:`training.adapters.duplicate.shadow_sources`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.entities.league import LeaguePool


class DuplicateShadowSourcePort(ABC):
    """Resolves the TorchScript path used for the shadow seat each iteration."""

    @abstractmethod
    def resolve(
        self,
        *,
        iteration: int,
        learner_ts_path: str,
        pool: "LeaguePool | None",
    ) -> str:
        """Return the TorchScript path for the shadow policy at ``iteration``.

        ``learner_ts_path`` is the current iteration's exported learner —
        adapters should fall back to it when they have no prior artefact
        (for example on iteration 0, before any snapshot has been frozen).

        ``pool`` is the live :class:`LeaguePool` (or ``None`` when league
        play is disabled). Adapters that need snapshot metadata (Elo,
        games-played, paths) pull them from here.
        """
