"""Port: duplicate reward — computes per-trajectory rewards from paired scores.

Given active-group experiences and matched shadow-group scores, produce a
per-step reward array whose terminal entries equal the empirical duplicate
advantage ``(R_learner − R_shadow) / 100`` (or a transformation thereof).

Adapters live under ``training.adapters.duplicate``. This port deliberately
keeps numpy out of its type signature to respect import-linter guardrails;
adapters take plain ``Any`` payloads and call numpy internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from training.entities.duplicate_pod import DuplicatePod


class DuplicateRewardPort(ABC):
    """Maps (active experiences, shadow scores, pods) → per-step reward array."""

    @abstractmethod
    def compute_rewards(
        self,
        active_raw: Any,
        shadow_scores: Any,
        pods: list["DuplicatePod"],
        shadow_contracts: Any = None,
    ) -> Any:
        """Return a per-step reward array of shape ``(n_active_steps,)``.

        Non-terminal steps receive ``0.0``; terminal steps receive the
        duplicate-advantage value for their ``(pod, learner-seat)`` pair.
        Downstream ``ppo_batch_preparation`` will feed this array as the
        ``precomputed_rewards`` override into the GAE input stream
        (conservative mode) or directly into ``_broadcast_terminal_advantage``
        (actor-only mode).

        ``shadow_contracts`` (optional) is a 2D ``np.ndarray`` of shape
        ``(n_pods, n_games_per_group)`` giving the contract that was
        played by the shadow table paired with each active game. When
        provided, the adapter MAY redirect the reward away from the
        terminal cardplay step for trajectories whose learner-side
        contract differs from the shadow-side contract — credit
        assignment then flows only to bidding-phase decision heads. When
        ``None`` (legacy behaviour) the adapter places the reward at the
        terminal cardplay step regardless of bid divergence.
        """
