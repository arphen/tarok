"""Entity: DuplicateRunResult — the return value of a seeded-pods self-play run.

Produced by ``SelfPlayPort.run_seeded_pods`` (Phase 1) and consumed by
``CollectDuplicateExperiences`` (Phase 2). Analogous to the raw dict returned
by ``SelfPlayPort.run`` today but split into active-group experiences and
shadow-group terminal scores.

Numpy arrays are typed as ``Any`` so that this module stays free of numerical
library imports under the import-linter ``entities`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DuplicateRunResult:
    """Outcome of running a batch of duplicate pods.

    Attributes
    ----------
    active
        Experience dict for the active group only, schema identical to the
        dict returned by ``SelfPlayPort.run``. Only learner trajectories
        appear here (shadow runs do not emit experiences).
    shadow_scores
        Final scores of the shadow games, indexed so that
        ``shadow_scores[pod_idx, game_idx_within_pod, seat_idx]`` gives the
        per-seat score of the shadow table paired with active game
        ``(pod_idx, game_idx_within_pod)``. ``np.ndarray`` of shape
        ``(n_pods, n_games_per_group, 4)``.
    pod_ids
        Pod index (0..n_pods-1) for every step in ``active``. Lets the reward
        adapter look up the matching shadow score. ``np.ndarray`` of shape
        ``(n_active_steps,)``.
    learner_positions
        Seat index of the learner for each active game, keyed by
        ``(pod_idx, game_idx_within_pod)``. ``np.ndarray`` of shape
        ``(n_pods, n_games_per_group)``.
    active_game_ids
        Global game id for each ``(pod_idx, game_idx_within_pod)``. Same
        values you will see in ``active["game_ids"]``. ``np.ndarray`` of
        shape ``(n_pods, n_games_per_group)``.
    shadow_contracts
        Contract played by the shadow table paired with active game
        ``(pod_idx, game_idx_within_pod)``. ``np.ndarray`` of shape
        ``(n_pods, n_games_per_group)`` and dtype ``uint8``. ``None`` if
        the adapter did not capture this signal (legacy behaviour). Used
        by ``ShadowScoreRewardAdapter`` to detect bid-divergence between
        learner and shadow and redirect the reward away from the card
        head when the two pods played different contracts.
    """

    active: Any
    shadow_scores: Any
    pod_ids: Any
    learner_positions: Any
    active_game_ids: Any
    shadow_contracts: Any = None
