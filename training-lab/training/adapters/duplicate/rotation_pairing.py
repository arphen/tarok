"""Rotation-based duplicate pairing adapter.

For each pod:

1. Sample a single ``deck_seed`` and a triple of opponent seat tokens.
2. Build 4 active seatings rotating the learner through seats 0..3, with
   the three opponents filling the remaining slots in the same cyclic order
   (so every opponent identity appears at every seat exactly once across
   the pod).
3. Build 4 matched shadow seatings by replacing the learner token with the
   shadow token at the same position.

The ``"rotation_4game"`` variant drops rotation and uses a single learner
seat position (seat 0). The ``"single_seat_2game"`` variant further drops
the active/shadow pairing to a single learner + single shadow game — useful
as a cheap debugging config.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from training.entities.duplicate_pod import DuplicatePod
from training.ports.duplicate_pairing_port import DuplicatePairingPort

if TYPE_CHECKING:
    from training.entities.league import LeagueConfig, LeaguePool


_DEFAULT_OPPONENT_ROSTER: tuple[str, ...] = ("bot_v5", "bot_v6", "bot_m6")


def _pool_opponent_tokens(pool: "LeaguePool | LeagueConfig | None") -> tuple[str, ...]:
    """Extract opponent seat tokens from live LeaguePool/config, or fall back.

    Prefer runtime ``LeaguePool.entries`` (which includes dynamic snapshot
    ghosts) when available; otherwise fall back to static config opponents.
    """
    if pool is None:
        return _DEFAULT_OPPONENT_ROSTER

    # Runtime league pool path: entries include heuristic anchors and
    # nn_checkpoint snapshots added during training.
    entries = getattr(pool, "entries", None)
    if entries:
        tokens: list[str] = []
        for entry in entries:
            opp = getattr(entry, "opponent", None)
            if opp is None:
                continue
            token = opp.seat_token() if hasattr(opp, "seat_token") else None
            if token:
                tokens.append(str(token))
        if tokens:
            return tuple(tokens)

    # Static config path.
    opponents = getattr(pool, "opponents", ())
    if opponents:
        tokens = []
        for opp in opponents:
            path = getattr(opp, "path", None)
            token = path if path else getattr(opp, "type", None)
            if token:
                tokens.append(str(token))
        if tokens:
            return tuple(tokens)

    return _DEFAULT_OPPONENT_ROSTER


class RotationPairingAdapter(DuplicatePairingPort):
    """Default pairing: 8-game rotation (4 active + 4 shadow).

    ``max_opponent_triplets`` caps how many distinct ordered opponent
    triplets are sampled per ``build_pods`` call. Pods are then assigned a
    triplet round-robin. This is a pure performance knob: the Rust
    self-play adapter groups pods by seating before calling the engine, so
    fewer distinct triplets → fewer ``run_self_play`` invocations → fewer
    TorchScript model reloads. With 400 pods and a 4-bot league, the
    uncapped sampler produces up to 24 distinct ordered triplets per
    iteration; each one becomes its own Rust call (×4 rotation variants ×
    2 for active+shadow ≈ 192 model reloads). Capping at 4 cuts that to
    32 — a ~6× throughput win with no effect on duplicate-invariant
    semantics (same deck seed still drives the deal regardless of seating).

    Set ``max_opponent_triplets=None`` to restore the legacy per-pod
    independent sampling behaviour.
    """

    def __init__(
        self,
        pairing: str = "rotation_8game",
        max_opponent_triplets: int | None = 4,
    ) -> None:
        if pairing not in {
            "rotation_8game",
            "rotation_4game",
            "single_seat_2game",
            "rotation_6game",
        }:
            raise ValueError(f"Unknown pairing mode: {pairing!r}")
        if max_opponent_triplets is not None and max_opponent_triplets < 1:
            raise ValueError(
                f"max_opponent_triplets must be >= 1 or None, got {max_opponent_triplets}"
            )
        self._pairing = pairing
        self._max_opponent_triplets = max_opponent_triplets

    @property
    def _n_seats(self) -> int:
        return 3 if self._pairing == "rotation_6game" else 4

    @property
    def _n_opponents(self) -> int:
        return self._n_seats - 1

    def build_pods(
        self,
        pool: "LeaguePool | LeagueConfig | None",
        learner_seat_token: str,
        shadow_seat_token: str,
        n_pods: int,
        rng_seed: int,
    ) -> list[DuplicatePod]:
        if n_pods < 0:
            raise ValueError(f"n_pods must be >= 0, got {n_pods}")
        if n_pods == 0:
            return []

        rng = random.Random(rng_seed)
        opponent_tokens = _pool_opponent_tokens(pool)
        n_opp = self._n_opponents
        if len(opponent_tokens) < n_opp:
            # Pad by repetition so we can always fill the opponent slots.
            opponent_tokens = (
                tuple((opponent_tokens * n_opp)[:n_opp])
                if opponent_tokens
                else _DEFAULT_OPPONENT_ROSTER[:n_opp]
            )

        def _sample_triplet() -> tuple[str, ...]:
            if len(opponent_tokens) >= n_opp:
                return tuple(rng.sample(list(opponent_tokens), n_opp))
            return tuple(rng.choices(list(opponent_tokens), k=n_opp))

        # Pre-sample a bounded pool of ordered triplets so the downstream
        # Rust runner can batch pods sharing the same seating into a single
        # `run_self_play` invocation. See class docstring for rationale.
        if self._max_opponent_triplets is None:
            triplet_pool: list[tuple[str, ...]] | None = None
        else:
            pool_size = min(self._max_opponent_triplets, n_pods)
            triplet_pool = [_sample_triplet() for _ in range(pool_size)]

        pods: list[DuplicatePod] = []
        for pod_idx in range(n_pods):
            deck_seed = rng.getrandbits(64)
            if triplet_pool is None:
                opponents: tuple[str, ...] = _sample_triplet()
            else:
                opponents = triplet_pool[pod_idx % len(triplet_pool)]

            active_seatings, shadow_seatings, learner_positions = self._build_seatings(
                opponents=opponents,
                learner_token=learner_seat_token,
                shadow_token=shadow_seat_token,
            )
            pods.append(
                DuplicatePod(
                    deck_seed=int(deck_seed),
                    opponents=opponents,
                    active_seatings=active_seatings,
                    shadow_seatings=shadow_seatings,
                    learner_positions=learner_positions,
                )
            )
        return pods

    def _build_seatings(
        self,
        opponents: tuple[str, ...],
        learner_token: str,
        shadow_token: str,
    ) -> tuple[
        tuple[tuple[str, ...], ...],
        tuple[tuple[str, ...], ...],
        tuple[int, ...],
    ]:
        if self._pairing == "rotation_8game":
            positions: tuple[int, ...] = (0, 1, 2, 3)
        elif self._pairing == "rotation_4game":
            positions = (0, 0)  # two active+shadow games at the same seat
        elif self._pairing == "rotation_6game":
            positions = (0, 1, 2)  # 3 active + 3 shadow = 6 games (ThreePlayer)
        else:  # single_seat_2game
            positions = (0,)

        n_seats = self._n_seats
        active: list[tuple[str, ...]] = []
        shadow: list[tuple[str, ...]] = []
        for pos in positions:
            seats_active: list[str] = [""] * n_seats
            seats_shadow: list[str] = [""] * n_seats
            opp_iter = iter(opponents)
            for i in range(n_seats):
                if i == pos:
                    seats_active[i] = learner_token
                    seats_shadow[i] = shadow_token
                else:
                    tok = next(opp_iter)
                    seats_active[i] = tok
                    seats_shadow[i] = tok
            active.append(tuple(seats_active))
            shadow.append(tuple(seats_shadow))
        return tuple(active), tuple(shadow), tuple(positions)
