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
    """Default pairing: 8-game rotation (4 active + 4 shadow)."""

    def __init__(self, pairing: str = "rotation_8game") -> None:
        if pairing not in {"rotation_8game", "rotation_4game", "single_seat_2game"}:
            raise ValueError(f"Unknown pairing mode: {pairing!r}")
        self._pairing = pairing

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
        if len(opponent_tokens) < 3:
            # Pad by repetition so we can always fill 3 opponent slots.
            opponent_tokens = tuple((opponent_tokens * 3)[:3]) if opponent_tokens else (
                _DEFAULT_OPPONENT_ROSTER
            )

        pods: list[DuplicatePod] = []
        for pod_idx in range(n_pods):
            deck_seed = rng.getrandbits(64)
            # Sample 3 opponents without replacement when the roster allows; otherwise
            # fall back to a random sample with replacement.
            if len(opponent_tokens) >= 3:
                opponents = tuple(rng.sample(list(opponent_tokens), 3))
            else:
                opponents = tuple(rng.choices(list(opponent_tokens), k=3))

            active_seatings, shadow_seatings, learner_positions = self._build_seatings(
                opponents=opponents,
                learner_token=learner_seat_token,
                shadow_token=shadow_seat_token,
            )
            pods.append(
                DuplicatePod(
                    deck_seed=int(deck_seed),
                    opponents=opponents,  # type: ignore[arg-type]
                    active_seatings=active_seatings,
                    shadow_seatings=shadow_seatings,
                    learner_positions=learner_positions,
                )
            )
        return pods

    def _build_seatings(
        self,
        opponents: tuple[str, str, str],
        learner_token: str,
        shadow_token: str,
    ) -> tuple[
        tuple[tuple[str, str, str, str], ...],
        tuple[tuple[str, str, str, str], ...],
        tuple[int, ...],
    ]:
        if self._pairing == "rotation_8game":
            positions = (0, 1, 2, 3)
        elif self._pairing == "rotation_4game":
            positions = (0, 0)  # two active+shadow games at the same seat
        else:  # single_seat_2game
            positions = (0,)

        active: list[tuple[str, str, str, str]] = []
        shadow: list[tuple[str, str, str, str]] = []
        a, b, c = opponents
        for pos in positions:
            seats_active: list[str] = [a, b, c, a]  # placeholder; will be overwritten
            seats_shadow: list[str] = [a, b, c, a]
            # Fill all 4 seats: learner goes to `pos`, the 3 opponents fill the other 3
            # seats in fixed cyclic order (a, b, c).
            opps_cycle = [a, b, c]
            opp_iter = iter(opps_cycle)
            for i in range(4):
                if i == pos:
                    seats_active[i] = learner_token
                    seats_shadow[i] = shadow_token
                else:
                    tok = next(opp_iter)
                    seats_active[i] = tok
                    seats_shadow[i] = tok
            active.append((seats_active[0], seats_active[1], seats_active[2], seats_active[3]))
            shadow.append((seats_shadow[0], seats_shadow[1], seats_shadow[2], seats_shadow[3]))
        return tuple(active), tuple(shadow), tuple(positions)
