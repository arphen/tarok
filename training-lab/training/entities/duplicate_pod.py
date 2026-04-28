"""Entity: DuplicatePod describes the 8-game (or 4-/2-game) schedule for a single deck.

A pod is the atomic unit of duplicate self-play: one seeded deal played at both
an *active* table (learner vs. opponents) and a *shadow* table (frozen learner
snapshot vs. the same opponents). The difference in scores is the empirical
duplicate advantage used as the PPO reward.

This dataclass is a pure value object. It must remain free of infrastructure,
serialisation, and numerical-library imports so that it can be referenced from
``training.use_cases`` under the import-linter contracts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DuplicatePod:
    """One seeded deal played at two tables with rotated learner seating.

    Attributes
    ----------
    deck_seed
        Deterministic seed for the deal. Two games that share ``deck_seed``
        receive the byte-for-byte identical deck in the Rust engine.
    opponents
        Seat tokens for the three non-learner seats. These are sampled once
        per pod from the league pool and held constant across all games in
        the pod (so that luck from opponent identity also cancels).
    active_seatings
        Rotated seat configurations in which the learner plays. Each tuple is
        a 4-tuple of seat tokens (one of which equals the ``learner`` token).
    shadow_seatings
        Matched seat configurations in which the learner token has been
        replaced by the shadow (frozen snapshot) token. ``shadow_seatings[i]``
        is paired with ``active_seatings[i]`` — same deck, same opponent
        positions.
    learner_positions
        Seat indices (0..3) in each ``active_seatings[i]`` where the learner
        sits. Stored explicitly so the reward adapter does not have to
        re-parse seat tokens.
    """

    deck_seed: int
    opponents: tuple[str, ...]
    active_seatings: tuple[tuple[str, ...], ...]
    shadow_seatings: tuple[tuple[str, ...], ...]
    learner_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        n_active = len(self.active_seatings)
        n_shadow = len(self.shadow_seatings)
        n_pos = len(self.learner_positions)
        if n_active != n_shadow:
            raise ValueError(
                f"active_seatings ({n_active}) and shadow_seatings ({n_shadow}) "
                "must be the same length"
            )
        if n_pos != n_active:
            raise ValueError(
                f"learner_positions ({n_pos}) must match active_seatings ({n_active})"
            )
        # Seat count is inferred from the first seating; supports 3 or 4
        # seats (FourPlayer / ThreePlayer variants).
        if not self.active_seatings:
            return
        n_seats = len(self.active_seatings[0])
        if n_seats not in (3, 4):
            raise ValueError(f"seating must have 3 or 4 seats, got {n_seats}")
        if len(self.opponents) != n_seats - 1:
            raise ValueError(
                f"opponents must have {n_seats - 1} entries for a {n_seats}-seat pod, "
                f"got {len(self.opponents)}"
            )
        for seating in self.active_seatings:
            if len(seating) != n_seats:
                raise ValueError(
                    f"Each seating must have exactly {n_seats} seats, got {len(seating)}"
                )
        for seating in self.shadow_seatings:
            if len(seating) != n_seats:
                raise ValueError(
                    f"Each seating must have exactly {n_seats} seats, got {len(seating)}"
                )
        for pos in self.learner_positions:
            if not (0 <= pos < n_seats):
                raise ValueError(
                    f"learner_positions entries must be in 0..{n_seats - 1}, got {pos}"
                )

    @property
    def n_seats(self) -> int:
        if not self.active_seatings:
            return 4
        return len(self.active_seatings[0])

    @property
    def n_games_per_group(self) -> int:
        return len(self.active_seatings)
