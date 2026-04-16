"""Use case: sample league seats for a single training iteration."""

from __future__ import annotations

import random

from training.entities.league import LeaguePool


class SampleLeagueSeats:
    """Pure logic — no I/O.

    Returns a seat_config string like ``"nn,checkpoints/Ana.pt,bot_v5,nn"``
    suitable for passing directly to ``run_self_play``.

    Rules:
    - Seat 0 is always ``"nn"`` (the learner).
    - Seats 1–3 are drawn from ``pool.entries`` according to sampling weights.
    - ``min_nn_per_game`` is enforced: if fewer than that many seats are ``"nn"``
      after sampling, the deficit is filled by replacing seats (right-to-left)
      with ``"nn"``.
    """

    def execute(self, pool: LeaguePool) -> str:
        cfg = pool.config
        entries = pool.entries

        if not entries:
            # No opponents yet — fall back to full self-play
            return "nn,nn,nn,nn"

        weights = pool.sampling_weights()

        # Sample seats 1–3 independently
        sampled = random.choices(entries, weights=weights, k=3)
        seat_tokens = [e.opponent.seat_token() for e in sampled]

        # Enforce min_nn_per_game (seat 0 always nn, so we need min-1 more)
        nn_needed = max(0, cfg.min_nn_per_game - 1)
        nn_count = sum(1 for t in seat_tokens if t == "nn")
        if nn_count < nn_needed:
            deficit = nn_needed - nn_count
            # Replace non-nn slots from right-to-left
            for i in reversed(range(3)):
                if seat_tokens[i] != "nn" and deficit > 0:
                    seat_tokens[i] = "nn"
                    deficit -= 1

        return "nn," + ",".join(seat_tokens)
