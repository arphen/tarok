"""Adapter: duplicate-tournament (paired-deal) league Elo calibration.

For each (candidate, opponent) pair, play ``N`` games with the candidate at
seat 0 and the opponent's seat-token at seats 1..3 (filling with two other
pool opponents for variety). Then replay **the same** ``N`` deck seeds with
the opponent at seat 0 instead, using the same 3 filler tokens. Score the
paired deals head-to-head:

* win_rate = (candidate_wins + 0.5 * draws) / N
* implied_elo = opponent_elo + ELO_PER_PLACEMENT * (2 * wr - 1)

The final candidate Elo is the median implied value across opponents — the
same robust aggregation used by the self-play adapter. Because the two runs
per opponent share ``deck_seeds``, the dealing luck is identical and cancels
on the pairing, giving a much lower-variance rating signal than independent
mixed-seat games.

Heuristic bots (no model path) are handled natively because
``te.run_self_play`` accepts bot tokens at any seat.
"""

from __future__ import annotations

import random
from itertools import cycle
from typing import Any, Callable

import numpy as np
import tarok_engine as te

from training.entities.league import LeaguePool, LeaguePoolEntry
from training.ports.league_calibration_port import LeagueCalibrationPort
from training.use_cases.league_calibration_utils import _ELO_PER_PLACEMENT


_LEARNER_TOKEN = "nn"


def _is_learner_token(tok: str) -> bool:
    # Treat anything that's not a registered bot token as an NN path.
    return tok == _LEARNER_TOKEN or tok.endswith(".pt")


def _run_paired(
    *,
    seat0_token: str,
    seat0_path: str | None,
    filler_tokens: tuple[str, str],
    deck_seeds: list[int],
    concurrency: int,
    lapajne_mc_worlds: int | None,
    lapajne_mc_sims: int | None,
) -> np.ndarray:
    """Return the ``n_games x 4`` score array for this side of the pairing.

    When ``seat0_token`` is a learner/NN token or an ``.pt`` path, seat 0 is
    rendered as ``"nn"`` and the model loaded at that seat is ``seat0_path``.
    For heuristic bots we render the bot token directly and pass ``None`` for
    the model path (the Rust engine uses ``model_path`` only for NN seats).
    """
    if _is_learner_token(seat0_token):
        seat0_rendered = _LEARNER_TOKEN
        model_path = seat0_path
    else:
        seat0_rendered = seat0_token
        model_path = None

    seat_cfg = f"{seat0_rendered},{filler_tokens[0]},{filler_tokens[1]},{filler_tokens[0]}"
    # Pattern-4 seat: reuse filler_tokens[0] for parity; the important thing
    # is that both sides of the pair see the same seat_cfg at seats 1..3.

    kwargs: dict[str, Any] = dict(
        n_games=len(deck_seeds),
        concurrency=min(concurrency, max(1, len(deck_seeds))),
        explore_rate=0.0,
        seat_config=seat_cfg,
        include_replay_data=False,
        include_oracle_states=False,
        deck_seeds=deck_seeds,
    )
    if model_path is not None:
        kwargs["model_path"] = model_path
    if lapajne_mc_worlds is not None:
        kwargs["lapajne_mc_worlds"] = lapajne_mc_worlds
    if lapajne_mc_sims is not None:
        kwargs["lapajne_mc_sims"] = lapajne_mc_sims

    raw = te.run_self_play(**kwargs)
    return np.asarray(raw["scores"], dtype=np.int64)


def _paired_win_rate(candidate_scores: np.ndarray, opponent_scores: np.ndarray) -> float | None:
    """Fraction of deck-paired games where seat-0 candidate > seat-0 opponent.

    Draws count as 0.5. Returns None when there are no usable games.
    """
    n = min(len(candidate_scores), len(opponent_scores))
    if n == 0:
        return None
    c0 = candidate_scores[:n, 0]
    o0 = opponent_scores[:n, 0]
    wins = int(np.sum(c0 > o0))
    draws = int(np.sum(c0 == o0))
    return (wins + 0.5 * draws) / float(n)


def _pick_fillers(entries: list[LeaguePoolEntry], target_idx: int) -> tuple[str, str]:
    filler_indices = [i for i in range(len(entries)) if i != target_idx]
    if not filler_indices:
        filler_indices = [target_idx]
    cyc = cycle(filler_indices)
    a = entries[next(cyc)].opponent.seat_token()
    b = entries[next(cyc)].opponent.seat_token()
    return a, b


class DuplicateTournamentLeagueCalibrationAdapter(LeagueCalibrationPort):
    """Paired-deal duplicate-tournament Elo calibration."""

    def __init__(self, rng_seed: int = 0) -> None:
        self._rng_seed = int(rng_seed)

    # ---- initial --------------------------------------------------------

    def calibrate_initial(
        self,
        *,
        pool: LeaguePool,
        model_path: str,
        concurrency: int,
        session_size: int,
        n_games_per_pair: int,
        anchor_name: str | None,
        anchor_elo: float,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
        on_mixed_result: Callable[[int, int, str, tuple[str, str, str], tuple[float, float, float, float]], None] | None = None,
    ) -> bool:
        # Strategy: pin anchor at ``anchor_elo``; for every other pool entry
        # compute a paired-deal H2H against the anchor. Then compute the
        # learner's implied Elo vs the *median* pool entry.
        entries = list(pool.entries)
        if not entries:
            pool.learner_elo = float(anchor_elo)
            return True

        if n_games_per_pair <= 0:
            return False

        # Resolve anchor.
        anchor_idx = 0
        if anchor_name is not None:
            for i, e in enumerate(entries):
                if e.opponent.name == anchor_name:
                    anchor_idx = i
                    break
        anchor = entries[anchor_idx]
        anchor_tok = anchor.opponent.seat_token()
        anchor_path = anchor.opponent.path

        rng = random.Random(self._rng_seed)
        total_pairs = len(entries) - 1
        progress = 0

        for i, target in enumerate(entries):
            if i == anchor_idx:
                target.elo = float(anchor_elo)
                continue

            target_tok = target.opponent.seat_token()
            target_path = target.opponent.path

            # Shared deck seeds across the two sides of the pair.
            deck_seeds = [rng.getrandbits(63) for _ in range(n_games_per_pair)]
            fillers = _pick_fillers(entries, target_idx=i)

            candidate_scores = _run_paired(
                seat0_token=target_tok,
                seat0_path=target_path,
                filler_tokens=fillers,
                deck_seeds=deck_seeds,
                concurrency=concurrency,
                lapajne_mc_worlds=lapajne_mc_worlds,
                lapajne_mc_sims=lapajne_mc_sims,
            )
            anchor_scores = _run_paired(
                seat0_token=anchor_tok,
                seat0_path=anchor_path,
                filler_tokens=fillers,
                deck_seeds=deck_seeds,
                concurrency=concurrency,
                lapajne_mc_worlds=lapajne_mc_worlds,
                lapajne_mc_sims=lapajne_mc_sims,
            )
            wr = _paired_win_rate(candidate_scores, anchor_scores)
            if wr is None:
                target.elo = float(anchor_elo)
                continue

            target.elo = float(anchor_elo) + _ELO_PER_PLACEMENT * (2.0 * wr - 1.0)
            progress += 1
            if on_mixed_result is not None:
                # Reuse the existing presenter hook shape: report wr as the
                # target's average "placement" against the anchor so the UI
                # shows something meaningful.
                pseudo_place = 2.0 - wr
                on_mixed_result(
                    progress, total_pairs, target.opponent.name, fillers + (anchor.opponent.name,),
                    (pseudo_place, 1.0 + wr, pseudo_place, pseudo_place),
                )

        # Learner Elo: paired H2H vs the anchor using the same approach.
        # The learner is always an NN checkpoint at ``model_path``.
        deck_seeds = [rng.getrandbits(63) for _ in range(n_games_per_pair)]
        learner_fillers = (
            entries[(anchor_idx + 1) % len(entries)].opponent.seat_token(),
            entries[(anchor_idx + 2) % len(entries)].opponent.seat_token(),
        )
        learner_scores = _run_paired(
            seat0_token=_LEARNER_TOKEN,
            seat0_path=model_path,
            filler_tokens=learner_fillers,
            deck_seeds=deck_seeds,
            concurrency=concurrency,
            lapajne_mc_worlds=lapajne_mc_worlds,
            lapajne_mc_sims=lapajne_mc_sims,
        )
        anchor_scores = _run_paired(
            seat0_token=anchor_tok,
            seat0_path=anchor_path,
            filler_tokens=learner_fillers,
            deck_seeds=deck_seeds,
            concurrency=concurrency,
            lapajne_mc_worlds=lapajne_mc_worlds,
            lapajne_mc_sims=lapajne_mc_sims,
        )
        wr = _paired_win_rate(learner_scores, anchor_scores)
        if wr is None:
            pool.learner_elo = float(anchor_elo) - 100.0
        else:
            pool.learner_elo = float(anchor_elo) + _ELO_PER_PLACEMENT * (2.0 * wr - 1.0)
        return True

    # ---- candidate ------------------------------------------------------

    def calibrate_candidate(
        self,
        *,
        pool: LeaguePool,
        model_path: str,
        concurrency: int,
        session_size: int,
        n_games_per_opponent: int,
        lapajne_mc_worlds: int | None = None,
        lapajne_mc_sims: int | None = None,
    ) -> float | None:
        opponents = list(pool.entries)
        if not opponents:
            return None
        if n_games_per_opponent <= 0:
            return None

        rng = random.Random(self._rng_seed)
        implied_elos: list[float] = []

        for target_idx, target in enumerate(opponents):
            target_tok = target.opponent.seat_token()
            target_path = target.opponent.path
            fillers = _pick_fillers(opponents, target_idx=target_idx)

            deck_seeds = [rng.getrandbits(63) for _ in range(n_games_per_opponent)]

            candidate_scores = _run_paired(
                seat0_token=_LEARNER_TOKEN,
                seat0_path=model_path,
                filler_tokens=fillers,
                deck_seeds=deck_seeds,
                concurrency=concurrency,
                lapajne_mc_worlds=lapajne_mc_worlds,
                lapajne_mc_sims=lapajne_mc_sims,
            )
            opponent_scores = _run_paired(
                seat0_token=target_tok,
                seat0_path=target_path,
                filler_tokens=fillers,
                deck_seeds=deck_seeds,
                concurrency=concurrency,
                lapajne_mc_worlds=lapajne_mc_worlds,
                lapajne_mc_sims=lapajne_mc_sims,
            )
            wr = _paired_win_rate(candidate_scores, opponent_scores)
            if wr is None:
                continue
            implied = target.elo + _ELO_PER_PLACEMENT * (2.0 * wr - 1.0)
            implied_elos.append(implied)

        if not implied_elos:
            return None

        implied_elos.sort()
        mid = len(implied_elos) // 2
        if len(implied_elos) % 2 == 0:
            return (implied_elos[mid - 1] + implied_elos[mid]) / 2.0
        return implied_elos[mid]
