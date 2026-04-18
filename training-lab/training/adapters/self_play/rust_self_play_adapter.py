"""Adapter: Rust self-play engine."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tarok_engine as te

from training.ports import SelfPlayPort


log = logging.getLogger(__name__)


def _mode_id_from_state(state: np.ndarray) -> int:
    """Map contract one-hot slice to v4 card-play mode id.

    Mode ids: 0=solo, 1=klop_berac, 2=partner_play, 3=color_valat.
    """
    contract_offset = int(te.CONTRACT_OFFSET)
    contract_size = int(te.CONTRACT_SIZE)
    contract_end = contract_offset + contract_size
    if state.shape[0] < contract_end:
        return 2

    contract_slice = state[contract_offset:contract_end]
    max_idx = int(np.argmax(contract_slice))
    max_val = float(np.max(contract_slice))
    if max_val <= 0.0:
        return 2
    if 4 <= max_idx <= 7:
        return 0
    if max_idx == 0 or max_idx == 8:
        return 1
    if max_idx == 9:
        return 3
    return 2


class RustSelfPlay(SelfPlayPort):
    def run(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        explore_rate: float,
        concurrency: int,
        include_replay_data: bool = False,
        include_oracle_states: bool = False,
    ) -> dict[str, Any]:
        raw = te.run_self_play(
            n_games=n_games,
            concurrency=concurrency,
            model_path=model_path,
            explore_rate=explore_rate,
            seat_config=seat_config,
            include_replay_data=include_replay_data,
            include_oracle_states=include_oracle_states,
        )

        # Compatibility shim for older engine builds that don't expose game_modes.
        if "game_modes" not in raw:
            states = np.asarray(raw["states"], dtype=np.float32)
            raw["game_modes"] = np.asarray([_mode_id_from_state(s) for s in states], dtype=np.int8)
            log.warning(
                "Rust run_self_play returned no game_modes; derived from state contract slice. "
                "Rebuild engine-rs to export game_modes natively.",
            )

        return raw

    def compute_run_stats(
        self,
        raw: dict[str, Any],
        seat_labels: list[str],
    ) -> tuple[int, tuple[float, float, float, float], dict[int, tuple[int, int, int]]]:
        players_np = np.asarray(raw["players"])
        nn_seats = [i for i, s in enumerate(seat_labels) if s == "nn"]
        n_learner = int(sum(np.sum(players_np == s) for s in nn_seats))

        scores_arr = raw.get("scores")
        if scores_arr is not None and len(scores_arr) > 0:
            scores_np = np.asarray(scores_arr)
            ms = np.mean(scores_np, axis=0)
            mean_scores: tuple[float, float, float, float] = (
                float(ms[0]), float(ms[1]), float(ms[2]), float(ms[3])
            )
            seat_outcomes: dict[int, tuple[int, int, int]] = {}
            learner_scores = scores_np[:, 0]
            for si in range(1, 4):
                if seat_labels[si] == "nn":
                    continue
                opp_scores = scores_np[:, si]
                wins = int(np.sum(learner_scores > opp_scores))
                losses = int(np.sum(learner_scores < opp_scores))
                draws = int(np.sum(learner_scores == opp_scores))
                seat_outcomes[si] = (wins, losses, draws)
        else:
            mean_scores = (0.0, 0.0, 0.0, 0.0)
            seat_outcomes = {}

        return n_learner, mean_scores, seat_outcomes
