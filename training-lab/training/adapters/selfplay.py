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
