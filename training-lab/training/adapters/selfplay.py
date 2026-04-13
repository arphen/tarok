"""Adapter: Rust self-play engine."""

from __future__ import annotations

from typing import Any

import tarok_engine as te

from training.ports import SelfPlayPort


class RustSelfPlay(SelfPlayPort):
    def run(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        explore_rate: float,
        concurrency: int,
    ) -> dict[str, Any]:
        return te.run_self_play(
            n_games=n_games,
            concurrency=concurrency,
            model_path=model_path,
            explore_rate=explore_rate,
            seat_config=seat_config,
        )
