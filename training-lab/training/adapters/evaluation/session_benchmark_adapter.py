"""Adapter: session-based benchmark via Rust self-play."""

from __future__ import annotations

import numpy as np
import tarok_engine as te

from training.ports import BenchmarkPort


class SessionBenchmark(BenchmarkPort):
    def measure_placement(
        self,
        model_path: str,
        n_games: int,
        seat_config: str,
        concurrency: int,
        session_size: int,
    ) -> float:
        raw = te.run_self_play(
            n_games=n_games,
            concurrency=concurrency,
            model_path=model_path,
            explore_rate=0.0,
            seat_config=seat_config,
        )
        scores = np.array(raw["scores"])
        n_total = scores.shape[0]
        n_sessions = max(1, n_total // session_size)

        placements: list[float] = []
        for s in range(n_sessions):
            start = s * session_size
            end = min(start + session_size, n_total)
            if start >= n_total:
                break
            cumulative = scores[start:end].sum(axis=0)
            placement = int(np.sum(cumulative > cumulative[0])) + 1
            placements.append(placement)

        return float(np.mean(placements)) if placements else 2.5
