"""EvaluateModel — run N games vs reference opponents and compute metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from training_lab.entities.network import TarokNet
from training_lab.ports.game_simulator import GameSimulatorPort

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    win_rate: float
    avg_score: float
    avg_placement: float
    games_played: int
    scores: list[float]


class EvaluateModel:
    """Run N games with a trained network and compute performance metrics.

    Placement is computed at the SESSION level (cumulative score across
    a session of games), not per-game.  In Tarok the losing team scores 0,
    so per-game placement is meaningless — it just reflects team assignment.
    """

    def __init__(self, simulator: GameSimulatorPort, session_size: int = 50):
        self.simulator = simulator
        self.session_size = session_size

    def run(
        self,
        network: TarokNet,
        n_games: int = 100,
    ) -> EvalResult:
        results = self.simulator.play_batch(
            network=network,
            n_games=n_games,
            explore_rate=0.0,  # greedy evaluation
        )

        total_score = 0.0
        all_scores = []

        for result in results:
            agent_score = result.scores[0]
            total_score += agent_score
            all_scores.append(agent_score)

        # Session-level placement: group games into sessions,
        # sum scores per player, then rank player 0.
        n = len(results)
        n_sessions = max(1, n // self.session_size)
        session_placements = []
        session_wins = 0

        for s in range(n_sessions):
            start = s * self.session_size
            end = min(start + self.session_size, n)
            if start >= n:
                break
            cumulative = [0.0] * 4
            for r in results[start:end]:
                for p in range(4):
                    cumulative[p] += r.scores[p]
            # Rank player 0
            placement = sum(1 for p in range(1, 4) if cumulative[p] > cumulative[0]) + 1
            session_placements.append(placement)
            if placement == 1:
                session_wins += 1

        ns = max(len(session_placements), 1)
        return EvalResult(
            win_rate=session_wins / ns,
            avg_score=total_score / max(n, 1),
            avg_placement=sum(session_placements) / ns if session_placements else 2.5,
            games_played=n,
            scores=all_scores,
        )
