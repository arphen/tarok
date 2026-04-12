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
    """Run N games with a trained network and compute performance metrics."""

    def __init__(self, simulator: GameSimulatorPort):
        self.simulator = simulator

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
        wins = 0
        placements = []
        all_scores = []

        for result in results:
            scores = result.scores
            # Player 0 is the agent being evaluated
            agent_score = scores[0]
            total_score += agent_score
            all_scores.append(agent_score)

            # Placement: 1st = highest score
            sorted_scores = sorted(scores, reverse=True)
            placement = sorted_scores.index(agent_score) + 1
            placements.append(placement)

            if placement == 1:
                wins += 1

        n = len(results)
        return EvalResult(
            win_rate=wins / max(n, 1),
            avg_score=total_score / max(n, 1),
            avg_placement=sum(placements) / max(n, 1),
            games_played=n,
            scores=all_scores,
        )
