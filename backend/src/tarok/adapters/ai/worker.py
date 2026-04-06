"""Parallel self-play worker — runs games in a subprocess and returns experiences.

Workers are long-lived: the pool initializer creates agents once per process,
and each ``play_games_worker`` call just loads fresh weights and plays games.
This avoids the overhead of re-importing modules and rebuilding neural networks
on every session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch

from tarok.adapters.ai.agent import RLAgent, Experience
from tarok.adapters.ai.encoding import DecisionType
from tarok.adapters.ai.stockskis_player import StockSkisPlayer
from tarok.use_cases.game_loop import GameLoop


@dataclass
class GameStats:
    """Per-game statistics returned by a worker."""
    reward_p0: float
    won_p0: bool
    bid_p0: bool
    is_klop: bool
    is_solo: bool
    contract_name: str  # e.g. "klop", "three", "two", "one", "solo_three", etc.
    score_p0: int       # raw score (not /100)
    declarer_p0: bool   # True if Player 0 was the declarer (won the bidding)


@dataclass
class WorkerResult:
    """Everything a worker returns after playing its batch of games."""
    experiences: list[dict]  # serialized Experience dicts (CPU tensors)
    game_stats: list[GameStats]


# ---------------------------------------------------------------------------
# Per-process global state — initialised once by ``init_worker``
# ---------------------------------------------------------------------------
_worker_agents: list[RLAgent] | None = None
_worker_stockskis: list[StockSkisPlayer] | None = None
_worker_stockskis_ratio: float = 0.0


def init_worker(
    hidden_size: int,
    explore_rate: float,
    stockskis_ratio: float = 0.0,
    stockskis_strength: float = 1.0,
) -> None:
    """Pool initializer — called once in each worker subprocess."""
    global _worker_agents, _worker_stockskis, _worker_stockskis_ratio
    _worker_agents = [
        RLAgent(name=f"W-{i}", hidden_size=hidden_size, device="cpu",
                explore_rate=explore_rate)
        for i in range(4)
    ]
    shared_net = _worker_agents[0].network
    for agent in _worker_agents[1:]:
        agent.network = shared_net
    for agent in _worker_agents:
        agent.set_training(True)

    _worker_stockskis_ratio = stockskis_ratio
    if stockskis_ratio > 0:
        _worker_stockskis = [
            StockSkisPlayer(name=f"WSkis-{i}", strength=stockskis_strength)
            for i in range(3)
        ]


def _serialize_experience(exp: Experience) -> dict:
    """Convert Experience to a plain dict with CPU tensors for pickling."""
    return {
        "state": exp.state.cpu(),
        "action": exp.action,
        "log_prob": exp.log_prob.cpu(),
        "value": exp.value.cpu(),
        "decision_type": exp.decision_type.value,
        "reward": exp.reward,
        "done": exp.done,
        "oracle_state": exp.oracle_state.cpu() if exp.oracle_state is not None else None,
        "game_id": exp.game_id,
        "step_in_game": exp.step_in_game,
    }


def deserialize_experience(d: dict) -> Experience:
    """Reconstruct Experience from a serialized dict."""
    return Experience(
        state=d["state"],
        action=d["action"],
        log_prob=d["log_prob"],
        value=d["value"],
        decision_type=DecisionType(d["decision_type"]),
        reward=d["reward"],
        done=d["done"],
        oracle_state=d.get("oracle_state"),
        game_id=d.get("game_id", 0),
        step_in_game=d.get("step_in_game", 0),
    )


def play_games_worker(args: tuple) -> WorkerResult:
    """Top-level function for multiprocessing — must be picklable.

    Args is a tuple of (state_dict, num_games, dealer_offset).
    Agents are already initialised via ``init_worker``.
    """
    import random as _rng_mod

    state_dict, num_games, dealer_offset = args
    assert _worker_agents is not None, "init_worker was not called"

    agents = _worker_agents
    # Load the latest shared weights
    agents[0].network.load_state_dict(state_dict)

    all_experiences: list[dict] = []
    all_stats: list[GameStats] = []

    async def _run():
        rng = _rng_mod.Random()
        for g in range(num_games):
            # Decide whether to use StockŠkis opponents for this game
            use_stockskis = (
                _worker_stockskis is not None
                and _worker_stockskis_ratio > 0
                and rng.random() < _worker_stockskis_ratio
            )

            if use_stockskis:
                game_agents: list = [agents[0]] + _worker_stockskis  # type: ignore
            else:
                game_agents = list(agents)

            for agent in game_agents:
                agent.clear_experiences()

            game = GameLoop(game_agents)
            state, scores = await game.run(dealer=(dealer_offset + g) % 4)

            is_klop = state.contract is not None and state.contract.is_klop
            is_solo = state.contract is not None and state.contract.is_solo
            agent0_bids = [b for b in state.bids if b.player == 0 and b.contract is not None]
            contract_name = state.contract.name.lower() if state.contract else "klop"
            raw_score = scores.get(0, 0)
            declarer_p0 = state.declarer == 0

            if use_stockskis:
                # Only collect experiences from agent 0 (the learner)
                reward = scores.get(0, 0) / 100.0
                agents[0].finalize_game(reward)
                for exp in agents[0].experiences:
                    all_experiences.append(_serialize_experience(exp))
            else:
                for i, agent in enumerate(agents):
                    reward = scores.get(i, 0) / 100.0
                    agent.finalize_game(reward)
                    for exp in agent.experiences:
                        all_experiences.append(_serialize_experience(exp))

            all_stats.append(GameStats(
                reward_p0=raw_score / 100.0,
                won_p0=raw_score > 0,
                bid_p0=bool(agent0_bids),
                is_klop=is_klop,
                is_solo=is_solo,
                contract_name=contract_name,
                score_p0=raw_score,
                declarer_p0=declarer_p0,
            ))

    asyncio.run(_run())
    return WorkerResult(experiences=all_experiences, game_stats=all_stats)
