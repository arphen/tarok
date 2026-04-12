"""Training Lab — interactive neural network training and evaluation.

Provides endpoints for:
1. Creating a fresh neural network with a generated persona
2. Evaluating it against v1/v2/v3/v3.2 heuristic bots (real games)
3. Generating expert data from v2/v3 bots (imitation learning)
4. Self-play PPO training with all existing improvements
5. Saving snapshots to Hall of Fame for model selection / FSP
6. Polling progress and win-rate history
"""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from tarok.adapters.ai.network import TarokNet
from tarok.adapters.ai.hof_manager import HoFManager


def _detect_device() -> str:
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_self_play_device() -> str:
    """Pick the best available device for self-play training."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
from tarok.adapters.ai.agent import RLAgent, Experience
from tarok.adapters.ai.encoding import (
    DecisionType,
    BID_ACTION_SIZE,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
    CARD_ACTION_SIZE,
    ANNOUNCE_ACTION_SIZE,
)
from tarok.adapters.ai.imitation import imitation_pretrain
from tarok.adapters.ai.lookahead_agent import LookaheadAgent
from tarok.adapters.ai.network_bank import NetworkBank
from tarok.adapters.ai.opponent_pool import (
    OpponentPool,
    PureSelfPlayOpponent,
    FSPOpponent,
    StockSkisOpponent,
    HoFOpponent,
    OpponentGameResult,
)
from tarok.adapters.ai.stockskis_v5 import StockSkisPlayerV5
from tarok.adapters.api.spectator_observer import SpectatorObserver
from tarok.use_cases.game_loop import GameLoop

try:
    from tarok.adapters.ai.batch_game_runner import BatchGameRunner, GameResult, GameExperience
    from tarok.adapters.ai.compute import ComputeBackend, create_backend
    from tarok.adapters.ai.rust_game_loop import RustGameLoop
    import tarok_engine as _te_check
    _HAS_RUST = _te_check is not None
except Exception:
    _HAS_RUST = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PPOTrainer and supporting types (migrated from trainer.py)
# ---------------------------------------------------------------------------

_ACTION_SIZES = {
    DecisionType.BID: BID_ACTION_SIZE,
    DecisionType.KING_CALL: KING_ACTION_SIZE,
    DecisionType.TALON_PICK: TALON_ACTION_SIZE,
    DecisionType.CARD_PLAY: CARD_ACTION_SIZE,
    DecisionType.ANNOUNCE: ANNOUNCE_ACTION_SIZE,
}


@dataclass
class ContractStats:
    """Per-contract stats, split by role (declarer vs defender)."""
    # As declarer (P0 won the bidding for this contract)
    decl_played: int = 0
    decl_won: int = 0
    decl_total_score: int = 0
    # As defender (another player declared this contract)
    def_played: int = 0
    def_won: int = 0
    def_total_score: int = 0

    @property
    def played(self) -> int:
        return self.decl_played + self.def_played

    @property
    def decl_win_rate(self) -> float:
        return self.decl_won / max(self.decl_played, 1)

    @property
    def def_win_rate(self) -> float:
        return self.def_won / max(self.def_played, 1)

    @property
    def decl_avg_score(self) -> float:
        return self.decl_total_score / max(self.decl_played, 1)

    @property
    def def_avg_score(self) -> float:
        return self.def_total_score / max(self.def_played, 1)

    def to_dict(self) -> dict:
        return {
            "played": self.played,
            "decl_played": self.decl_played,
            "decl_won": self.decl_won,
            "decl_win_rate": round(self.decl_win_rate, 4),
            "decl_avg_score": round(self.decl_avg_score, 1),
            "def_played": self.def_played,
            "def_won": self.def_won,
            "def_win_rate": round(self.def_win_rate, 4),
            "def_avg_score": round(self.def_avg_score, 1),
        }


@dataclass
class TrainerOpponentStats:
    """Aggregate stats against a specific opponent pool (detailed role split)."""
    games: int = 0
    wins: int = 0
    total_score: int = 0
    total_place: float = 0.0
    bids: int = 0
    decl_games: int = 0
    decl_wins: int = 0
    decl_total_score: int = 0
    def_games: int = 0
    def_wins: int = 0
    def_total_score: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.games, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.games, 1)

    @property
    def avg_place(self) -> float:
        return self.total_place / max(self.games, 1)

    @property
    def bid_rate(self) -> float:
        return self.bids / max(self.games, 1)

    @property
    def decl_win_rate(self) -> float:
        return self.decl_wins / max(self.decl_games, 1)

    @property
    def decl_avg_score(self) -> float:
        return self.decl_total_score / max(self.decl_games, 1)

    @property
    def def_win_rate(self) -> float:
        return self.def_wins / max(self.def_games, 1)

    @property
    def def_avg_score(self) -> float:
        return self.def_total_score / max(self.def_games, 1)

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "avg_score": round(self.avg_score, 2),
            "avg_place": round(self.avg_place, 2),
            "bid_rate": round(self.bid_rate, 4),
            "decl_games": self.decl_games,
            "decl_win_rate": round(self.decl_win_rate, 4),
            "decl_avg_score": round(self.decl_avg_score, 2),
            "def_games": self.def_games,
            "def_win_rate": round(self.def_win_rate, 4),
            "def_avg_score": round(self.def_avg_score, 2),
        }


# Contract names we individually track
_TRACKED_CONTRACTS = ["klop", "three", "two", "one", "solo_three", "solo_two", "solo_one", "solo", "berac"]


@dataclass
class TrainingMetrics:
    run_id: str = ""
    episode: int = 0
    total_episodes: int = 0
    session: int = 0
    total_sessions: int = 0
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    avg_placement: float = 0.0
    entropy: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    games_per_second: float = 0.0
    bid_rate: float = 0.0
    klop_rate: float = 0.0
    solo_rate: float = 0.0
    contract_stats: dict[str, ContractStats] = field(
        default_factory=lambda: {c: ContractStats() for c in _TRACKED_CONTRACTS}
    )
    reward_history: list[float] = field(default_factory=list)
    avg_placement_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    bid_rate_history: list[float] = field(default_factory=list)
    klop_rate_history: list[float] = field(default_factory=list)
    solo_rate_history: list[float] = field(default_factory=list)
    contract_win_rate_history: dict[str, list[float]] = field(
        default_factory=lambda: {c: [] for c in _TRACKED_CONTRACTS}
    )
    session_avg_score_history: list[float] = field(default_factory=list)
    # Table scoring: sum of points per session when P0 is declaring team (family rules)
    table_score_history: list[float] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    # Avg finishing place of StockŠkis opponents per session (1=best, 4=worst)
    stockskis_place_history: list[float] = field(default_factory=list)
    # Lookahead opponent metrics per session
    lookahead_score_history: list[float] = field(default_factory=list)
    lookahead_bid_rate_history: list[float] = field(default_factory=list)
    # Tarok count vs bid: for each tarok count (0..12), track which contract was played
    tarok_count_bids: dict[int, dict[str, int]] = field(
        default_factory=lambda: {i: {} for i in range(13)}
    )
    # Evaluation signal and detailed breakdown against StockSkis v5 opponents
    vs_v5: TrainerOpponentStats = field(default_factory=TrainerOpponentStats)
    vs_v5_contract_stats: dict[str, TrainerOpponentStats] = field(
        default_factory=lambda: {c: TrainerOpponentStats() for c in _TRACKED_CONTRACTS}
    )
    vs_v5_avg_placement_history: list[float] = field(default_factory=list)
    vs_v5_avg_score_history: list[float] = field(default_factory=list)
    vs_v5_avg_place_history: list[float] = field(default_factory=list)
    vs_v5_bid_rate_history: list[float] = field(default_factory=list)
    vs_v5_eval_signal_history: list[float] = field(default_factory=list)
    # Per-opponent avg placement history (one entry per session)
    placement_selfplay_history: list[float] = field(default_factory=list)
    placement_hof_history: list[float] = field(default_factory=list)
    placement_v5_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "episode": self.episode,
            "total_episodes": self.total_episodes,
            "session": self.session,
            "total_sessions": self.total_sessions,
            "avg_reward": round(self.avg_reward, 2),
            "avg_loss": round(self.avg_loss, 4),
            "avg_placement": round(self.avg_placement, 4),
            "entropy": round(self.entropy, 4),
            "value_loss": round(self.value_loss, 4),
            "policy_loss": round(self.policy_loss, 4),
            "games_per_second": round(self.games_per_second, 2),
            "bid_rate": round(self.bid_rate, 4),
            "klop_rate": round(self.klop_rate, 4),
            "solo_rate": round(self.solo_rate, 4),
            "contract_stats": {k: v.to_dict() for k, v in self.contract_stats.items()},
            "history_offset": max(0, len(self.reward_history) - 500),
            "reward_history": self.reward_history[-500:],
            "avg_placement_history": self.avg_placement_history[-500:],
            "loss_history": self.loss_history[-500:],
            "bid_rate_history": self.bid_rate_history[-500:],
            "klop_rate_history": self.klop_rate_history[-500:],
            "solo_rate_history": self.solo_rate_history[-500:],
            "contract_win_rate_history": {
                k: v[-500:] for k, v in self.contract_win_rate_history.items()
            },
            "session_avg_score_history": self.session_avg_score_history[-500:],
            "table_score_history": self.table_score_history[-500:],
            "stockskis_place_history": self.stockskis_place_history[-500:],
            "lookahead_score_history": self.lookahead_score_history[-500:],
            "lookahead_bid_rate_history": self.lookahead_bid_rate_history[-500:],
            "snapshots": self.snapshots,
            "tarok_count_bids": {
                str(k): v for k, v in self.tarok_count_bids.items()
            },
            "vs_v5": self.vs_v5.to_dict(),
            "vs_v5_contract_stats": {
                k: v.to_dict() for k, v in self.vs_v5_contract_stats.items()
            },
            "vs_v5_win_rate_history": self.vs_v5_avg_placement_history[-500:],
            "vs_v5_avg_score_history": self.vs_v5_avg_score_history[-500:],
            "vs_v5_avg_place_history": self.vs_v5_avg_place_history[-500:],
            "vs_v5_bid_rate_history": self.vs_v5_bid_rate_history[-500:],
            "vs_v5_eval_signal_history": self.vs_v5_eval_signal_history[-500:],
            "placement_selfplay_history": self.placement_selfplay_history[-500:],
            "placement_hof_history": self.placement_hof_history[-500:],
            "placement_v5_history": self.placement_v5_history[-500:],
        }


class PPOTrainer:
    """Session-based self-play PPO trainer for Tarok agents.

    Each training session plays *games_per_session* games, collects all
    experiences (bids + king calls + talon picks + card plays), then
    performs a PPO update.  This teaches the agent the expected value of
    each bid given a hand, and the long-run payoff of conservative vs
    aggressive play.
    """

    def __init__(
        self,
        agents: list[RLAgent],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        games_per_session: int = 20,
        device: str = "cpu",
        save_dir: str = "checkpoints",
        # --- Fictitious Self-Play ---
        fsp_ratio: float = 0.3,
        bank_size: int = 20,
        bank_save_interval: int = 5,
        # --- StockŠkis opponents ---
        stockskis_ratio: float = 0.0,
        stockskis_strength: float = 1.0,
        stockskis_version: int = 5,
        # --- Lookahead opponents ---
        lookahead_ratio: float = 0.0,
        lookahead_sims: int = 20,
        lookahead_perfect_info: bool = True,
        # --- Hall of Fame opponents ---
        hof_ratio: float = 0.0,
        hof_dir: str = "checkpoints/hall_of_fame",
        # --- Rust engine ---
        use_rust_engine: bool = False,
        warmup_games: int = 0,
        # --- Batched self-play ---
        batch_concurrency: int = 32,  # concurrent games for batched NN inference
        # --- Scheduling ---
        lr_schedule: str = "cosine",       # "cosine" | "none"
        lr_min_ratio: float = 0.1,         # min LR = lr * lr_min_ratio
        entropy_schedule: str = "linear",  # "linear" | "none"
        entropy_coef_end: float = 0.002,   # final entropy coef
        value_clip: float = 0.0,           # >0 enables value function clipping
        # --- v2: Oracle guiding ---
        oracle_guiding_coef: float = 0.1,  # weight of oracle distillation loss
        # --- v2: pMCPA ---
        pmcpa_rollouts: int = 0,           # 0 disables; >0 = rollouts per hand
    ):
        self.agents = agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.games_per_session = games_per_session
        self.device = torch.device(device)
        self.compute = create_backend(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # All agents share the same network for self-play
        self.shared_network = agents[0].network
        self.shared_network = self.compute.prepare_network(self.shared_network)
        self.optimizer = optim.Adam(self.shared_network.parameters(), lr=lr)

        # Sync all agents to use the same network
        for agent in agents:
            agent.network = self.shared_network

        # --- Scheduling ---
        self._lr_init = lr
        self._lr_schedule = lr_schedule
        self._lr_min_ratio = lr_min_ratio
        self._entropy_coef_init = entropy_coef
        self._entropy_schedule = entropy_schedule
        self._entropy_coef_end = entropy_coef_end
        self._value_clip = value_clip
        self._oracle_guiding_coef = oracle_guiding_coef
        self._pmcpa_rollouts = pmcpa_rollouts

        # --- Fictitious Self-Play ---
        self.fsp_ratio = fsp_ratio
        self.bank_save_interval = bank_save_interval
        self.network_bank = NetworkBank(max_size=bank_size)
        # Separate opponent network for FSP games (same architecture, different weights)
        self.opponent_network: TarokNet | None = None
        if fsp_ratio > 0:
            # Infer hidden size from the shared network's first linear layer
            hidden_size = self.shared_network.shared[0].out_features
            self.opponent_network = TarokNet(
                hidden_size=hidden_size,
                oracle_critic=self.shared_network.oracle_critic_enabled,
            )
            self.opponent_network = self.compute.prepare_network(self.opponent_network)

        # --- StockŠkis opponents ---
        self.stockskis_ratio = stockskis_ratio
        self.stockskis_version = 5
        self._stockskis_opponents: list | None = None
        if stockskis_ratio > 0:
            self._stockskis_opponents = [
                StockSkisPlayerV5(name=f"StockŠkis-v5-{i}", strength=stockskis_strength)
                for i in range(3)
            ]

        # --- Lookahead opponents ---
        self.lookahead_ratio = lookahead_ratio
        self._lookahead_opponents: list[LookaheadAgent] | None = None
        if lookahead_ratio > 0:
            self._lookahead_opponents = [
                LookaheadAgent(
                    n_simulations=lookahead_sims,
                    name=f"Lookahead-{i}",
                    perfect_information=lookahead_perfect_info,
                )
                for i in range(3)
            ]

        # --- Hall of Fame opponents ---
        self.hof_ratio = hof_ratio
        self._hof_opponent: HoFOpponent | None = None
        if hof_ratio > 0:
            self._hof_opponent = HoFOpponent(hof_dir=hof_dir, weight=hof_ratio)

        self.metrics = TrainingMetrics()
        self._running = False
        self._metrics_callback: list = []
        self._rng = random.Random()

        # --- Opponent Pool (unified sampling) ---
        self.opponent_pool = OpponentPool(rng=self._rng)
        # Weights here represent the share of games going to each opponent type;
        # they do NOT need to sum to 1 — OpponentPool normalizes internally.
        remaining = 1.0 - stockskis_ratio - fsp_ratio - lookahead_ratio - hof_ratio
        if remaining > 0:
            self.opponent_pool.add(PureSelfPlayOpponent(weight=remaining))
        if self.fsp_ratio > 0:
            self.opponent_pool.add(FSPOpponent(self.network_bank, weight=self.fsp_ratio))
        if self.stockskis_ratio > 0:
            self.opponent_pool.add(StockSkisOpponent(
                version=self.stockskis_version,
                strength=stockskis_strength,
                weight=self.stockskis_ratio,
            ))
        if self.hof_ratio > 0 and self._hof_opponent is not None:
            self.opponent_pool.add(self._hof_opponent)

        # --- Rust engine ---
        self.use_rust_engine = use_rust_engine and _HAS_RUST
        self.warmup_games = warmup_games
        self.batch_concurrency = batch_concurrency

    def add_metrics_callback(self, callback) -> None:
        self._metrics_callback.append(callback)

    def _update_schedule(self, session_idx: int, total_sessions: int) -> None:
        """Update LR and entropy coefficient based on training progress."""
        progress = session_idx / max(total_sessions - 1, 1)  # 0.0 → 1.0

        # Learning rate schedule
        if self._lr_schedule == "cosine":
            lr_min = self._lr_init * self._lr_min_ratio
            lr = lr_min + 0.5 * (self._lr_init - lr_min) * (1 + math.cos(math.pi * progress))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        # Entropy coefficient schedule
        if self._entropy_schedule == "linear":
            self.entropy_coef = (
                self._entropy_coef_init
                + (self._entropy_coef_end - self._entropy_coef_init) * progress
            )

    def _record_opponent_result(self, mode: str, result: OpponentGameResult) -> None:
        """Record a game result to the correct opponent in the pool."""
        name_map = {
            "batch": "self-play",
            "stockskis": f"stockskis-v{self.stockskis_version}",
            "hof": "hof",
            "fsp": "fsp",
            "lookahead": "lookahead",
        }
        target_name = name_map.get(mode, "self-play")
        self.opponent_pool.record_session_result(target_name, result)

    def _enter_stockskis_mode(self) -> list:
        """Replace opponents (agents 1-3) with StockŠkis heuristic bots."""
        assert self._stockskis_opponents is not None
        original = list(self.agents)
        for i in range(1, 4):
            self.agents[i] = self._stockskis_opponents[i - 1]  # type: ignore[assignment]
        return original

    def _exit_stockskis_mode(self, original_agents: list) -> None:
        """Restore the original agents after a StockŠkis game."""
        for i in range(4):
            self.agents[i] = original_agents[i]

    def _enter_lookahead_mode(self) -> list:
        """Replace opponents (agents 1-3) with Lookahead search bots."""
        assert self._lookahead_opponents is not None
        original = list(self.agents)
        for i in range(1, 4):
            self.agents[i] = self._lookahead_opponents[i - 1]  # type: ignore[assignment]
        return original

    def _exit_lookahead_mode(self, original_agents: list) -> None:
        """Restore the original agents after a Lookahead game."""
        for i in range(4):
            self.agents[i] = original_agents[i]

    def _enter_hof_mode(self) -> list:
        """Replace opponents (agents 1-3) with a random HoF model opponent."""
        assert self._hof_opponent is not None
        original = list(self.agents)
        players = self._hof_opponent.make_players(self.shared_network)
        if players:
            for i in range(1, 4):
                self.agents[i] = players[i - 1]  # type: ignore[assignment]
        return original

    def _exit_hof_mode(self, original_agents: list) -> None:
        """Restore the original agents after a HoF game."""
        for i in range(4):
            self.agents[i] = original_agents[i]

    @staticmethod
    def _eval_signal(stats: TrainerOpponentStats) -> float:
        """Scalar evaluation signal from win rate + score + placement."""
        wr = stats.win_rate
        score_term = max(-1.0, min(1.0, stats.avg_score / 120.0))
        place_term = max(0.0, min(1.0, (4.0 - stats.avg_place) / 3.0))
        return 0.55 * wr + 0.30 * ((score_term + 1.0) / 2.0) + 0.15 * place_term

    @staticmethod
    def _update_opponent_stats(
        stats: TrainerOpponentStats,
        contract_stats: dict[str, TrainerOpponentStats],
        contract_name: str,
        raw_score: int,
        won: bool,
        bid: bool,
        place: float,
        declarer_p0: bool,
    ) -> None:
        stats.games += 1
        stats.wins += 1 if won else 0
        stats.total_score += raw_score
        stats.total_place += place
        stats.bids += 1 if bid else 0
        if declarer_p0:
            stats.decl_games += 1
            stats.decl_wins += 1 if won else 0
            stats.decl_total_score += raw_score
        else:
            stats.def_games += 1
            stats.def_wins += 1 if won else 0
            stats.def_total_score += raw_score

        if contract_name in contract_stats:
            cs = contract_stats[contract_name]
            cs.games += 1
            cs.wins += 1 if won else 0
            cs.total_score += raw_score
            cs.total_place += place
            cs.bids += 1 if bid else 0
            if declarer_p0:
                cs.decl_games += 1
                cs.decl_wins += 1 if won else 0
                cs.decl_total_score += raw_score
            else:
                cs.def_games += 1
                cs.def_wins += 1 if won else 0
                cs.def_total_score += raw_score

    def _enter_fsp_mode(self) -> None:
        """Switch opponents (agents 1-3) to historical weights for FSP."""
        snap, snapshot_id = self.network_bank.sample()
        if snap is None or self.opponent_network is None:
            return
        self.opponent_network.load_state_dict(snap)
        # Track which snapshot was used for per-instance stats
        for opp in self.opponent_pool.opponents:
            if opp.name == "fsp" and hasattr(opp, '_last_snapshot_id'):
                opp._last_snapshot_id = snapshot_id
                break
        for agent in self.agents[1:]:
            agent.network = self.opponent_network
            agent.set_training(False)  # Don't record experiences for opponents

    def _exit_fsp_mode(self) -> None:
        """Switch opponents back to the shared (learning) network."""
        for agent in self.agents[1:]:
            agent.network = self.shared_network
            agent.set_training(True)

    async def train(self, num_sessions: int) -> TrainingMetrics:
        """Run session-based self-play training.

        Each session = *games_per_session* games → one PPO update.
        Metrics update after every game for live dashboard feedback.
        """
        # --- Optional warmup phase ---
        if self.warmup_games > 0 and self.use_rust_engine:
            await self._run_warmup()

        self._running = True
        total_games = num_sessions * self.games_per_session
        self.metrics.total_episodes = total_games
        self.metrics.total_sessions = num_sessions

        # Generate a unique run ID (short hash from timestamp + config)
        run_seed = f"{time.time()}-{num_sessions}-{self.games_per_session}-{id(self)}"
        self.metrics.run_id = hashlib.sha256(run_seed.encode()).hexdigest()[:8]
        print(f"[Trainer] Run #{self.metrics.run_id} — {num_sessions} sessions × {self.games_per_session} games")

        recent_rewards: list[float] = []
        recent_bids: list[float] = []
        recent_klops: list[float] = []
        recent_solos: list[float] = []
        # Per-game records for rolling contract stats: (contract_name, declarer_p0, raw_score, won)
        recent_games: list[tuple[str, bool, int, bool]] = []
        start_time = time.time()
        game_count = 0
        # Running sums for O(1) rolling-window metrics
        _rsum = 0.0; _bsum = 0.0; _ksum = 0.0; _ssum = 0.0

        for agent in self.agents:
            agent.set_training(True)

        snapshot_interval = max(1, num_sessions // 10)

        for session_idx in range(num_sessions):
            if not self._running:
                break

            # Update LR and entropy schedules
            self._update_schedule(session_idx, num_sessions)

            self.metrics.session = session_idx + 1
            all_experiences: list[Experience] = []
            session_scores: list[int] = []
            session_table_scores: list[int] = []  # table scoring: only declaring team
            session_stockskis_places: list[float] = []
            session_lookahead_scores: list[int] = []
            session_lookahead_bids: list[float] = []
            session_v5 = TrainerOpponentStats()
            session_v5_contracts: dict[str, TrainerOpponentStats] = {
                c: TrainerOpponentStats() for c in _TRACKED_CONTRACTS
            }
            # Cumulative per-player scores for session-level placement.
            # In Tarok, the losing team scores 0, so per-game placement
            # is meaningless — it just reflects team assignment.
            # Placement is computed from cumulative session scores.
            session_cumulative_scores: list[int] = [0, 0, 0, 0]

            # --- Batched self-play (Rust engine only) ---
            # Split games into batched NN games and sequential StockŠkis games
            can_batch = (
                self.use_rust_engine
                and _HAS_RUST
                and self.batch_concurrency > 1
            )
            if can_batch:
                batched_exps, batched_stats, stockskis_exps, stockskis_stats = (
                    await self._play_session_batched(
                        session_idx, game_count, start_time,
                    )
                )
                for result in batched_stats:
                    if not self._running:
                        break
                    raw_score = result["raw_score"]
                    contract_name = result["contract_name"]
                    is_klop = result["is_klop"]
                    is_solo = result["is_solo"]
                    declarer_p0 = result["declarer_p0"]
                    agent0_bids = result["agent0_bids"]

                    if result.get("initial_tarok_counts"):
                        tarok_counts = result["initial_tarok_counts"]
                        tarok_count = tarok_counts[0] if isinstance(tarok_counts, (list, tuple)) else tarok_counts.get(0, 0)
                        bucket = self.metrics.tarok_count_bids.setdefault(tarok_count, {})
                        bucket[contract_name] = bucket.get(contract_name, 0) + 1

                    game_count += 1
                    session_scores.append(raw_score)
                    # Table scoring: only record when P0 is on the declaring team
                    _partner_p0 = result.get("partner_p0", False)
                    if is_klop or declarer_p0 or _partner_p0:
                        session_table_scores.append(raw_score)
                    r = raw_score / 100.0
                    if is_klop:
                        w = 1.0 if raw_score > 0 else 0.0
                    elif declarer_p0:
                        w = 1.0 if raw_score > 0 else 0.0
                    else:
                        w = 1.0 if result.get("declarer_lost", False) else 0.0
                    
                    # Accumulate all 4 players' scores for session-level placement
                    all_sc = result.get("all_scores", {})
                    for pid in range(4):
                        session_cumulative_scores[pid] += all_sc.get(pid, 0)

                    b = 1.0 if agent0_bids else 0.0
                    k = 1.0 if is_klop else 0.0
                    s = 1.0 if is_solo else 0.0

                    # Record to opponent pool for per-instance tracking
                    game_mode = result.get("game_mode", "batch")
                    self._record_opponent_result(game_mode, OpponentGameResult(
                        raw_score=raw_score,
                        won=(w > 0),
                        contract_name=contract_name,
                        place=0.0,  # placeholder; real placement computed at session end
                        declarer_p0=declarer_p0,
                        bid=bool(agent0_bids),
                    ))

                    recent_rewards.append(r)
                    recent_bids.append(b); recent_klops.append(k); recent_solos.append(s)
                    recent_games.append((contract_name, declarer_p0, raw_score, w > 0))
                    _rsum += r; _bsum += b; _ksum += k; _ssum += s

                    if contract_name in self.metrics.contract_stats:
                        cs = self.metrics.contract_stats[contract_name]
                        if declarer_p0:
                            cs.decl_played += 1
                            if raw_score > 0:
                                cs.decl_won += 1
                            cs.decl_total_score += raw_score
                        else:
                            cs.def_played += 1
                            if w > 0:
                                cs.def_won += 1
                            cs.def_total_score += raw_score

                    window = self.games_per_session * 10
                    if len(recent_rewards) > window:
                        _rsum -= recent_rewards[-window - 1]
                        _bsum -= recent_bids[-window - 1]
                        _ksum -= recent_klops[-window - 1]
                        _ssum -= recent_solos[-window - 1]
                        old_cn, old_decl, old_score, old_won = recent_games[-window - 1]
                        if old_cn in self.metrics.contract_stats:
                            old_cs = self.metrics.contract_stats[old_cn]
                            if old_decl:
                                old_cs.decl_played -= 1
                                if old_score > 0:
                                    old_cs.decl_won -= 1
                                old_cs.decl_total_score -= old_score
                            else:
                                old_cs.def_played -= 1
                                if old_won:
                                    old_cs.def_won -= 1
                                old_cs.def_total_score -= old_score

                    self.metrics.episode = game_count
                    n = min(len(recent_rewards), window)
                    self.metrics.avg_reward = _rsum / n
                    self.metrics.bid_rate = _bsum / n
                    self.metrics.klop_rate = _ksum / n
                    self.metrics.solo_rate = _ssum / n

                # Process sequential StockŠkis game stats the same way
                for result in stockskis_stats:
                    if not self._running:
                        break
                    raw_score = result["raw_score"]
                    contract_name = result["contract_name"]
                    is_klop = result["is_klop"]
                    is_solo = result["is_solo"]
                    declarer_p0 = result["declarer_p0"]
                    agent0_bids = result["agent0_bids"]

                    if result.get("initial_tarok_counts"):
                        tarok_counts = result["initial_tarok_counts"]
                        tarok_count = tarok_counts[0] if isinstance(tarok_counts, (list, tuple)) else tarok_counts.get(0, 0)
                        bucket = self.metrics.tarok_count_bids.setdefault(tarok_count, {})
                        bucket[contract_name] = bucket.get(contract_name, 0) + 1

                    # Track StockŠkis finishing places
                    if "stockskis_place" in result:
                        session_stockskis_places.append(result["stockskis_place"])

                    game_count += 1
                    session_scores.append(raw_score)
                    # Table scoring: only record when P0 is on the declaring team
                    _partner_p0 = result.get("partner_p0", False)
                    if is_klop or declarer_p0 or _partner_p0:
                        session_table_scores.append(raw_score)
                    r = raw_score / 100.0
                    if is_klop:
                        w = 1.0 if raw_score > 0 else 0.0
                    elif declarer_p0:
                        w = 1.0 if raw_score > 0 else 0.0
                    else:
                        w = 1.0 if result.get("declarer_lost", False) else 0.0
                    
                    # Accumulate all 4 players' scores for session-level placement
                    all_sc = result.get("all_scores", {})
                    for pid in range(4):
                        session_cumulative_scores[pid] += all_sc.get(pid, 0)

                    b = 1.0 if agent0_bids else 0.0
                    k = 1.0 if is_klop else 0.0
                    s = 1.0 if is_solo else 0.0

                    # Record to opponent pool for per-instance tracking
                    game_mode = result.get("game_mode", "stockskis")
                    self._record_opponent_result(game_mode, OpponentGameResult(
                        raw_score=raw_score,
                        won=(w > 0),
                        contract_name=contract_name,
                        place=0.0,  # placeholder; real placement computed at session end
                        declarer_p0=declarer_p0,
                        bid=bool(agent0_bids),
                    ))

                    if self.stockskis_version == 5:
                        self._update_opponent_stats(
                            session_v5,
                            session_v5_contracts,
                            contract_name=contract_name,
                            raw_score=raw_score,
                            won=(w > 0),
                            bid=bool(agent0_bids),
                            place=0.0,
                            declarer_p0=declarer_p0,
                        )

                    recent_rewards.append(r)
                    recent_bids.append(b); recent_klops.append(k); recent_solos.append(s)
                    recent_games.append((contract_name, declarer_p0, raw_score, w > 0))
                    _rsum += r; _bsum += b; _ksum += k; _ssum += s

                    if contract_name in self.metrics.contract_stats:
                        cs = self.metrics.contract_stats[contract_name]
                        if declarer_p0:
                            cs.decl_played += 1
                            if raw_score > 0:
                                cs.decl_won += 1
                            cs.decl_total_score += raw_score
                        else:
                            cs.def_played += 1
                            if w > 0:
                                cs.def_won += 1
                            cs.def_total_score += raw_score

                    window = self.games_per_session * 10
                    if len(recent_rewards) > window:
                        _rsum -= recent_rewards[-window - 1]
                        _bsum -= recent_bids[-window - 1]
                        _ksum -= recent_klops[-window - 1]
                        _ssum -= recent_solos[-window - 1]
                        old_cn, old_decl, old_score, old_won = recent_games[-window - 1]
                        if old_cn in self.metrics.contract_stats:
                            old_cs = self.metrics.contract_stats[old_cn]
                            if old_decl:
                                old_cs.decl_played -= 1
                                if old_score > 0:
                                    old_cs.decl_won -= 1
                                old_cs.decl_total_score -= old_score
                            else:
                                old_cs.def_played -= 1
                                if old_won:
                                    old_cs.def_won -= 1
                                old_cs.def_total_score -= old_score

                    self.metrics.episode = game_count
                    n = min(len(recent_rewards), window)
                    self.metrics.avg_reward = _rsum / n
                    self.metrics.bid_rate = _bsum / n
                    self.metrics.klop_rate = _ksum / n
                    self.metrics.solo_rate = _ssum / n

                all_experiences.extend(batched_exps)
                all_experiences.extend(stockskis_exps)

                # Update throughput
                elapsed = time.time() - start_time
                self.metrics.games_per_second = game_count / max(elapsed, 1e-6)
                for cb in self._metrics_callback:
                    await cb(self.metrics)
                await asyncio.sleep(0)

            else:
                # --- Sequential fallback (no Rust engine or concurrency=1) ---
                for g in range(self.games_per_session):
                    if not self._running:
                        break

                    # --- Opponent mode selection ---
                    # StockŠkis mode: use heuristic bots as opponents
                    use_stockskis = (
                        self.stockskis_ratio > 0
                        and self._stockskis_opponents is not None
                        and self._rng.random() < self.stockskis_ratio
                    )
                    # HoF mode: use Hall of Fame models as frozen opponents
                    use_hof = (
                        not use_stockskis
                        and self.hof_ratio > 0
                        and self._hof_opponent is not None
                        and self._hof_opponent.is_available()
                        and self._rng.random() < self.hof_ratio
                    )
                    # Lookahead mode: use Monte Carlo search bots as opponents (only if not StockŠkis)
                    use_lookahead = (
                        not use_stockskis
                        and not use_hof
                        and self.lookahead_ratio > 0
                        and self._lookahead_opponents is not None
                        and self._rng.random() < self.lookahead_ratio
                    )
                    # FSP mode: use historical network weights (only if not StockŠkis)
                    use_fsp = (
                        not use_stockskis
                        and not use_hof
                        and not use_lookahead
                        and self.fsp_ratio > 0
                        and self.network_bank.is_ready
                        and self._rng.random() < self.fsp_ratio
                    )
                    external_opponents = use_fsp or use_stockskis or use_lookahead or use_hof

                    original_agents = None
                    if use_stockskis:
                        original_agents = self._enter_stockskis_mode()
                    elif use_hof:
                        original_agents = self._enter_hof_mode()
                    elif use_lookahead:
                        original_agents = self._enter_lookahead_mode()
                    elif use_fsp:
                        self._enter_fsp_mode()

                    # Clear experiences from previous game
                    for agent in self.agents:
                        if hasattr(agent, 'clear_experiences'):
                            agent.clear_experiences()

                    # Play one game (Rust engine or Python engine)
                    if self.use_rust_engine and not use_stockskis:
                        game = RustGameLoop(self.agents)
                    else:
                        game = GameLoop(self.agents)
                    try:
                        state, scores = await game.run(dealer=(game_count + g) % 4)
                    except Exception:
                        log.exception("Training game crashed; skipping (session=%s game=%s)", session_idx + 1, g + 1)
                        continue
                    finally:
                        # Restore agents after external-opponent game (even on crash)
                        if use_stockskis and original_agents is not None:
                            self._exit_stockskis_mode(original_agents)
                        elif use_hof and original_agents is not None:
                            self._exit_hof_mode(original_agents)
                        elif use_lookahead and original_agents is not None:
                            self._exit_lookahead_mode(original_agents)
                        elif use_fsp:
                            self._exit_fsp_mode()

                    # Extract stats
                    is_klop = state.contract is not None and state.contract.is_klop
                    is_solo = state.contract is not None and state.contract.is_solo
                    agent0_bids = [b for b in state.bids if b.player == 0 and b.contract is not None]
                    contract_name = state.contract.name.lower() if state.contract else "klop"
                    raw_score = scores.get(0, 0)
                    declarer_p0 = state.declarer == 0

                    # Track StockŠkis finishing places (1=best, 4=worst)
                    if use_stockskis:
                        sorted_players = sorted(scores, key=lambda p: scores[p], reverse=True)
                        places = {p: rank + 1 for rank, p in enumerate(sorted_players)}
                        avg_sk_place = sum(places[p] for p in range(1, 4)) / 3.0
                        session_stockskis_places.append(avg_sk_place)

                    # Track tarok count vs contract for player 0
                    if state.initial_tarok_counts:
                        tarok_count = state.initial_tarok_counts[0]
                        bucket = self.metrics.tarok_count_bids.setdefault(tarok_count, {})
                        bucket[contract_name] = bucket.get(contract_name, 0) + 1

                    # Finalize rewards and collect experiences
                    for i, agent in enumerate(self.agents):
                        reward = scores.get(i, 0) / 100.0
                        agent.finalize_game(reward)
                        # With external opponents, only agent 0 records experiences
                        if not external_opponents or i == 0:
                            all_experiences.extend(agent.experiences)

                    game_count += 1
                    session_scores.append(raw_score)
                    # Table scoring: only record when P0 is on the declaring team
                    _partner_p0 = state.partner == 0 if state.partner is not None else False
                    if is_klop or declarer_p0 or _partner_p0:
                        session_table_scores.append(raw_score)
                    r = raw_score / 100.0
                    if is_klop:
                        w = 1.0 if raw_score > 0 else 0.0
                    elif declarer_p0:
                        w = 1.0 if raw_score > 0 else 0.0
                    else:
                        declarer_score = scores.get(state.declarer, 0) if state.declarer is not None else 0
                        w = 1.0 if declarer_score < 0 else 0.0
                        
                    # Accumulate all 4 players' scores for session-level placement
                    for pid in range(4):
                        session_cumulative_scores[pid] += scores.get(pid, 0)

                    b = 1.0 if agent0_bids else 0.0
                    k = 1.0 if is_klop else 0.0
                    s = 1.0 if is_solo else 0.0

                    # Record to opponent pool for per-instance tracking
                    seq_mode = (
                        "stockskis" if use_stockskis else
                        "hof" if use_hof else
                        "fsp" if use_fsp else
                        "lookahead" if use_lookahead else
                        "batch"
                    )
                    self._record_opponent_result(seq_mode, OpponentGameResult(
                        raw_score=raw_score,
                        won=(w > 0),
                        contract_name=contract_name,
                        place=0.0,  # placeholder; real placement computed at session end
                        declarer_p0=declarer_p0,
                        bid=bool(agent0_bids),
                    ))

                    # StockŠkis v5 per-contract stats (after w is computed)
                    if use_stockskis and self.stockskis_version == 5:
                        self._update_opponent_stats(
                            session_v5,
                            session_v5_contracts,
                            contract_name=contract_name,
                            raw_score=raw_score,
                            won=(w > 0),
                            bid=bool(agent0_bids),
                            place=0.0,
                            declarer_p0=declarer_p0,
                        )

                    # Track lookahead scores + bid rate (per-game, aggregated per-session)
                    if use_lookahead:
                        session_lookahead_scores.append(raw_score)
                        session_lookahead_bids.append(b)
                    recent_rewards.append(r)
                    recent_bids.append(b); recent_klops.append(k); recent_solos.append(s)
                    recent_games.append((contract_name, declarer_p0, raw_score, w > 0))
                    _rsum += r; _bsum += b; _ksum += k; _ssum += s

                    # Per-contract tracking (split by role) — add current game
                    if contract_name in self.metrics.contract_stats:
                        cs = self.metrics.contract_stats[contract_name]
                        if declarer_p0:
                            cs.decl_played += 1
                            if raw_score > 0:
                                cs.decl_won += 1
                            cs.decl_total_score += raw_score
                        else:
                            cs.def_played += 1
                            if w > 0:
                                cs.def_won += 1
                            cs.def_total_score += raw_score

                    # Evict oldest entry once we exceed the window
                    window = self.games_per_session * 10
                    if len(recent_rewards) > window:
                        _rsum -= recent_rewards[-window - 1]
                        _bsum -= recent_bids[-window - 1]
                        _ksum -= recent_klops[-window - 1]
                        _ssum -= recent_solos[-window - 1]
                        # Also evict from rolling contract stats
                        old_cn, old_decl, old_score, old_won = recent_games[-window - 1]
                        if old_cn in self.metrics.contract_stats:
                            old_cs = self.metrics.contract_stats[old_cn]
                            if old_decl:
                                old_cs.decl_played -= 1
                                if old_score > 0:
                                    old_cs.decl_won -= 1
                                old_cs.decl_total_score -= old_score
                            else:
                                old_cs.def_played -= 1
                                if old_won:
                                    old_cs.def_won -= 1
                                old_cs.def_total_score -= old_score

                    # Update live metrics (O(1) — no recomputation)
                    self.metrics.episode = game_count
                    n = min(len(recent_rewards), window)
                    self.metrics.avg_reward = _rsum / n
                    self.metrics.bid_rate = _bsum / n
                    self.metrics.klop_rate = _ksum / n
                    self.metrics.solo_rate = _ssum / n

                    # Throttle callbacks: every 10 games (print flush + async yield is expensive)
                    if game_count % 10 == 0 or g == self.games_per_session - 1:
                        elapsed = time.time() - start_time
                        self.metrics.games_per_second = game_count / max(elapsed, 1e-6)
                        for cb in self._metrics_callback:
                            await cb(self.metrics)
                        # Yield to the event loop so FastAPI can serve HTTP requests
                        await asyncio.sleep(0)

            # Per-session average score (internal zero-sum signal)
            if session_scores:
                self.metrics.session_avg_score_history.append(
                    round(sum(session_scores) / len(session_scores), 2)
                )

            # Per-session table score (family rules: only count declaring team + klop)
            table_total = sum(session_table_scores)
            self.metrics.table_score_history.append(round(table_total, 2))

            # --- PPO update on the full session ---
            if all_experiences:
                loss_info = self._ppo_update(all_experiences)
                self.metrics.policy_loss = loss_info["policy_loss"]
                self.metrics.value_loss = loss_info["value_loss"]
                self.metrics.entropy = loss_info["entropy"]
                self.metrics.avg_loss = loss_info["total_loss"]

            # Yield after PPO update so the event loop can serve HTTP requests
            await asyncio.sleep(0)

            # --- Push to network bank for Fictitious Self-Play ---
            if self.fsp_ratio > 0 and (session_idx + 1) % self.bank_save_interval == 0:
                self.network_bank.push(self.shared_network.state_dict())

            # StockŠkis avg finishing place for this session (NaN if no StockŠkis games)
            if session_stockskis_places:
                self.metrics.stockskis_place_history.append(
                    round(sum(session_stockskis_places) / len(session_stockskis_places), 2)
                )

            # Lookahead opponent metrics for this session
            if session_lookahead_scores:
                self.metrics.lookahead_score_history.append(
                    round(sum(session_lookahead_scores) / len(session_lookahead_scores), 2)
                )
                self.metrics.lookahead_bid_rate_history.append(
                    round(sum(session_lookahead_bids) / len(session_lookahead_bids), 4)
                )

            # Update v5 evaluation block (can be used as a learning/eval signal)
            self.metrics.vs_v5 = session_v5
            self.metrics.vs_v5_contract_stats = session_v5_contracts
            self.metrics.vs_v5_avg_placement_history.append(round(session_v5.win_rate, 4))
            self.metrics.vs_v5_avg_score_history.append(round(session_v5.avg_score, 2))
            self.metrics.vs_v5_avg_place_history.append(round(session_v5.avg_place, 2))
            self.metrics.vs_v5_bid_rate_history.append(round(session_v5.bid_rate, 4))
            self.metrics.vs_v5_eval_signal_history.append(
                round(self._eval_signal(session_v5), 4)
            )

            # --- Session-level placement from cumulative scores ---
            # In Tarok, per-game placement is meaningless (opponents always score 0).
            # Placement must be computed from cumulative scores across the whole session.
            sorted_session = sorted(range(4), key=lambda pid: session_cumulative_scores[pid], reverse=True)
            session_placement = float(sorted_session.index(0) + 1)  # 1=best, 4=worst
            self.metrics.avg_placement = session_placement

            # --- Append per-session history for charts ---
            self.metrics.reward_history.append(self.metrics.avg_reward)
            self.metrics.avg_placement_history.append(session_placement)
            self.metrics.loss_history.append(self.metrics.avg_loss)
            self.metrics.bid_rate_history.append(self.metrics.bid_rate)
            self.metrics.klop_rate_history.append(self.metrics.klop_rate)
            self.metrics.solo_rate_history.append(self.metrics.solo_rate)

            # Per-opponent placement history — now uses the session-level placement
            # (all game types within a session contribute to a single placement)
            self.metrics.placement_selfplay_history.append(session_placement)
            self.metrics.placement_hof_history.append(session_placement)
            self.metrics.placement_v5_history.append(session_placement)

            # Per-contract declarer win rate history
            for cname in _TRACKED_CONTRACTS:
                cs = self.metrics.contract_stats[cname]
                self.metrics.contract_win_rate_history[cname].append(
                    round(cs.decl_win_rate, 4)
                )

            # Periodic snapshot checkpoint
            if (session_idx + 1) % snapshot_interval == 0 or session_idx == num_sessions - 1:
                snap_info = self._save_checkpoint(game_count, is_snapshot=True)
                self.metrics.snapshots.append(snap_info)

        # Save final completed snapshot
        custom_name = f"tarok_agent_completed_S{num_sessions}_ep{game_count}.pt"
        snap_info = self._save_checkpoint(game_count, is_snapshot=True, custom_name=custom_name)
        if snap_info:
            self.metrics.snapshots.append(snap_info)
        return self.metrics

    async def _play_session_batched(
        self,
        session_idx: int,
        game_count: int,
        start_time: float,
    ) -> tuple[list[Experience], list[dict], list[Experience], list[dict]]:
        """Play a session using batched NN inference.

        Returns (batched_experiences, batched_stats, stockskis_experiences, stockskis_stats).
        Batched games use BatchGameRunner; StockŠkis games run sequentially.
        """
        # Decide per-game modes up front
        n_stockskis = 0
        n_hof = 0
        n_batched = 0
        game_modes: list[str] = []  # "batch" | "stockskis" | "hof"
        for _ in range(self.games_per_session):
            if (
                self.stockskis_ratio > 0
                and self._stockskis_opponents is not None
                and self._rng.random() < self.stockskis_ratio
            ):
                game_modes.append("stockskis")
                n_stockskis += 1
            elif (
                self.hof_ratio > 0
                and self._hof_opponent is not None
                and self._hof_opponent.is_available()
                and self._rng.random() < self.hof_ratio
            ):
                game_modes.append("hof")
                n_hof += 1
            else:
                game_modes.append("batch")
                n_batched += 1

        batched_experiences: list[Experience] = []
        batched_stats: list[dict] = []
        stockskis_experiences: list[Experience] = []
        stockskis_stats: list[dict] = []

        # --- Run batched NN games ---
        if n_batched > 0:
            # Use the shared network in eval mode for batched inference
            runner = BatchGameRunner(
                network=self.shared_network,
                concurrency=min(self.batch_concurrency, n_batched),
                oracle=self.shared_network.oracle_critic_enabled,
                compute=self.compute,
            )
            runner._rng = self._rng
            results = runner.run(
                total_games=n_batched,
                explore_rate=self.agents[0].explore_rate,
                dealer_offset=game_count,
                game_id_offset=game_count,
            )

            for result in results:
                scores = result.scores
                raw_score = scores.get(0, 0)

                # Convert GameExperience → Experience with rewards assigned
                game_exps: list[Experience] = []
                for gexp in result.experiences:
                    game_exps.append(Experience(
                        state=gexp.state,
                        action=gexp.action,
                        log_prob=gexp.log_prob,
                        value=gexp.value,
                        decision_type=gexp.decision_type,
                        reward=0.0,
                        done=False,
                        oracle_state=gexp.oracle_state,
                        legal_mask=gexp.legal_mask,
                        game_id=gexp.game_id,
                        step_in_game=gexp.step_in_game,
                    ))

                # Assign terminal reward to last experience
                if game_exps:
                    game_exps[-1].reward = raw_score / 100.0
                    game_exps[-1].done = True

                batched_experiences.extend(game_exps)

                # Determine if player 0 bid
                agent0_bids = (result.declarer_p0 and result.py_state.declarer == 0)

                # Determine if declarer lost (for defender win detection)
                declarer_lost = False
                if result.py_state.declarer is not None:
                    declarer_score = scores.get(result.py_state.declarer, 0)
                    declarer_lost = declarer_score < 0

                partner_p0 = result.py_state.partner == 0 if result.py_state.partner is not None else False
                batched_stats.append({
                    "raw_score": raw_score,
                    "contract_name": result.contract_name,
                    "is_klop": result.is_klop,
                    "is_solo": result.is_solo,
                    "declarer_p0": result.declarer_p0,
                    "partner_p0": partner_p0,
                    "agent0_bids": agent0_bids,
                    "declarer_lost": declarer_lost,
                    "all_scores": {p: scores.get(p, 0) for p in range(4)},
                    "initial_tarok_counts": result.initial_tarok_counts,
                    "game_mode": "batch",
                })

        # --- Run StockŠkis games sequentially ---
        if n_stockskis > 0:
            original_agents = self._enter_stockskis_mode()
            for g_idx in range(n_stockskis):
                if not self._running:
                    break
                for agent in self.agents:
                    if hasattr(agent, 'clear_experiences'):
                        agent.clear_experiences()

                game = RustGameLoop(self.agents)
                state, scores = await game.run(dealer=(game_count + n_batched + g_idx) % 4)

                # Only agent 0 records experiences vs StockŠkis
                reward = scores.get(0, 0) / 100.0
                self.agents[0].finalize_game(reward)
                stockskis_experiences.extend(self.agents[0].experiences)

                raw_score = scores.get(0, 0)
                contract_name = state.contract.name.lower() if state.contract else "klop"
                is_klop = state.contract is not None and state.contract.is_klop
                is_solo = state.contract is not None and state.contract.is_solo
                declarer_p0 = state.declarer == 0
                agent0_bids_list = [b for b in state.bids if b.player == 0 and b.contract is not None]

                # StockŠkis finishing places
                sorted_players = sorted(scores, key=lambda p: scores[p], reverse=True)
                places = {p: rank + 1 for rank, p in enumerate(sorted_players)}
                avg_sk_place = sum(places[p] for p in range(1, 4)) / 3.0

                declarer_lost = False
                if state.declarer is not None:
                    declarer_lost = scores.get(state.declarer, 0) < 0

                partner_p0 = state.partner == 0 if state.partner is not None else False
                stockskis_stats.append({
                    "raw_score": raw_score,
                    "contract_name": contract_name,
                    "is_klop": is_klop,
                    "is_solo": is_solo,
                    "declarer_p0": declarer_p0,
                    "partner_p0": partner_p0,
                    "agent0_bids": bool(agent0_bids_list),
                    "declarer_lost": declarer_lost,
                    "all_scores": {p: scores.get(p, 0) for p in range(4)},
                    "stockskis_place": avg_sk_place,
                    "initial_tarok_counts": state.initial_tarok_counts if hasattr(state, 'initial_tarok_counts') else {},
                    "game_mode": "stockskis",
                })

            self._exit_stockskis_mode(original_agents)

        # --- Run HoF games sequentially ---
        hof_experiences: list[Experience] = []
        hof_stats: list[dict] = []
        if n_hof > 0 and self._hof_opponent is not None:
            original_agents = self._enter_hof_mode()
            for g_idx in range(n_hof):
                if not self._running:
                    break
                for agent in self.agents:
                    if hasattr(agent, 'clear_experiences'):
                        agent.clear_experiences()

                game = RustGameLoop(self.agents)
                state, scores = await game.run(dealer=(game_count + n_batched + n_stockskis + g_idx) % 4)

                reward = scores.get(0, 0) / 100.0
                self.agents[0].finalize_game(reward)
                hof_experiences.extend(self.agents[0].experiences)

                raw_score = scores.get(0, 0)
                contract_name = state.contract.name.lower() if state.contract else "klop"
                is_klop = state.contract is not None and state.contract.is_klop
                is_solo = state.contract is not None and state.contract.is_solo
                declarer_p0 = state.declarer == 0
                agent0_bids_list = [b for b in state.bids if b.player == 0 and b.contract is not None]

                declarer_lost = False
                if state.declarer is not None:
                    declarer_lost = scores.get(state.declarer, 0) < 0

                partner_p0 = state.partner == 0 if state.partner is not None else False
                hof_stats.append({
                    "raw_score": raw_score,
                    "contract_name": contract_name,
                    "is_klop": is_klop,
                    "is_solo": is_solo,
                    "declarer_p0": declarer_p0,
                    "partner_p0": partner_p0,
                    "agent0_bids": bool(agent0_bids_list),
                    "declarer_lost": declarer_lost,
                    "initial_tarok_counts": state.initial_tarok_counts if hasattr(state, 'initial_tarok_counts') else {},
                    "game_mode": "hof",
                })

            self._exit_hof_mode(original_agents)

        # Merge HoF stats into stockskis stats (same format, processed the same way)
        stockskis_experiences.extend(hof_experiences)
        stockskis_stats.extend(hof_stats)

        return batched_experiences, batched_stats, stockskis_experiences, stockskis_stats

    async def _run_warmup(self) -> None:
        """Pre-train the value network from random Rust games."""
        from tarok.adapters.ai.warmup import warmup_value_network

        async def on_progress(info: dict):
            for cb in self._metrics_callback:
                # Reuse metrics structure to report warmup progress
                self.metrics.episode = 0
                self.metrics.avg_loss = info.get("chunk_loss", 0)
                self.metrics.games_per_second = info.get("gen_speed", 0)
                await cb(self.metrics)
            await asyncio.sleep(0)

        def sync_progress(info: dict):
            # Schedule the async callback on the event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(on_progress(info))

        result = warmup_value_network(
            network=self.shared_network,
            num_games=self.warmup_games,
            batch_size=2048,
            epochs=3,
            lr=1e-3,
            chunk_size=10_000,
            device=str(self.compute.device),
            include_oracle=self.shared_network.oracle_critic_enabled,
            progress_callback=sync_progress,
        )

        # Save a warmup checkpoint
        self._save_checkpoint(0, is_snapshot=True, custom_name="tarok_agent_warmup.pt")

    def _ppo_update(self, experiences: list[Experience]) -> dict[str, float]:
        """Perform PPO update on collected experiences, grouped by decision type."""
        if not experiences:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        # --- 1. Compute GAE per-game in temporal order ---
        # Group experiences by game_id, preserving step order
        from collections import defaultdict as _dd
        games: dict[int, list[Experience]] = _dd(list)
        for exp in experiences:
            games[exp.game_id].append(exp)

        gae_advantages: dict[int, float] = {}  # id(exp) -> advantage
        gae_returns: dict[int, float] = {}     # id(exp) -> return

        for game_exps in games.values():
            # Sort by temporal step within the game
            game_exps.sort(key=lambda e: e.step_in_game)
            n = len(game_exps)

            # Compute GAE backwards through the game
            advantages = [0.0] * n
            returns = [0.0] * n
            last_gae = 0.0
            for t in reversed(range(n)):
                exp = game_exps[t]
                if t == n - 1:
                    next_value = 0.0  # terminal
                else:
                    next_value = game_exps[t + 1].value.item()
                delta = exp.reward + self.gamma * next_value - exp.value.item()
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
                advantages[t] = last_gae
                returns[t] = last_gae + exp.value.item()

            for t, exp in enumerate(game_exps):
                gae_advantages[id(exp)] = advantages[t]
                gae_returns[id(exp)] = returns[t]

        # --- 2. Group by decision type for correct head routing ---
        grouped: dict[DecisionType, list[Experience]] = defaultdict(list)
        for exp in experiences:
            grouped[exp.decision_type].append(exp)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for dt, dt_exps in grouped.items():
            if not dt_exps:
                continue

            action_size = _ACTION_SIZES[dt]

            states = self.compute.stack_to_device([e.state for e in dt_exps])
            actions = self.compute.tensor_to_device([e.action for e in dt_exps], dtype=torch.long)
            old_log_probs = self.compute.stack_to_device([e.log_prob for e in dt_exps])

            # Oracle states for critic (if PTIE is active)
            has_oracle = dt_exps[0].oracle_state is not None
            oracle_states = (
                self.compute.stack_to_device([e.oracle_state for e in dt_exps])
                if has_oracle else None
            )

            # Use pre-computed GAE advantages and returns
            advantages = self.compute.tensor_to_device(
                [gae_advantages[id(e)] for e in dt_exps], dtype=torch.float32,
            )
            returns = self.compute.tensor_to_device(
                [gae_returns[id(e)] for e in dt_exps], dtype=torch.float32,
            )

            # Normalise advantages
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Use stored legal masks from game time
            if dt_exps[0].legal_mask is not None:
                legal_masks = self.compute.stack_to_device([e.legal_mask for e in dt_exps])
            else:
                legal_masks = torch.ones(len(dt_exps), action_size, dtype=torch.float32)
                legal_masks = self.compute.to_device(legal_masks)

            # Old values for value clipping
            old_values = self.compute.stack_to_device([e.value for e in dt_exps])

            for _ in range(self.epochs_per_update):
                indices = torch.randperm(len(dt_exps))

                for start in range(0, len(dt_exps), self.batch_size):
                    end = min(start + self.batch_size, len(dt_exps))
                    batch_idx = indices[start:end]

                    b_states = states[batch_idx]
                    b_actions = actions[batch_idx]
                    b_old_log_probs = old_log_probs[batch_idx]
                    b_advantages = advantages[batch_idx]
                    b_returns = returns[batch_idx]
                    b_masks = legal_masks[batch_idx]
                    b_oracle = oracle_states[batch_idx] if oracle_states is not None else None
                    b_old_values = old_values[batch_idx]

                    new_log_probs, new_values, entropy = self.shared_network.evaluate_action(
                        b_states, b_actions, b_masks, dt,
                        oracle_state=b_oracle,
                    )

                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    ) * b_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss with optional clipping (PPO best practice)
                    if self._value_clip > 0:
                        v_clipped = b_old_values + torch.clamp(
                            new_values - b_old_values,
                            -self._value_clip, self._value_clip,
                        )
                        vl_unclipped = (new_values - b_returns) ** 2
                        vl_clipped = (v_clipped - b_returns) ** 2
                        value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
                    else:
                        value_loss = nn.functional.mse_loss(new_values, b_returns)

                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy.mean()
                    )

                    # --- v2: Oracle guiding distillation loss ---
                    if (
                        self._oracle_guiding_coef > 0
                        and self.shared_network.oracle_critic_enabled
                        and b_oracle is not None
                    ):
                        actor_feats = self.shared_network.get_actor_features(b_states)
                        with torch.no_grad():
                            critic_feats = self.shared_network.get_critic_features(b_oracle)
                        oracle_guide_loss = 1.0 - nn.functional.cosine_similarity(
                            actor_feats, critic_feats, dim=-1
                        ).mean()
                        loss = loss + self._oracle_guiding_coef * oracle_guide_loss

                    # Skip corrupted batches
                    if not torch.isfinite(loss):
                        log.warning("Non-finite loss (%.4g) at session %d, skipping batch", loss.item(), self.metrics.session)
                        continue

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.shared_network.parameters(), 0.5)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates += 1

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "total_loss": (total_policy_loss + total_value_loss) / n,
        }

    def _save_checkpoint(self, episode: int, is_snapshot: bool = False, custom_name: str | None = None) -> dict:
        data = {
            "run_id": self.metrics.run_id,
            "episode": episode,
            "session": self.metrics.session,
            "model_state_dict": self.shared_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics.to_dict(),
        }

        # Always save as 'latest'
        latest = self.save_dir / "tarok_agent_latest.pt"
        torch.save(data, latest)

        if not is_snapshot and not custom_name:
            return {}

        file_name = custom_name if custom_name else f"tarok_agent_ep{episode}.pt"
        path = self.save_dir / file_name
        torch.save(data, path)

        info = {
            "filename": path.name,
            "run_id": self.metrics.run_id,
            "episode": episode,
            "session": self.metrics.session,
            "win_rate": round(self.metrics.avg_placement, 4),
            "avg_reward": round(self.metrics.avg_reward, 2),
            "games_per_second": round(self.metrics.games_per_second, 1),
        }
        return info

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.compute.device, weights_only=True)
        self.shared_network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Persona naming — random female name + surname + age
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Ana", "Maja", "Nina", "Eva", "Lara", "Sara", "Zala", "Nika", "Tina", "Katja",
    "Petra", "Urška", "Ines", "Lina", "Pia", "Mila", "Ajda", "Iris", "Hana", "Vida",
    "Ella", "Sofia", "Luna", "Alba", "Neja", "Teja", "Kaja", "Lea", "Anja", "Ema",
    "Vila", "Živa", "Meta", "Neža", "Brina", "Ula", "Alja", "Klara", "Gaja", "Lana",
]

_LAST_NAMES = [
    "Novak", "Horvat", "Kovač", "Krajnc", "Zupan", "Potočnik", "Mlakar", "Kos",
    "Vidmar", "Golob", "Turk", "Korošec", "Košir", "Bizjak", "Mezgec", "Oblak",
    "Kern", "Repnik", "Žagar", "Hribar", "Pintar", "Kolenc", "Štrukelj", "Ribič",
]


def _generate_persona(rng: random.Random | None = None) -> dict:
    """Generate a random persona for a lab-trained model."""
    r = rng or random.Random()
    first = r.choice(_FIRST_NAMES)
    last = r.choice(_LAST_NAMES)
    return {"first_name": first, "last_name": last, "age": 0}


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def _model_hash(network: TarokNet) -> str:
    """Short deterministic hash of model weights for unique identification."""
    h = hashlib.sha256()
    for p in network.parameters():
        h.update(p.data.cpu().numpy().tobytes()[:64])
    return h.hexdigest()[:8]


def _display_name(persona: dict, hash_str: str) -> str:
    """Human-readable model name: 'Ana Novak (age 3) #a1b2c3d4'."""
    return f"{persona['first_name']} {persona['last_name']} (age {persona['age']}) #{hash_str}"


# ---------------------------------------------------------------------------
# Hall of Fame — filesystem persistence
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINTS_DIR = BACKEND_ROOT / "checkpoints"
HOF_DIR = CHECKPOINTS_DIR / "hall_of_fame"

# Global HoF manager — max 10 auto models, pinned models are exempt
_hof_manager = HoFManager(HOF_DIR, max_auto=10)


def _ensure_hof_dir():
    HOF_DIR.mkdir(parents=True, exist_ok=True)


def get_hof_manager() -> HoFManager:
    """Return the global HoFManager instance."""
    return _hof_manager


def save_to_hof(
    network: TarokNet,
    persona: dict,
    eval_history: list[dict],
    phase_label: str = "",
    pinned: bool = False,
) -> dict:
    """Save a model snapshot to the Hall of Fame directory.

    If *pinned* is True the model is stored in the ``pinned/`` subdirectory
    and is exempt from auto-eviction.  Auto models are capped at
    ``_hof_manager.max_auto`` (default 10); weakest are auto-evicted.
    """
    return _hof_manager.save(network, persona, eval_history, phase_label=phase_label, pinned=pinned)


def list_hof() -> list[dict]:
    """List all Hall of Fame models (pinned first, then auto)."""
    return _hof_manager.list()


def remove_from_hof(model_hash: str) -> bool:
    """Remove a model from the Hall of Fame by model_hash.

    Works for both pinned and auto models.
    Returns True if the model was found and removed.
    """
    return _hof_manager.remove(model_hash)


def pin_hof(model_hash: str) -> bool:
    """Pin a HoF model (exempt from auto-eviction)."""
    return _hof_manager.pin(model_hash)


def unpin_hof(model_hash: str) -> bool:
    """Unpin a HoF model (subject to auto-eviction again)."""
    return _hof_manager.unpin(model_hash)


def promote_checkpoint_to_hof(checkpoint_filename: str) -> dict | None:
    """Promote an existing checkpoint file into the Hall of Fame.

    Loads the .pt file, extracts metadata, and saves it to the HoF directory.
    Returns the HoF entry info or None if the file doesn't exist / is invalid.
    """
    # Try checkpoints dir first, then HoF dir (for re-promoting)
    pt_path = CHECKPOINTS_DIR / checkpoint_filename
    if not pt_path.exists():
        return None
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    state_dict = data.get("model_state_dict")
    if state_dict is None:
        return None
    hidden_size = data.get("hidden_size", 256)
    net = TarokNet(hidden_size=hidden_size)
    net.load_state_dict(state_dict)
    persona = data.get("persona", {"first_name": "Promoted", "last_name": "Model", "age": 0})
    eval_history = data.get("eval_history", [])
    phase_label = data.get("phase_label", "manual-promote")
    return save_to_hof(net, persona, eval_history, phase_label=phase_label)


def _resolve_checkpoint_path(choice: str) -> Path:
    """Resolve a checkpoint choice to a file under checkpoints/."""
    root = CHECKPOINTS_DIR.resolve()
    if choice == "latest":
        path = (root / "tarok_agent_latest.pt").resolve()
    else:
        path = (root / choice).resolve()

    if not path.is_relative_to(root):
        raise ValueError("checkpoint must be inside checkpoints/")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass
class LabState:
    """Mutable state for the training lab."""
    network: TarokNet | None = None
    hidden_size: int = 256
    phase: str = "idle"  # idle | evaluating | training | self_play | done
    loaded_from_checkpoint: bool = False
    # Persona identity
    persona: dict = field(default_factory=dict)
    model_hash: str = ""
    display_name: str = ""
    # Evaluation results
    eval_history: list[dict] = field(default_factory=list)
    # Training progress
    training_step: int = 0
    total_training_steps: int = 0
    current_loss: float = 0.0
    expert_games_generated: int = 0
    expert_experiences: int = 0
    training_sessions_done: int = 0
    total_training_sessions: int = 0
    # Self-play stats
    self_play_games: int = 0
    self_play_sessions: int = 0
    # Per-session training metrics (like TrainingDashboard)
    sp_avg_placement: float = 0.0
    sp_avg_reward: float = 0.0
    sp_avg_score: float = 0.0
    sp_bid_rate: float = 0.0
    sp_klop_rate: float = 0.0
    sp_solo_rate: float = 0.0
    sp_games_per_second: float = 0.0
    # History arrays (one entry per session)
    sp_avg_placement_history: list[float] = field(default_factory=list)
    sp_avg_reward_history: list[float] = field(default_factory=list)
    sp_avg_score_history: list[float] = field(default_factory=list)
    sp_loss_history: list[float] = field(default_factory=list)
    sp_bid_rate_history: list[float] = field(default_factory=list)
    sp_klop_rate_history: list[float] = field(default_factory=list)
    sp_solo_rate_history: list[float] = field(default_factory=list)
    # Score extremes per session (detect valat disasters)
    sp_min_score_history: list[float] = field(default_factory=list)
    sp_max_score_history: list[float] = field(default_factory=list)
    # Active training program: "imitation" | "self_play"
    active_program: str = ""
    # Running task
    running: bool = False
    error: str | None = None
    # Saved snapshots
    snapshots: list[dict] = field(default_factory=list)
    # Population Based Training state
    pbt_enabled: bool = False
    pbt_generation: int = 0
    pbt_total_generations: int = 0
    pbt_population_size: int = 0
    pbt_member_index: int = 0
    pbt_member_total: int = 0
    pbt_population: list[dict[str, Any]] = field(default_factory=list)
    pbt_generation_history: list[dict[str, Any]] = field(default_factory=list)
    pbt_events: list[dict[str, Any]] = field(default_factory=list)
    # Training summary (populated at completion)
    training_summary: dict[str, Any] | None = None
    # Opponent pool stats (updated per session during self-play)
    opponent_stats: dict[str, dict] = field(default_factory=dict)
    # Contract stats (rolling window, updated per session)
    contract_stats: dict[str, dict] = field(default_factory=dict)
    # Tarok count vs contract distribution
    tarok_count_bids: dict[str, dict[str, int]] = field(default_factory=dict)


# Global lab state
_lab = LabState()
_lab_task: asyncio.Task | None = None

# Process pool for background evaluation (uses idle CPU cores)
_eval_pool: ProcessPoolExecutor | None = None


def _get_eval_pool() -> ProcessPoolExecutor:
    global _eval_pool
    if _eval_pool is None:
        workers = max(2, (os.cpu_count() or 4) // 2)
        _eval_pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        )
    return _eval_pool


def _eval_member_in_process(
    state_dict: dict, hidden_size: int, eval_games: int,
    eval_bots: list[str] | None = None,
) -> dict:
    """Run in a subprocess: evaluate one network snapshot against selected bots."""
    if eval_bots is None:
        eval_bots = ["v1", "v2", "v3", "v5"]
    net = TarokNet(hidden_size=hidden_size)
    net.load_state_dict(state_dict)
    net.eval()
    return {
        f"vs_{v}": _evaluate_vs_bots_sync(net, eval_games, v)
        for v in eval_bots
    }


# Training pool — runs member self-play + PPO in parallel on idle CPU cores
_train_pool: ProcessPoolExecutor | None = None


def _get_train_pool() -> ProcessPoolExecutor:
    global _train_pool
    if _train_pool is None:
        workers = max(2, (os.cpu_count() or 4) // 2)
        _train_pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        )
    return _train_pool


def _train_member_in_process(
    network_state: dict,
    optimizer_state: dict,
    hparams: dict[str, Any],
    hidden_size: int,
    num_sessions: int,
    games_per_session: int,
    start_game_index: int,
    member_index: int = 0,
    progress_queue: Any = None,
) -> dict[str, Any]:
    """Run in a subprocess: self-play + PPO for one population member.

    Returns updated network/optimizer state_dicts and training metrics.
    Sends periodic progress updates via *progress_queue* if provided.
    """
    agents = [
        RLAgent(name=f"P{s}", hidden_size=hidden_size, device="cpu",
                explore_rate=float(hparams.get("explore_rate", 0.1)))
        for s in range(4)
    ]
    agents[0].network.load_state_dict(network_state)
    for a in agents[1:]:
        a.network = agents[0].network

    trainer = PPOTrainer(
        agents=agents,
        lr=float(hparams.get("lr", 3e-4)),
        gamma=float(hparams["gamma"]),
        gae_lambda=float(hparams["gae_lambda"]),
        clip_epsilon=float(hparams["clip_epsilon"]),
        value_coef=float(hparams["value_coef"]),
        entropy_coef=float(hparams["entropy_coef"]),
        epochs_per_update=int(hparams["epochs_per_update"]),
        batch_size=int(hparams["batch_size"]),
        games_per_session=games_per_session,
        device="cpu",
        stockskis_ratio=0.0,
        fsp_ratio=0.0,
        use_rust_engine=True,
        save_dir="/tmp/pbt_worker",
        batch_concurrency=32,
        value_clip=0.2,
    )
    trainer.optimizer.load_state_dict(optimizer_state)
    trainer._running = True

    import time as _time

    total_rewards = 0.0
    total_wins = 0.0
    total_scores = 0.0
    games_played = 0
    all_experiences: list = []
    t_start = _time.perf_counter()

    loop = asyncio.new_event_loop()
    try:
        for session_offset in range(num_sessions):
            exps, stats, sk_exps, sk_stats = loop.run_until_complete(
                trainer._play_session_batched(
                    session_offset, start_game_index + games_played, _time.time(),
                )
            )
            all_experiences.extend(exps)
            all_experiences.extend(sk_exps)
            session_wins = 0.0
            session_scores = 0.0
            for stat in stats + sk_stats:
                raw_score = stat["raw_score"]
                total_scores += raw_score
                session_scores += raw_score
                total_rewards += raw_score / 100.0
                total_wins += 1.0 if raw_score > 0 else 0.0
                session_wins += 1.0 if raw_score > 0 else 0.0
                games_played += 1
            trainer.metrics.session += 1

            # Report progress after each session
            if progress_queue is not None:
                elapsed = _time.perf_counter() - t_start
                n_session_games = len(stats) + len(sk_stats)
                try:
                    progress_queue.put_nowait({
                        "type": "session",
                        "member": member_index,
                        "session": session_offset + 1,
                        "total_sessions": num_sessions,
                        "games": games_played,
                        "games_per_sec": games_played / max(elapsed, 0.01),
                        "win_rate": total_wins / max(games_played, 1),
                        "avg_score": total_scores / max(games_played, 1),
                    })
                except Exception:
                    pass
    finally:
        loop.close()

    # Report PPO phase
    if progress_queue is not None:
        try:
            progress_queue.put_nowait({
                "type": "ppo_start",
                "member": member_index,
                "experiences": len(all_experiences),
            })
        except Exception:
            pass

    loss_info = trainer._ppo_update(all_experiences) if all_experiences else {}
    n = max(games_played, 1)

    if progress_queue is not None:
        try:
            progress_queue.put_nowait({
                "type": "done",
                "member": member_index,
                "games": games_played,
                "elapsed": _time.perf_counter() - t_start,
            })
        except Exception:
            pass

    return {
        "network_state": trainer.shared_network.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "avg_reward": total_rewards / n,
        "win_rate": total_wins / n,
        "avg_score": total_scores / n,
        "loss": loss_info.get("policy_loss", 0.0),
        "games": games_played,
    }


_PBT_HPARAM_SPACE: dict[str, dict[str, Any]] = {
    "lr": {"low": 1e-5, "high": 5e-3, "log": True, "integer": False, "sigma": 0.18},
    "gamma": {"low": 0.9, "high": 0.999, "log": False, "integer": False, "sigma": 0.015},
    "gae_lambda": {"low": 0.85, "high": 0.99, "log": False, "integer": False, "sigma": 0.02},
    "clip_epsilon": {"low": 0.05, "high": 0.35, "log": False, "integer": False, "sigma": 0.05},
    "value_coef": {"low": 0.1, "high": 1.2, "log": False, "integer": False, "sigma": 0.12},
    "entropy_coef": {"low": 0.001, "high": 0.08, "log": True, "integer": False, "sigma": 0.2},
    "epochs_per_update": {"low": 2, "high": 8, "log": False, "integer": True, "sigma": 1.0},
    "batch_size": {"low": 32, "high": 256, "log": False, "integer": True, "sigma": 24.0},
    "explore_rate": {"low": 0.02, "high": 0.25, "log": False, "integer": False, "sigma": 0.03},
    "fsp_ratio": {"low": 0.0, "high": 0.5, "log": False, "integer": False, "sigma": 0.08},
}


def _default_pbt_hparams(learning_rate: float, fsp_ratio: float) -> dict[str, Any]:
    return {
        "lr": learning_rate,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "epochs_per_update": 4,
        "batch_size": 64,
        "explore_rate": 0.1,
        "fsp_ratio": fsp_ratio,
    }


def _round_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in hparams.items():
        if isinstance(value, int):
            rounded[key] = value
        else:
            rounded[key] = round(float(value), 6)
    return rounded


def _clip_hparam(name: str, value: float) -> Any:
    spec = _PBT_HPARAM_SPACE[name]
    clipped = max(spec["low"], min(spec["high"], value))
    if spec["integer"]:
        return int(round(clipped))
    return float(clipped)


def _mutate_hparams(base_hparams: dict[str, Any], rng: random.Random, scale: float) -> dict[str, Any]:
    mutated = dict(base_hparams)
    mutated_count = 0

    for name, spec in _PBT_HPARAM_SPACE.items():
        if rng.random() > 0.55:
            continue

        mutated_count += 1
        value = float(mutated[name])
        sigma = spec["sigma"] * max(scale, 1e-3)
        if spec["log"]:
            log_value = math.log10(max(value, spec["low"]))
            log_value += rng.gauss(0.0, sigma)
            value = 10 ** log_value
        else:
            value += rng.gauss(0.0, sigma)
        mutated[name] = _clip_hparam(name, value)

    if mutated_count == 0:
        forced = rng.choice(list(_PBT_HPARAM_SPACE))
        spec = _PBT_HPARAM_SPACE[forced]
        value = float(mutated[forced])
        if spec["log"]:
            value = 10 ** (math.log10(max(value, spec["low"])) + rng.gauss(0.0, spec["sigma"] * max(scale, 1e-3)))
        else:
            value += rng.gauss(0.0, spec["sigma"] * max(scale, 1e-3))
        mutated[forced] = _clip_hparam(forced, value)

    return _round_hparams(mutated)


def _score_game(state, scores: dict[int, int], player_idx: int = 0) -> tuple[int, float]:
    raw_score = scores.get(player_idx, 0)
    is_klop = state.contract is not None and state.contract.is_klop

    if is_klop:
        win = 1.0 if raw_score > 0 else 0.0
    elif state.declarer == player_idx:
        win = 1.0 if raw_score > 0 else 0.0
    else:
        declarer_score = scores.get(state.declarer, 0) if state.declarer is not None else 0
        win = 1.0 if declarer_score < 0 else 0.0

    return raw_score, win


def _apply_member_hparams(member: dict[str, Any]) -> None:
    trainer = member["trainer"]
    agents = member["agents"]
    hparams = member["hparams"]

    trainer.gamma = float(hparams["gamma"])
    trainer.gae_lambda = float(hparams["gae_lambda"])
    trainer.clip_epsilon = float(hparams["clip_epsilon"])
    trainer.value_coef = float(hparams["value_coef"])
    trainer.entropy_coef = float(hparams["entropy_coef"])
    trainer.epochs_per_update = int(hparams["epochs_per_update"])
    trainer.batch_size = int(hparams["batch_size"])
    trainer.fsp_ratio = float(hparams["fsp_ratio"])

    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = float(hparams["lr"])

    if trainer.fsp_ratio > 0 and trainer.opponent_network is None:
        hidden_size = trainer.shared_network.shared[0].out_features
        trainer.opponent_network = TarokNet(
            hidden_size=hidden_size,
            oracle_critic=trainer.shared_network.oracle_critic_enabled,
        ).to(trainer.device)

    for agent in agents:
        agent.explore_rate = float(hparams["explore_rate"])


def _snapshot_member(member: dict[str, Any]) -> dict[str, Any]:
    trainer = member["trainer"]
    return {
        "network_state": copy.deepcopy(trainer.shared_network.state_dict()),
        "optimizer_state": copy.deepcopy(trainer.optimizer.state_dict()),
        "hparams": dict(member["hparams"]),
        "best_eval": copy.deepcopy(member.get("best_eval") or {}),
    }


def _restore_member_from_snapshot(member: dict[str, Any], snapshot: dict[str, Any]) -> None:
    trainer = member["trainer"]
    trainer.shared_network.load_state_dict(snapshot["network_state"])
    trainer.optimizer.load_state_dict(snapshot["optimizer_state"])
    member["hparams"] = _round_hparams(snapshot["hparams"])
    member["best_eval"] = copy.deepcopy(snapshot.get("best_eval") or {})
    _apply_member_hparams(member)


def _member_state_dict(member: dict[str, Any]) -> dict[str, Any]:
    best_eval = member.get("best_eval") or {}
    out = {
        "index": member["index"],
        "label": member["label"],
        "fitness": round(float(member.get("fitness", 0.0)), 4),
        "batch_avg_reward": round(float(member.get("batch_avg_reward", 0.0)), 4),
        "batch_win_rate": round(float(member.get("batch_win_rate", 0.0)), 4),
        "loss": round(float(member.get("loss", 0.0)), 4),
        "games": int(member.get("games", 0)),
        "status": member.get("status", "idle"),
        "copied_from": member.get("copied_from"),
        "mutations": int(member.get("mutations", 0)),
        "survival_count": int(member.get("survival_count", 0)),
        "model_hash": member.get("model_hash", ""),
        "hparams": _round_hparams(member.get("hparams", {})),
    }
    for v in ["v1", "v2", "v3", "v4", "v5"]:
        if f"vs_{v}" in best_eval:
            out[f"vs_{v}"] = round(float(best_eval[f"vs_{v}"]), 4)
            out[f"avg_score_{v}"] = round(float(best_eval.get(f"avg_score_{v}", 0.0)), 2)
    return out


def _sync_population_state(members: list[dict[str, Any]]) -> None:
    global _lab
    _lab.pbt_population = [_member_state_dict(member) for member in members]


def _create_network(hidden_size: int = 256) -> TarokNet:
    """Create a fresh TarokNet with random weights."""
    net = TarokNet(hidden_size=hidden_size)
    return net


def _network_device(network: TarokNet) -> torch.device:
    """Return the device the network currently lives on."""
    return next(network.parameters()).device


def _make_eval_agent(network: TarokNet) -> RLAgent:
    """Wrap a TarokNet in an RLAgent set to greedy evaluation mode."""
    dev = _network_device(network)
    hidden_size = network.shared[0].out_features
    agent = RLAgent(name="Lab-NN", hidden_size=hidden_size, device=str(dev), explore_rate=0.0)
    # Reuse the current network in-place; do not move shared weights across devices.
    agent.network = network
    agent.set_training(False)
    return agent


def _make_opponents(version: str) -> list:
    """Create 3 heuristic bot opponents."""
    return [StockSkisPlayerV5(name=f"V5-{i}") for i in range(3)]


def _evaluate_vs_bots_sync(
    network: TarokNet,
    num_games: int = 100,
    version: str = "v1",
) -> dict:
    """Synchronous eval — can run in a thread pool."""
    network.eval()
    agent = _make_eval_agent(network)
    opponents = _make_opponents(version)

    wins = 0
    total_diff = 0
    total_points = 0
    total_place = 0.0
    total_bids = 0
    decl_games = 0
    decl_wins = 0
    decl_points = 0
    def_games = 0
    def_wins = 0
    def_points = 0
    contract_breakdown: dict[str, dict[str, float]] = {
        name: {
            "games": 0,
            "wins": 0,
            "points": 0.0,
            "decl_games": 0,
            "decl_wins": 0,
            "decl_points": 0.0,
            "def_games": 0,
            "def_wins": 0,
            "def_points": 0.0,
        }
        for name in ["klop", "three", "two", "one", "solo_three", "solo_two", "solo_one", "solo", "berac"]
    }

    loop = asyncio.new_event_loop()
    try:
        for g in range(num_games):
            players = [agent] + opponents
            game = GameLoop(players, rng=random.Random(g))
            _state, scores = loop.run_until_complete(game.run(dealer=g % 4))

            raw_score = scores.get(0, 0)
            opp_avg = sum(scores.get(i, 0) for i in range(1, 4)) / 3
            total_diff += raw_score - opp_avg
            total_points += raw_score

            # Placement (1=best, 4=worst) for evaluation richness.
            sorted_players = sorted(scores, key=lambda p: scores[p], reverse=True)
            places = {p: rank + 1 for rank, p in enumerate(sorted_players)}
            total_place += float(places.get(0, 4))

            # Bid participation as a behaviour signal.
            p0_bids = [b for b in _state.bids if b.player == 0 and b.contract is not None]
            total_bids += 1 if p0_bids else 0

            is_klop = _state.contract is not None and _state.contract.is_klop
            if is_klop:
                won = raw_score > 0
            elif _state.declarer == 0:
                won = raw_score > 0
            else:
                declarer_score = scores.get(_state.declarer, 0) if _state.declarer is not None else 0
                won = declarer_score < 0

            if won:
                wins += 1

            declarer_p0 = _state.declarer == 0
            if declarer_p0:
                decl_games += 1
                decl_points += raw_score
                if won:
                    decl_wins += 1
            else:
                def_games += 1
                def_points += raw_score
                if won:
                    def_wins += 1

            contract_name = _state.contract.name.lower() if _state.contract is not None else "klop"
            if contract_name in contract_breakdown:
                c = contract_breakdown[contract_name]
                c["games"] += 1
                c["points"] += float(raw_score)
                if won:
                    c["wins"] += 1
                if declarer_p0:
                    c["decl_games"] += 1
                    c["decl_points"] += float(raw_score)
                    if won:
                        c["decl_wins"] += 1
                else:
                    c["def_games"] += 1
                    c["def_points"] += float(raw_score)
                    if won:
                        c["def_wins"] += 1

            agent.clear_experiences()
    finally:
        loop.close()

    win_rate = wins / max(num_games, 1)
    avg_score = total_diff / max(num_games, 1)
    avg_points = total_points / max(num_games, 1)
    avg_place = total_place / max(num_games, 1)
    bid_rate = total_bids / max(num_games, 1)
    decl_win_rate = decl_wins / max(decl_games, 1)
    def_win_rate = def_wins / max(def_games, 1)
    decl_avg_points = decl_points / max(decl_games, 1)
    def_avg_points = def_points / max(def_games, 1)

    # Composite eval signal prioritizing win rate, then points, then placement.
    score_term = max(-1.0, min(1.0, avg_points / 120.0))
    place_term = max(0.0, min(1.0, (4.0 - avg_place) / 3.0))
    eval_signal = 0.55 * win_rate + 0.30 * ((score_term + 1.0) / 2.0) + 0.15 * place_term

    contract_stats_out: dict[str, dict[str, float]] = {}
    for cname, c in contract_breakdown.items():
        games = int(c["games"])
        decl_games_c = int(c["decl_games"])
        def_games_c = int(c["def_games"])
        contract_stats_out[cname] = {
            "games": games,
            "win_rate": round(float(c["wins"]) / max(games, 1), 4),
            "avg_points": round(float(c["points"]) / max(games, 1), 2),
            "decl_games": decl_games_c,
            "decl_win_rate": round(float(c["decl_wins"]) / max(decl_games_c, 1), 4),
            "decl_avg_points": round(float(c["decl_points"]) / max(decl_games_c, 1), 2),
            "def_games": def_games_c,
            "def_win_rate": round(float(c["def_wins"]) / max(def_games_c, 1), 4),
            "def_avg_points": round(float(c["def_points"]) / max(def_games_c, 1), 2),
        }

    return {
        "win_rate": round(win_rate, 4),
        "avg_score": round(avg_score, 2),
        "avg_points": round(avg_points, 2),
        "avg_place": round(avg_place, 2),
        "bid_rate": round(bid_rate, 4),
        "decl_win_rate": round(decl_win_rate, 4),
        "def_win_rate": round(def_win_rate, 4),
        "decl_avg_points": round(decl_avg_points, 2),
        "def_avg_points": round(def_avg_points, 2),
        "eval_signal": round(eval_signal, 4),
        "contract_stats": contract_stats_out,
        "games": num_games,
    }


async def _evaluate_vs_bots(
    network: TarokNet,
    num_games: int = 100,
    version: str = "v1",
) -> dict:
    """Evaluate NN agent vs heuristic bots, running in a thread to keep the event loop responsive."""
    return await asyncio.get_event_loop().run_in_executor(
        None, _evaluate_vs_bots_sync, network, num_games, version,
    )


def _evaluate_vs_nn_opponent_sync(
    network: TarokNet,
    opponent_state_dict: dict,
    opponent_hidden_size: int,
    num_games: int = 20,
) -> dict:
    """Evaluate network against a frozen NN opponent (HoF or FSP snapshot)."""
    import logging
    log = logging.getLogger(__name__)

    network.eval()
    agent = _make_eval_agent(network)

    # Build opponent network + agents
    opp_net = TarokNet(hidden_size=opponent_hidden_size)
    try:
        opp_net.load_state_dict(opponent_state_dict)
    except Exception:
        log.warning("Failed to load opponent state dict for evaluation")
        return {"win_rate": 0.0, "avg_score": 0.0, "avg_place": 4.0, "games": 0}
    opp_net.eval()

    opp_agents = []
    for i in range(3):
        a = RLAgent(name=f"Opp-{i}", hidden_size=opponent_hidden_size, device="cpu", explore_rate=0.0)
        a.network = opp_net
        a.set_training(False)
        opp_agents.append(a)

    wins = 0
    total_score = 0
    total_place = 0.0

    loop = asyncio.new_event_loop()
    try:
        for g in range(num_games):
            players = [agent] + opp_agents
            game = GameLoop(players, rng=random.Random(g + 9999))
            _state, scores = loop.run_until_complete(game.run(dealer=g % 4))

            raw_score = scores.get(0, 0)
            total_score += raw_score

            sorted_players = sorted(scores, key=lambda p: scores[p], reverse=True)
            places = {p: rank + 1 for rank, p in enumerate(sorted_players)}
            total_place += float(places.get(0, 4))

            is_klop = _state.contract is not None and _state.contract.is_klop
            if is_klop:
                won = raw_score > 0
            elif _state.declarer == 0:
                won = raw_score > 0
            else:
                declarer_score = scores.get(_state.declarer, 0) if _state.declarer is not None else 0
                won = declarer_score < 0
            if won:
                wins += 1

            agent.clear_experiences()
    finally:
        loop.close()

    n = max(num_games, 1)
    return {
        "win_rate": round(wins / n, 4),
        "avg_score": round(total_score / n, 2),
        "avg_place": round(total_place / n, 2),
        "games": num_games,
    }


async def _evaluate_vs_nn_opponent(
    network: TarokNet,
    opponent_state_dict: dict,
    opponent_hidden_size: int,
    num_games: int = 20,
) -> dict:
    """Async wrapper for NN opponent evaluation."""
    return await asyncio.get_event_loop().run_in_executor(
        None, _evaluate_vs_nn_opponent_sync, network, opponent_state_dict,
        opponent_hidden_size, num_games,
    )


async def _evaluate_vs_opponents(
    network: TarokNet,
    trainer,
    num_games: int = 20,
) -> dict[str, dict]:
    """Evaluate network against configured opponents (HoF models, FSP snapshots).

    Returns {opponent_label: {win_rate, avg_score, avg_place, games}}.
    """
    import logging
    log = logging.getLogger(__name__)
    results: dict[str, dict] = {}

    # Evaluate against each HoF model
    if hasattr(trainer, '_hof_opponent') and trainer._hof_opponent is not None:
        hof_opp = trainer._hof_opponent
        hof_opp._refresh_cache()
        for model_path in hof_opp._cached_models:
            try:
                data = torch.load(model_path, map_location="cpu", weights_only=False)
                state_dict = data.get("model_state_dict")
                if state_dict is None:
                    continue
                hidden_size = data.get("hidden_size", network.shared[0].out_features)
                # Extract human-readable label from filename
                # e.g. hof_Ema_Mlakar_age28_c6fa1e53.pt -> Ema Mlakar (28)
                name = model_path.stem  # without .pt
                label = _hof_filename_to_label(name)
                r = await _evaluate_vs_nn_opponent(
                    network, state_dict, hidden_size, num_games,
                )
                results[f"hof:{label}"] = r
            except Exception:
                log.warning("Failed to evaluate vs HoF model %s", model_path.name)

    # Evaluate against latest FSP snapshot
    if hasattr(trainer, 'network_bank') and trainer.network_bank.is_ready:
        snap, snapshot_id = trainer.network_bank.sample()
        if snap is not None:
            hidden_size = network.shared[0].out_features
            r = await _evaluate_vs_nn_opponent(
                network, snap, hidden_size, num_games,
            )
            results[f"fsp:{snapshot_id}"] = r

    return results


def _hof_filename_to_label(name: str) -> str:
    """Convert HoF filename stem to a readable label.

    e.g. 'hof_Ema_Mlakar_age28_c6fa1e53' -> 'Ema Mlakar (28)'
    """
    import re
    name = name.removeprefix("hof_")
    # Try to extract name parts and age
    m = re.match(r'(.+?)_age(\d+)_([a-f0-9]+)$', name)
    if m:
        person = m.group(1).replace('_', ' ')
        age = m.group(2)
        return f"{person} ({age})"
    # Try island format: Name_Name_island1_gen2_hash
    m = re.match(r'(.+?)_island(\d+)_gen(\d+)_([a-f0-9]+)$', name)
    if m:
        person = m.group(1).replace('_', ' ')
        island = m.group(2)
        gen = m.group(3)
        return f"{person} i{island}g{gen}"
    return name.replace('_', ' ')


async def _evaluate_vs_selected_bots(
    network: TarokNet,
    num_games: int,
    eval_bots: list[str],
) -> dict[str, dict]:
    """Evaluate NN against each selected bot version.
    Returns {version_str: {win_rate, avg_score, games}}."""
    results = {}
    for v in eval_bots:
        results[v] = await _evaluate_vs_bots(network, num_games, v)
    return results


def _eval_results_to_history_entry(
    results: dict[str, dict],
    step: int,
    label: str,
    program: str,
    loss: float = 0,
    experiences: int = 0,
    games: int = 0,
) -> dict:
    """Convert per-version eval results into an eval_history entry."""
    entry: dict = {
        "step": step,
        "label": label,
        "program": program,
        "loss": loss,
        "experiences": experiences,
        "games": games,
    }
    for v, r in results.items():
        entry[f"vs_{v}"] = r["win_rate"]
        entry[f"avg_score_{v}"] = r["avg_score"]
        entry[f"avg_points_{v}"] = r.get("avg_points", 0.0)
        entry[f"avg_place_{v}"] = r.get("avg_place", 0.0)
        entry[f"bid_rate_{v}"] = r.get("bid_rate", 0.0)
        entry[f"decl_win_rate_{v}"] = r.get("decl_win_rate", 0.0)
        entry[f"def_win_rate_{v}"] = r.get("def_win_rate", 0.0)
        entry[f"decl_avg_points_{v}"] = r.get("decl_avg_points", 0.0)
        entry[f"def_avg_points_{v}"] = r.get("def_avg_points", 0.0)
        entry[f"eval_signal_{v}"] = r.get("eval_signal", 0.0)
        if v == "v5":
            entry["contract_stats_v5"] = r.get("contract_stats", {})
    return entry


async def _save_sample_replay(network: TarokNet, generation: int, member: dict[str, Any]) -> None:
    sample_agent = _make_eval_agent(network)
    opponents = _make_opponents("v3")
    players = [sample_agent] + opponents
    player_names = [member["label"], "V3-0", "V3-1", "V3-2"]
    replay_name = f"lab-pbt-gen-{generation:03d}-member-{member['index'] + 1}.json"
    observer = SpectatorObserver(
        websockets=[],
        player_names=player_names,
        delay=0,
        replay_name=replay_name,
        replay_metadata={
            "source": "lab_pbt",
            "label": f"PBT Gen {generation} Sample",
            "generation": generation,
            "member_index": member["index"],
            "member_label": member["label"],
        },
    )
    game = GameLoop(players, observer=observer, rng=random.Random(generation))
    await game.run(dealer=generation % 4)


async def _run_lab_session(
    expert_games: int,
    expert_source: str,
    eval_bots: list[str],
    training_epochs: int,
    eval_games: int,
    num_rounds: int,
    batch_size: int,
    learning_rate: float,
    chunk_size: int,
):
    """Imitation learning pipeline:
    1. Eval fresh network (plays real games — should be helpless)
    2. For each round: train on v2/v3 expert data → eval vs selected bots → save snapshot
    """
    global _lab

    try:
        _lab.phase = "evaluating"
        _lab.active_program = "imitation"
        _lab.total_training_sessions = num_rounds
        existing_il_rounds = sum(1 for entry in _lab.eval_history if entry.get("program") == "imitation")

        if not _lab.loaded_from_checkpoint or not _lab.eval_history:
            # Initial eval for a fresh model, or a loaded checkpoint without saved eval history.
            results = await _evaluate_vs_selected_bots(_lab.network, eval_games, eval_bots)
            entry = _eval_results_to_history_entry(
                results,
                step=len(_lab.eval_history),
                label="Loaded checkpoint" if _lab.loaded_from_checkpoint else "Fresh (random)",
                program="init",
            )
            _lab.eval_history.append(entry)
            _update_persona_hash()
            await asyncio.sleep(0)

        if not _lab.running:
            return

        games_per_round = expert_games // num_rounds
        loop = asyncio.get_event_loop()

        for round_idx in range(num_rounds):
            if not _lab.running:
                break

            _lab.phase = "training"
            _lab.training_sessions_done = round_idx
            await asyncio.sleep(0)

            def _train_round():
                result = imitation_pretrain(
                    _lab.network,
                    num_games=games_per_round,
                    batch_size=batch_size,
                    epochs=training_epochs,
                    lr=learning_rate,
                    chunk_size=min(chunk_size, games_per_round),
                    device=_detect_device(),
                    include_oracle=False,
                    progress_callback=lambda info: _on_training_progress(info),
                    expert_source=expert_source,
                )
                return result

            result = await loop.run_in_executor(None, _train_round)

            _lab.expert_games_generated += games_per_round
            _lab.expert_experiences += result.get("total_experiences", 0)
            _lab.current_loss = result.get("avg_policy_loss", 0)

            # Age the persona after each training round
            _lab.persona["age"] = _lab.persona.get("age", 0) + 1

            if not _lab.running:
                break

            # Evaluate
            _lab.phase = "evaluating"
            await asyncio.sleep(0)

            results = await _evaluate_vs_selected_bots(_lab.network, eval_games, eval_bots)

            step = len(_lab.eval_history)
            entry = _eval_results_to_history_entry(
                results,
                step=step,
                label=f"IL Round {existing_il_rounds + round_idx + 1}",
                program="imitation",
                loss=_lab.current_loss,
                experiences=_lab.expert_experiences,
                games=_lab.expert_games_generated,
            )
            _lab.eval_history.append(entry)

            # Save snapshot to HOF
            _update_persona_hash()
            info = save_to_hof(
                _lab.network, _lab.persona, _lab.eval_history,
                phase_label=f"imitation-r{round_idx + 1}",
            )
            _lab.snapshots.append(info)

        _lab.phase = "done"
        _lab.training_sessions_done = num_rounds

    except Exception as e:
        _lab.phase = "idle"
        _lab.error = str(e)
        import traceback
        traceback.print_exc()
    finally:
        _lab.running = False
        # Move network back to CPU for serving / checkpoint saving
        if _lab.network is not None:
            _lab.network = _lab.network.to("cpu")


def _build_pbt_population(
    population_size: int,
    hidden_size: int,
    device: str,
    seed_network: TarokNet,
    learning_rate: float,
    stockskis_ratio: float,
    fsp_ratio: float,
    mutation_scale: float,
) -> list[dict[str, Any]]:
    base_state = copy.deepcopy(seed_network.state_dict())
    base_hparams = _default_pbt_hparams(learning_rate, fsp_ratio)
    members: list[dict[str, Any]] = []
    rng = random.Random(time.time())

    for index in range(population_size):
        agents = [
            RLAgent(
                name=f"PBT-{index}-{seat}",
                hidden_size=hidden_size,
                device=device,
                explore_rate=float(base_hparams["explore_rate"]),
            )
            for seat in range(4)
        ]
        agents[0].network.load_state_dict(base_state)
        shared_network = agents[0].network
        for agent in agents[1:]:
            agent.network = shared_network

        trainer = PPOTrainer(
            agents=agents,
            lr=learning_rate,
            gamma=float(base_hparams["gamma"]),
            gae_lambda=float(base_hparams["gae_lambda"]),
            clip_epsilon=float(base_hparams["clip_epsilon"]),
            value_coef=float(base_hparams["value_coef"]),
            entropy_coef=float(base_hparams["entropy_coef"]),
            epochs_per_update=int(base_hparams["epochs_per_update"]),
            batch_size=int(base_hparams["batch_size"]),
            games_per_session=1,
            device=device,
            stockskis_ratio=stockskis_ratio,
            stockskis_strength=1.0,
            fsp_ratio=float(base_hparams["fsp_ratio"]),
            bank_size=20,
            use_rust_engine=True,
            save_dir=str(CHECKPOINTS_DIR / "pbt_tmp" / f"member_{index}"),
            value_clip=0.2,
        )
        trainer._running = True

        hparams = dict(base_hparams)
        if index > 0:
            hparams = _mutate_hparams(hparams, rng, mutation_scale)

        member = {
            "index": index,
            "label": f"Member {index + 1}",
            "trainer": trainer,
            "agents": agents,
            "hparams": _round_hparams(hparams),
            "fitness": 0.0,
            "batch_avg_reward": 0.0,
            "batch_win_rate": 0.0,
            "loss": 0.0,
            "games": 0,
            "status": "ready",
            "copied_from": None,
            "mutations": 0,
            "survival_count": 0,
            "best_eval": {},
            "model_hash": _model_hash(shared_network),
        }
        _apply_member_hparams(member)
        members.append(member)

    return members


async def _run_population_member(
    member: dict[str, Any],
    num_sessions: int,
    games_per_session: int,
    start_game_index: int,
) -> tuple[dict[str, float], int]:
    trainer = member["trainer"]
    agents = member["agents"]
    trainer.games_per_session = games_per_session

    total_rewards = 0.0
    total_wins = 0.0
    total_scores = 0.0
    games_played = 0
    policy_losses: list[float] = []
    accumulated_experiences: list = []

    can_batch = (
        trainer.use_rust_engine
        and trainer.batch_concurrency > 1
    )

    for session_offset in range(num_sessions):
        if not _lab.running:
            break

        all_experiences: list = []

        if can_batch:
            # --- Batched path: use trainer's optimised batched session ---
            import time as _time
            batched_exps, batched_stats, sk_exps, sk_stats = (
                await trainer._play_session_batched(
                    session_offset,
                    start_game_index + games_played,
                    _time.time(),
                )
            )
            all_experiences.extend(batched_exps)
            all_experiences.extend(sk_exps)

            for stat in batched_stats + sk_stats:
                raw_score = stat["raw_score"]
                total_scores += raw_score
                total_rewards += raw_score / 100.0
                total_wins += 1.0 if raw_score > 0 else 0.0
                games_played += 1
        else:
            # --- Sequential fallback (no Rust engine) ---
            for game_offset in range(games_per_session):
                if not _lab.running:
                    break

                use_stockskis = (
                    trainer.stockskis_ratio > 0
                    and trainer._stockskis_opponents is not None
                    and trainer._rng.random() < trainer.stockskis_ratio
                )
                use_fsp = (
                    not use_stockskis
                    and trainer.fsp_ratio > 0
                    and trainer.network_bank.is_ready
                    and trainer._rng.random() < trainer.fsp_ratio
                )
                external_opponents = use_stockskis or use_fsp

                original_agents = None
                if use_stockskis:
                    original_agents = trainer._enter_stockskis_mode()
                elif use_fsp:
                    trainer._enter_fsp_mode()

                for agent in agents:
                    agent.set_training(True)
                    agent.clear_experiences()

                game = GameLoop(trainer.agents)
                state, scores = await game.run(dealer=(start_game_index + games_played) % 4)

                if use_stockskis:
                    trainer._exit_stockskis_mode(original_agents)
                elif use_fsp:
                    trainer._exit_fsp_mode()

                raw_score, win = _score_game(state, scores)
                total_scores += raw_score
                total_rewards += raw_score / 100.0
                total_wins += win
                games_played += 1

                for seat, agent in enumerate(agents):
                    reward = scores.get(seat, 0) / 100.0
                    agent.finalize_game(reward)
                    if not external_opponents or seat == 0:
                        all_experiences.extend(agent.experiences)

                if games_played % 4 == 0 or game_offset == games_per_session - 1:
                    await asyncio.sleep(0)

        accumulated_experiences.extend(all_experiences)

        trainer.metrics.session += 1
        if trainer.fsp_ratio > 0 and trainer.metrics.session % trainer.bank_save_interval == 0:
            trainer.network_bank.push(copy.deepcopy(trainer.shared_network.state_dict()))

        await asyncio.sleep(0)

    # Single PPO update on all accumulated experiences (fewer, larger updates)
    if accumulated_experiences:
        loss_info = trainer._ppo_update(accumulated_experiences)
        policy_losses.append(loss_info.get("policy_loss", 0.0))
        member["loss"] = loss_info.get("policy_loss", 0.0)

    batch_games = max(games_played, 1)
    summary = {
        "avg_reward": total_rewards / batch_games,
        "win_rate": total_wins / batch_games,
        "avg_score": total_scores / batch_games,
        "loss": sum(policy_losses) / max(len(policy_losses), 1),
        "games": float(games_played),
    }
    return summary, start_game_index + games_played


async def _evaluate_population_member(member: dict[str, Any], eval_games: int, eval_bots: list[str] | None = None) -> dict[str, float]:
    if eval_bots is None:
        eval_bots = ["v1", "v2", "v3", "v5"]
    network = member["trainer"].shared_network
    results = await _evaluate_vs_selected_bots(network, eval_games, eval_bots)

    avg_reward_norm = max(0.0, min(1.0, (member.get("batch_avg_reward", 0.0) + 1.0) / 2.0))

    # Fitness: weighted by bot strength. Give highest weight to strongest evaluated bot.
    wr_values = [results[v]["win_rate"] for v in eval_bots]
    if len(wr_values) >= 3:
        weights = [0.15, 0.25, 0.45] + [0.45] * (len(wr_values) - 3)
        total_w = sum(weights[:len(wr_values)]) + 0.15
        fitness = sum(w * wr for w, wr in zip(weights, wr_values)) / total_w + 0.15 * avg_reward_norm / total_w
    elif len(wr_values) == 2:
        fitness = 0.35 * wr_values[0] + 0.50 * wr_values[1] + 0.15 * avg_reward_norm
    elif len(wr_values) == 1:
        fitness = 0.85 * wr_values[0] + 0.15 * avg_reward_norm
    else:
        fitness = avg_reward_norm

    out: dict[str, float] = {"fitness": round(fitness, 6)}
    for v, r in results.items():
        out[f"vs_{v}"] = r["win_rate"]
        out[f"avg_score_{v}"] = r["avg_score"]
    return out


def _set_best_member(best_member: dict[str, Any], generation: int) -> None:
    global _lab
    _lab.network = best_member["trainer"].shared_network
    _lab.persona["age"] = generation
    _lab.model_hash = _model_hash(_lab.network)
    _lab.display_name = f"{_lab.persona['first_name']} {_lab.persona['last_name']} (age {_lab.persona['age']}) #{_lab.model_hash}"


async def _background_pbt_eval(
    member_snapshots: list[dict[str, Any]],
    eval_games: int,
    generation: int,
    persona_snapshot: dict,
    eval_bots: list[str] | None = None,
) -> None:
    """Fire-and-forget: evaluate population in background processes, update _lab when done.

    Runs eval in a ProcessPoolExecutor so it uses idle CPU cores without
    blocking the training loop.  Results are written to _lab.eval_history
    and _lab.pbt_generation_history for the dashboard.
    """
    if eval_bots is None:
        eval_bots = ["v1", "v2", "v3", "v5"]
    global _lab
    try:
        pool = _get_eval_pool()
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                pool, _eval_member_in_process,
                snap["state_dict"], snap["hidden_size"], eval_games, eval_bots,
            )
            for snap in member_snapshots
        ]
        results = await asyncio.gather(*futures)

        best_snap = None
        best_fitness = -1.0
        gen_fitnesses: list[float] = []
        gen_best_wr: list[float] = []

        for snap, result in zip(member_snapshots, results):
            avg_reward_norm = max(0.0, min(1.0, (snap["batch_avg_reward"] + 1.0) / 2.0))
            # Dynamic fitness: equal weighting across evaluated bots
            wr_values = [result[f"vs_{v}"]["win_rate"] for v in eval_bots]
            if wr_values:
                wr_avg = sum(wr_values) / len(wr_values)
                fitness = 0.85 * wr_avg + 0.15 * avg_reward_norm
            else:
                fitness = avg_reward_norm
            gen_fitnesses.append(fitness)
            # Track the strongest bot's win rate (last in list = strongest)
            gen_best_wr.append(wr_values[-1] if wr_values else 0.0)
            if fitness > best_fitness:
                best_fitness = fitness
                best_snap = {**snap, **result, "fitness": fitness}

        if not best_snap:
            return

        avg_fitness = sum(gen_fitnesses) / max(len(gen_fitnesses), 1)

        gen_entry: dict[str, Any] = {
            "generation": generation,
            "avg_fitness": round(avg_fitness, 4),
            "min_fitness": round(min(gen_fitnesses), 4),
            "max_fitness": round(max(gen_fitnesses), 4),
            "avg_v3": round(sum(gen_best_wr) / max(len(gen_best_wr), 1), 4),
            "best_index": best_snap["index"],
            "best_label": best_snap["label"],
            "best_batch_reward": round(best_snap.get("batch_avg_reward", 0.0), 4),
        }
        for v in eval_bots:
            gen_entry[f"best_vs_{v}"] = round(best_snap[f"vs_{v}"]["win_rate"], 4)
        _lab.pbt_generation_history.append(gen_entry)

        eval_entry: dict[str, Any] = {
            "step": len(_lab.eval_history),
            "label": f"PBT Gen {generation}",
            "program": "self_play_pbt",
            "loss": best_snap.get("loss", 0.0),
            "experiences": 0,
            "games": _lab.self_play_games,
            "generation": generation,
            "best_fitness": round(best_snap["fitness"], 4),
            "avg_fitness": round(avg_fitness, 4),
        }
        for v in eval_bots:
            eval_entry[f"vs_{v}"] = best_snap[f"vs_{v}"]["win_rate"]
            eval_entry[f"avg_score_{v}"] = best_snap[f"vs_{v}"]["avg_score"]
        _lab.eval_history.append(eval_entry)

        # Save best to HoF
        net = TarokNet(hidden_size=best_snap["hidden_size"])
        net.load_state_dict(best_snap["state_dict"])
        persona_for_hof = dict(persona_snapshot)
        persona_for_hof["age"] = generation
        info = save_to_hof(net, persona_for_hof, list(_lab.eval_history), phase_label=f"pbt-g{generation}")
        _lab.snapshots.append(info)

        # Save sample replay
        await _save_sample_replay(net, generation, {"label": best_snap["label"], "index": best_snap["index"]})

    except Exception:
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Island-model PBT — each island is its own OS process with
# zero IPC.  Coordination via Hall of Fame on the file system.
# ---------------------------------------------------------------------------

_island_processes: list[mp.Process] = []
_island_stop_event: mp.Event | None = None


async def _run_island_pbt_session(
    num_sessions: int,
    games_per_session: int,
    eval_games: int,
    eval_bots: list[str],
    eval_interval: int,
    learning_rate: float,
    population_size: int,
    mutation_scale: float,
    fsp_ratio: float = 0.3,
    time_limit_minutes: float = 5.0,
):
    """Island-model PBT: N fully independent training loops,
    coordinating only via the Hall of Fame directory."""
    global _lab, _island_processes, _island_stop_event

    from tarok.adapters.ai.island_worker import (
        island_worker,
        read_all_island_stats,
        clear_island_stats,
        _default_hparams,
        _load_best_from_hof,
    )

    try:
        _lab.phase = "self_play"
        _lab.active_program = "self_play_pbt"
        _lab.pbt_enabled = True
        _lab.pbt_population_size = population_size

        hidden_size = _lab.hidden_size
        checkpoints_dir = str(CHECKPOINTS_DIR)

        time_limit_secs = time_limit_minutes * 60

        # Seed weights from current lab network (if any)
        seed_state = None
        if _lab.network is not None:
            seed_state = {k: v.cpu().clone() for k, v in _lab.network.state_dict().items()}

        # Clear old stats files
        clear_island_stats(CHECKPOINTS_DIR)

        # Create shared stop event
        ctx = mp.get_context("spawn")
        _island_stop_event = ctx.Event()

        # Base hparams — each island will mutate
        base_hp = _default_hparams(lr=learning_rate)

        # Spawn island processes
        _island_processes = []
        for i in range(population_size):
            p = ctx.Process(
                target=island_worker,
                kwargs=dict(
                    island_id=i,
                    stop_event=_island_stop_event,
                    checkpoints_dir=checkpoints_dir,
                    hidden_size=hidden_size,
                    games_per_session=games_per_session,
                    eval_games=eval_games,
                    eval_interval=eval_interval,
                    seed_state_dict=seed_state,
                    hparams=dict(base_hp),
                    mutation_scale=mutation_scale,
                    fsp_ratio=fsp_ratio,
                    time_limit_seconds=time_limit_secs,
                    eval_bots=eval_bots,
                ),
                daemon=True,
            )
            p.start()
            _island_processes.append(p)

        _lab.pbt_population = [
            {"index": i, "label": f"Island {i}", "name": f"Island {i}", "status": "starting",
             "fitness": 0, "games": 0, "hparams": {},
             "batch_avg_reward": 0, "batch_win_rate": 0,
             **{f"vs_{v}": 0 for v in eval_bots},
             **{f"avg_score_{v}": 0 for v in eval_bots},
             "loss": 0, "copied_from": None, "mutations": 0,
             "survival_count": 0, "model_hash": "",
             "games_per_sec": 0, "generation": 0, "persona": {}}
            for i in range(population_size)
        ]
        _lab.total_training_sessions = round(time_limit_secs)  # seconds

        # Poll island stats files for dashboard updates
        t_start = time.perf_counter()
        while _lab.running:
            elapsed = time.perf_counter() - t_start
            stats = read_all_island_stats(CHECKPOINTS_DIR)
            total_games = sum(s.get("total_games", 0) for s in stats)
            total_gps = sum(s.get("games_per_sec", 0) for s in stats)

            _lab.self_play_games = total_games
            _lab.sp_games_per_second = round(total_gps, 1)
            _lab.training_sessions_done = round(elapsed)  # seconds elapsed
            _lab.total_training_sessions = round(time_limit_secs)  # seconds total

            # Update per-island population display
            for s in stats:
                idx = s.get("island_id", 0)
                if idx < len(_lab.pbt_population):
                    island_hp = s.get("hparams", {})
                    island_name = s.get("name", f"Island {idx}")
                    _lab.pbt_population[idx].update({
                        "status": s.get("status", "running"),
                        "fitness": s.get("best_fitness", 0),
                        "games": s.get("total_games", 0),
                        "batch_avg_reward": s.get("avg_score", 0),
                        "batch_win_rate": s.get("session_win_rate", 0),
                        "loss": s.get("loss", 0),
                        "games_per_sec": s.get("games_per_sec", 0),
                        "generation": s.get("generation", 0),
                        "elapsed": s.get("elapsed", 0),
                        "time_limit": time_limit_secs,
                        "hparams": island_hp,
                        "model_hash": f"island-{idx}",
                        "label": island_name,
                        "name": island_name,
                        "persona": s.get("persona", {}),
                        **{k: s.get(k, 0) for k in s if k.startswith("vs_") or k.startswith("avg_score_")},
                    })

            # Best across all islands
            if stats:
                best = max(stats, key=lambda s: s.get("best_fitness", 0))
                _lab.sp_avg_placement = best.get("win_rate", 0)
                _lab.sp_avg_score = best.get("avg_score", 0)
                _lab.current_loss = best.get("loss", 0)
                _lab.pbt_generation = max(s.get("generation", 0) for s in stats)

            # Time limit
            if time_limit_secs > 0 and elapsed >= time_limit_secs:
                break

            # Check if all processes have died
            alive = [p for p in _island_processes if p.is_alive()]
            if not alive:
                break

            await asyncio.sleep(0.5)

        # Signal stop
        if _island_stop_event is not None:
            _island_stop_event.set()

        # Wait for processes to finish (up to 10 seconds)
        for p in _island_processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        # --- Post-training: run round-robin tournament between HoF models ---
        _lab.phase = "tournament"
        _lab.active_program = "island_tournament"

        from tarok.adapters.ai.island_worker import run_hof_tournament

        def _tournament_progress(matchup, total, name_a, name_b):
            _lab.pbt_member_index = matchup
            _lab.pbt_member_total = total

        tournament_results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_hof_tournament(
                HOF_DIR,
                games_per_matchup=eval_games,
                hidden_size=hidden_size,
                progress_callback=_tournament_progress,
            ),
        )

        # --- Cross-island learning: top 4 models play together, all learn ---
        cross_island_sessions = 0
        if len(tournament_results) >= 2 and _lab.running:
            _lab.phase = "cross_island"
            _lab.active_program = "cross_island_learning"
            top_models = tournament_results[:min(4, len(tournament_results))]

            # Load distinct networks for up to 4 agents
            cross_agents: list[RLAgent] = []
            for i, result in enumerate(top_models):
                pt_path = HOF_DIR / result["filename"]
                if pt_path.exists():
                    try:
                        from tarok.adapters.ai.network import TarokNet as _TN
                        data = torch.load(pt_path, map_location="cpu", weights_only=False)
                        net = _TN(hidden_size=hidden_size)
                        net.load_state_dict(data["model_state_dict"])
                        a = RLAgent(name=result["name"], hidden_size=hidden_size, device="cpu", explore_rate=0.05)
                        a.network = net
                        a.set_training(True)
                        cross_agents.append(a)
                    except Exception:
                        pass

            # Pad to 4 if needed
            while len(cross_agents) < 4:
                a = RLAgent(name=f"Clone-{len(cross_agents)}", hidden_size=hidden_size, device="cpu", explore_rate=0.05)
                if cross_agents:
                    a.network = copy.deepcopy(cross_agents[0].network)
                a.set_training(True)
                cross_agents.append(a)

            # Each agent has its own optimizer — true multi-agent learning
            cross_trainers = []
            for i, agent in enumerate(cross_agents):
                t = PPOTrainer(
                    agents=[agent] + [cross_agents[(i + k) % len(cross_agents)] for k in range(1, 4)],
                    lr=learning_rate,
                    games_per_session=games_per_session,
                    device="cpu",
                    stockskis_ratio=0.0,
                    fsp_ratio=0.0,
                    use_rust_engine=True,
                    save_dir=str(CHECKPOINTS_DIR / "island_tmp" / f"cross_{i}"),
                    batch_concurrency=32,
                )
                cross_trainers.append(t)

            # Run cross-training: agents play together, each learns from their own perspective
            cross_sessions = max(5, eval_interval)  # at least 5 sessions
            _lab.total_training_sessions = cross_sessions
            _lab.training_sessions_done = 0

            for sess in range(cross_sessions):
                if not _lab.running:
                    break

                # Play games with all 4 distinct agents
                game_loop = GameLoop(cross_agents, rng=random.Random(sess))
                state, scores = await game_loop.run(dealer=sess % 4)

                # Each agent collects its own experiences and updates
                for i, agent in enumerate(cross_agents):
                    if agent.experiences:
                        for exp in agent.experiences:
                            exp.reward = float(scores.get(i, 0))
                        loss_info = cross_trainers[i]._ppo_update(agent.experiences)
                        agent.clear_experiences()

                cross_island_sessions = sess + 1
                _lab.training_sessions_done = cross_island_sessions

                if sess % 5 == 4:
                    await asyncio.sleep(0)  # yield control

            # Find best cross-trained model by quick eval
            best_cross_idx = 0
            best_cross_wr = -1.0
            for i, agent in enumerate(cross_agents):
                agent.network.eval()
                result = _evaluate_vs_bots_sync(agent.network, min(eval_games, 20), "v3")
                wr = result if isinstance(result, (int, float)) else result.get("win_rate", 0) if isinstance(result, dict) else 0
                if wr > best_cross_wr:
                    best_cross_wr = wr
                    best_cross_idx = i

            # Save best cross-trained model
            best_cross_net = cross_agents[best_cross_idx].network
            if _lab.network is not None:
                _lab.network.load_state_dict(best_cross_net.state_dict())
            _lab.model_hash = _model_hash(_lab.network) if _lab.network else ""

        # --- Load best model from HoF if no cross-training happened ---
        if cross_island_sessions == 0:
            hof_data = _load_best_from_hof(HOF_DIR)
            if hof_data is not None and _lab.network is not None:
                _lab.network.load_state_dict(hof_data["model_state_dict"])
                _lab.model_hash = _model_hash(_lab.network)

        # --- Build training summary ---
        final_stats = read_all_island_stats(CHECKPOINTS_DIR)
        total_games_all = sum(s.get("total_games", 0) for s in final_stats)
        total_time_all = max(s.get("elapsed", 0) for s in final_stats) if final_stats else 0

        _lab.training_summary = {
            "num_islands": population_size,
            "total_games": total_games_all,
            "total_time_seconds": round(total_time_all, 1),
            "total_time_display": _format_duration(total_time_all),
            "cross_island_sessions": cross_island_sessions,
            "tournament_results": tournament_results[:10],  # top 10
            "champion": tournament_results[0] if tournament_results else None,
            "island_summaries": [
                {
                    "island_id": s.get("island_id", i),
                    "name": s.get("name", f"Island {i}"),
                    "persona": s.get("persona", {}),
                    "generations": s.get("generation", 0),
                    "games": s.get("total_games", 0),
                    "fitness": s.get("best_fitness", 0),
                    "games_per_sec": s.get("games_per_sec", 0),
                    **{k: s.get(k, 0) for k in s if k.startswith("vs_")},
                }
                for i, s in enumerate(final_stats)
            ],
            "eval_bots": eval_bots,
        }

        _lab.phase = "done"

    except Exception as e:
        _lab.phase = "idle"
        _lab.error = str(e)
        import traceback
        traceback.print_exc()
    finally:
        _lab.running = False
        if _island_stop_event is not None:
            _island_stop_event.set()
        for p in _island_processes:
            if p.is_alive():
                p.terminate()
        _island_processes = []
        _island_stop_event = None


async def _run_pbt_self_play_session(
    num_sessions: int,
    games_per_session: int,
    eval_games: int,
    eval_interval: int,
    learning_rate: float,
    stockskis_ratio: float,
    fsp_ratio: float,
    population_size: int,
    exploit_top_ratio: float,
    exploit_bottom_ratio: float,
    mutation_scale: float,
    eval_bots: list[str] | None = None,
):
    global _lab

    try:
        _lab.phase = "self_play"
        _lab.active_program = "self_play_pbt"
        _lab.pbt_enabled = True
        _lab.pbt_population_size = population_size

        batch_sessions = max(1, eval_interval)
        total_generations = max(1, math.ceil(num_sessions / batch_sessions))
        _lab.total_training_sessions = total_generations
        _lab.pbt_total_generations = total_generations

        hidden_size = _lab.network.shared[0].out_features
        # CPU is faster than MPS for this model size (256 hidden dims)
        # due to GPU transfer overhead dominating small batch inference
        device = "cpu"
        _lab.network = _lab.network.to(torch.device(device))

        members = _build_pbt_population(
            population_size=population_size,
            hidden_size=hidden_size,
            device=device,
            seed_network=_lab.network,
            learning_rate=learning_rate,
            stockskis_ratio=stockskis_ratio,
            fsp_ratio=fsp_ratio,
            mutation_scale=mutation_scale,
        )
        _sync_population_state(members)

        sessions_completed = 0
        global_game_index = 0
        rng = random.Random(time.time())
        base_age = _lab.persona.get("age", 0)
        progress_manager = mp.Manager()

        for iteration in range(1, total_generations + 1):
            generation = base_age + iteration
            if not _lab.running:
                break

            _lab.phase = "self_play"
            _lab.pbt_generation = generation
            _lab.training_sessions_done = iteration
            _lab.self_play_sessions = sessions_completed

            sessions_this_generation = min(batch_sessions, num_sessions - sessions_completed)
            if sessions_this_generation <= 0:
                break

            for member in members:
                member["status"] = "training"
            _lab.pbt_member_index = len(members)
            _lab.pbt_member_total = len(members)
            _sync_population_state(members)

            # Parallel training with live progress reporting
            pool = _get_train_pool()
            loop = asyncio.get_event_loop()
            progress_queue = progress_manager.Queue()
            gen_start = time.perf_counter()

            train_futures = [
                loop.run_in_executor(
                    pool,
                    functools.partial(
                        _train_member_in_process,
                        m["trainer"].shared_network.state_dict(),
                        m["trainer"].optimizer.state_dict(),
                        dict(m["hparams"]),
                        hidden_size,
                        sessions_this_generation,
                        games_per_session,
                        global_game_index + m["index"] * sessions_this_generation * games_per_session,
                        member_index=m["index"],
                        progress_queue=progress_queue,
                    ),
                )
                for m in members
            ]

            # Poll progress queue while waiting for all workers to finish
            pending = set(range(len(members)))
            member_games: dict[int, int] = {m["index"]: 0 for m in members}
            member_gps: dict[int, float] = {m["index"]: 0.0 for m in members}
            done_futures: set[int] = set()

            while pending:
                # Drain all available progress messages
                while not progress_queue.empty():
                    try:
                        msg = progress_queue.get_nowait()
                    except Exception:
                        break
                    idx = msg.get("member", 0)
                    if msg["type"] == "session":
                        member_games[idx] = msg["games"]
                        member_gps[idx] = msg["games_per_sec"]
                        _lab.self_play_games = sum(
                            m.get("games", 0) for m in members
                        ) + sum(member_games.values())
                        _lab.sp_games_per_second = sum(member_gps.values())
                        _lab.sp_avg_placement = msg["win_rate"]
                        _lab.sp_avg_score = msg["avg_score"]
                        # Update member status
                        if idx < len(members):
                            members[idx]["status"] = f"training ({msg['session']}/{msg['total_sessions']})"
                        _sync_population_state(members)
                    elif msg["type"] == "ppo_start":
                        if idx < len(members):
                            members[idx]["status"] = "ppo_update"
                        _sync_population_state(members)
                    elif msg["type"] == "done":
                        done_futures.add(idx)

                # Check which futures have completed
                newly_done = set()
                for i, fut in enumerate(train_futures):
                    if i in pending and fut.done():
                        newly_done.add(i)
                pending -= newly_done

                if pending:
                    await asyncio.sleep(0.25)

            train_results = [f.result() for f in train_futures]

            for member, result in zip(members, train_results):
                member["trainer"].shared_network.load_state_dict(result["network_state"])
                member["trainer"].optimizer.load_state_dict(result["optimizer_state"])
                # Sync agents to updated network
                for agent in member["agents"]:
                    agent.network = member["trainer"].shared_network
                member["batch_avg_reward"] = result["avg_reward"]
                member["batch_win_rate"] = result["win_rate"]
                member["loss"] = result["loss"]
                member["games"] += result["games"]
                member["model_hash"] = _model_hash(member["trainer"].shared_network)
                member["status"] = "trained"
                _lab.self_play_games += result["games"]
                global_game_index += result["games"]
            _sync_population_state(members)

            sessions_completed += sessions_this_generation
            _lab.self_play_sessions = sessions_completed

            # Rank by training metrics (instant, no eval needed)
            ranked = sorted(members, key=lambda m: m.get("batch_avg_reward", 0.0), reverse=True)
            best_member = ranked[0]
            for member in members:
                member["fitness"] = max(0.0, min(1.0, (member.get("batch_avg_reward", 0.0) + 1.0) / 2.0))
                member["status"] = "ranked"
            _sync_population_state(members)

            _set_best_member(best_member, generation)
            _lab.current_loss = float(best_member.get("loss", 0.0))

            # Snapshot weights and fire-and-forget eval in background processes
            member_snapshots = [
                {
                    "state_dict": {k: v.clone() for k, v in m["trainer"].shared_network.state_dict().items()},
                    "hidden_size": m["trainer"].shared_network.shared[0].out_features,
                    "index": m["index"],
                    "label": m["label"],
                    "batch_avg_reward": m.get("batch_avg_reward", 0.0),
                    "loss": m.get("loss", 0.0),
                }
                for m in members
            ]
            asyncio.create_task(_background_pbt_eval(
                member_snapshots, eval_games, generation, dict(_lab.persona),
                eval_bots=eval_bots,
            ))

            elite_count = max(1, int(round(len(members) * exploit_top_ratio)))
            replace_count = max(1, int(round(len(members) * exploit_bottom_ratio)))
            elites = ranked[:elite_count]
            laggards = ranked[-replace_count:]

            for member in members:
                member["survival_count"] += 1
                member["copied_from"] = None

            if iteration < total_generations:
                _lab.phase = "exploiting"
                for laggard_idx, laggard in enumerate(laggards):
                    donor = elites[laggard_idx % len(elites)]
                    if donor["index"] == laggard["index"]:
                        continue
                    snapshot = _snapshot_member(donor)
                    _restore_member_from_snapshot(laggard, snapshot)
                    laggard["hparams"] = _mutate_hparams(laggard["hparams"], rng, mutation_scale)
                    _apply_member_hparams(laggard)
                    laggard["copied_from"] = donor["index"]
                    laggard["mutations"] += 1
                    laggard["survival_count"] = 0
                    laggard["status"] = "mutated"
                    laggard["model_hash"] = _model_hash(laggard["trainer"].shared_network)
                    _lab.pbt_events.append({
                        "generation": generation,
                        "target": laggard["index"],
                        "source": donor["index"],
                        "hparams": _round_hparams(laggard["hparams"]),
                    })

                _lab.pbt_events = _lab.pbt_events[-40:]

            _sync_population_state(members)
            await asyncio.sleep(0)

        _lab.phase = "done"
        _lab.training_sessions_done = min(total_generations, _lab.pbt_generation)

    except Exception as e:
        _lab.phase = "idle"
        _lab.error = str(e)
        import traceback
        traceback.print_exc()
    finally:
        _lab.running = False
        _lab.pbt_member_index = 0
        _lab.pbt_member_total = 0
        if _lab.network is not None:
            _lab.network = _lab.network.to("cpu")
        try:
            progress_manager.shutdown()
        except Exception:
            pass


async def _run_self_play_session(
    num_sessions: int,
    games_per_session: int,
    eval_games: int,
    eval_bots: list[str],
    eval_interval: int,
    learning_rate: float,
    stockskis_ratio: float,
    fsp_ratio: float,
    hof_ratio: float = 0.0,
):
    """Self-play PPO training pipeline.

    Delegates game-playing to the PPOTrainer which handles all opponent types
    (self-play, StockSkis, HoF, FSP) and records per-opponent stats.
    Training game stats are emitted to eval_history each session for the chart.
    Detailed evaluation should be done via tournaments.
    """
    global _lab

    try:
        _lab.phase = "self_play"
        _lab.active_program = "self_play"
        _lab.total_training_sessions = num_sessions

        # Create 4 agents sharing the lab network
        hidden_size = _lab.network.shared[0].out_features
        agents = []
        # CPU is faster than MPS for this model size (256 hidden dims)
        # due to GPU transfer overhead per batch step
        dev = "cpu"
        _lab.network = _lab.network.to(torch.device(dev))
        for i in range(4):
            agent = RLAgent(name=f"Lab-{i}", hidden_size=hidden_size, device=dev, explore_rate=0.1)
            agent.network = _lab.network
            agents.append(agent)

        trainer = PPOTrainer(
            agents=agents,
            lr=learning_rate,
            games_per_session=games_per_session,
            device=dev,
            stockskis_ratio=stockskis_ratio,
            stockskis_strength=1.0,
            fsp_ratio=fsp_ratio,
            hof_ratio=hof_ratio,
            hof_dir=str(HOF_DIR),
            bank_size=20,
            use_rust_engine=True,
            lr_schedule="cosine",
            entropy_schedule="linear",
            entropy_coef_end=0.002,
            value_clip=0.2,
            # v2: oracle guiding distillation
            oracle_guiding_coef=0.1 if _lab.network.oracle_critic_enabled else 0.0,
        )

        trainer._running = True
        game_count = 0
        hof_interval = max(eval_interval, 1)

        # Rolling contract stats (cumulative across sessions)
        rolling_contract_stats: dict[str, ContractStats] = {
            c: ContractStats() for c in _TRACKED_CONTRACTS
        }
        # Tarok count distribution (cumulative)
        tarok_count_bids: dict[int, dict[str, int]] = {i: {} for i in range(13)}

        for session_idx in range(num_sessions):
            if not _lab.running:
                break

            # Update LR and entropy schedules
            trainer._update_schedule(session_idx, num_sessions)

            _lab.phase = "self_play"
            _lab.training_sessions_done = session_idx + 1
            _lab.self_play_sessions = session_idx + 1
            await asyncio.sleep(0)

            session_start = time.time()

            # Begin per-session opponent tracking
            trainer.opponent_pool.begin_session()

            # --- Delegate game-playing to trainer ---
            batched_exps, batched_stats, sk_exps, sk_stats = (
                await trainer._play_session_batched(
                    session_idx, game_count, session_start,
                )
            )

            all_experiences = list(batched_exps) + list(sk_exps)
            all_stats = list(batched_stats) + list(sk_stats)

            # Process stats for lab metrics
            session_scores: list[int] = []
            session_wins = 0
            session_cumulative_scores: list[int] = [0, 0, 0, 0]
            session_bids = 0
            session_klops = 0
            session_solos = 0

            for result in all_stats:
                raw_score = result["raw_score"]
                is_klop = result["is_klop"]
                is_solo = result["is_solo"]
                declarer_p0 = result["declarer_p0"]
                agent0_bids = result["agent0_bids"]
                game_mode = result.get("game_mode", "batch")

                session_scores.append(raw_score)
                game_count += 1
                _lab.self_play_games += 1

                # Win detection
                if is_klop:
                    won = raw_score > 0
                elif declarer_p0:
                    won = raw_score > 0
                else:
                    won = result.get("declarer_lost", False)

                if won:
                    session_wins += 1
                if is_klop:
                    session_klops += 1
                if declarer_p0:
                    session_bids += 1
                if is_solo and declarer_p0:
                    session_solos += 1

                # Accumulate all 4 players' scores for session-level placement
                all_sc = result.get("all_scores", {})
                for pid in range(4):
                    session_cumulative_scores[pid] += all_sc.get(pid, 0)
                trainer._record_opponent_result(game_mode, OpponentGameResult(
                    raw_score=raw_score,
                    won=won,
                    contract_name=result.get("contract_name", "klop"),
                    place=0.0,  # placeholder; real placement computed at session end
                    declarer_p0=declarer_p0,
                    bid=bool(agent0_bids),
                ))

                # Track contract stats (declarer vs defender)
                contract_name = result.get("contract_name", "klop")
                if contract_name in rolling_contract_stats:
                    cs = rolling_contract_stats[contract_name]
                    if declarer_p0:
                        cs.decl_played += 1
                        if raw_score > 0:
                            cs.decl_won += 1
                        cs.decl_total_score += raw_score
                    else:
                        cs.def_played += 1
                        if won:
                            cs.def_won += 1
                        cs.def_total_score += raw_score

                # Track tarok count vs contract
                if result.get("initial_tarok_counts"):
                    tarok_counts = result["initial_tarok_counts"]
                    tarok_count = tarok_counts[0] if isinstance(tarok_counts, (list, tuple)) else tarok_counts.get(0, 0)
                    bucket = tarok_count_bids.setdefault(tarok_count, {})
                    bucket[contract_name] = bucket.get(contract_name, 0) + 1

            await asyncio.sleep(0)

            if not _lab.running or not all_experiences:
                break

            # PPO update
            info = trainer._ppo_update(all_experiences)
            _lab.current_loss = info.get("policy_loss", 0)

            # Update per-session metrics
            n = max(len(session_scores), 1)
            elapsed = max(time.time() - session_start, 0.001)
            # Session-level placement from cumulative scores
            sorted_session = sorted(range(4), key=lambda pid: session_cumulative_scores[pid], reverse=True)
            session_placement = float(sorted_session.index(0) + 1)
            _lab.sp_avg_placement = session_placement
            _lab.sp_avg_reward = sum(s / 100.0 for s in session_scores) / n
            _lab.sp_avg_score = sum(session_scores) / n
            _lab.sp_bid_rate = session_bids / n
            _lab.sp_klop_rate = session_klops / n
            _lab.sp_solo_rate = session_solos / n
            _lab.sp_games_per_second = len(session_scores) / elapsed

            _lab.sp_avg_placement_history.append(round(_lab.sp_avg_placement, 4))
            _lab.sp_avg_reward_history.append(round(_lab.sp_avg_reward, 4))
            _lab.sp_avg_score_history.append(round(_lab.sp_avg_score, 2))
            _lab.sp_loss_history.append(round(_lab.current_loss, 4))
            _lab.sp_bid_rate_history.append(round(_lab.sp_bid_rate, 4))
            _lab.sp_klop_rate_history.append(round(_lab.sp_klop_rate, 4))
            _lab.sp_solo_rate_history.append(round(_lab.sp_solo_rate, 4))
            _lab.sp_min_score_history.append(float(min(session_scores)) if session_scores else 0.0)
            _lab.sp_max_score_history.append(float(max(session_scores)) if session_scores else 0.0)

            # Update contract stats and tarok count distribution
            _lab.contract_stats = {k: v.to_dict() for k, v in rolling_contract_stats.items()}
            _lab.tarok_count_bids = {str(k): v for k, v in tarok_count_bids.items()}

            # Build eval_history entry from training game stats (per-opponent)
            opp_session = trainer.opponent_pool.session_stats_dict()
            entry: dict = {
                "step": len(_lab.eval_history),
                "label": f"Session {session_idx + 1}",
                "program": "self_play",
                "loss": _lab.current_loss,
                "games": _lab.self_play_games,
            }
            for opp_name, opp_stats in opp_session.items():
                entry[f"vs_{opp_name}"] = opp_stats["win_rate"]
                entry[f"avg_score_{opp_name}"] = opp_stats["avg_score"]
                entry[f"avg_place_{opp_name}"] = opp_stats.get("avg_place", 4.0)
            _lab.eval_history.append(entry)

            # Update aggregate opponent pool stats for dashboard
            _lab.opponent_stats = trainer.opponent_pool.stats_dict()

            # Push current weights to network bank for FSP
            if fsp_ratio > 0:
                trainer.network_bank.push(_lab.network.state_dict())

            _lab.persona["age"] = _lab.persona.get("age", 0) + 1

            # Periodic HoF save
            if (session_idx + 1) % hof_interval == 0 or session_idx == num_sessions - 1:
                _update_persona_hash()
                snap_info = save_to_hof(
                    _lab.network, _lab.persona, _lab.eval_history,
                    phase_label=f"selfplay-s{session_idx + 1}",
                )
                _lab.snapshots.append(snap_info)

        _lab.phase = "done"
        _lab.training_sessions_done = num_sessions

    except Exception as e:
        _lab.phase = "idle"
        _lab.error = str(e)
        import traceback
        traceback.print_exc()
    finally:
        _lab.running = False
        # Move network back to CPU for serving / checkpoint saving
        if _lab.network is not None:
            _lab.network = _lab.network.to("cpu")


def _on_training_progress(info: dict):
    """Callback from imitation_pretrain progress updates."""
    global _lab
    _lab.current_loss = info.get("policy_loss", 0)


def _update_persona_hash():
    """Update model hash and display name after weights change."""
    global _lab
    if _lab.network and _lab.persona:
        _lab.model_hash = _model_hash(_lab.network)
        _lab.display_name = _display_name(_lab.persona, _lab.model_hash)


def get_lab_state() -> dict:
    """Return current lab state as a dict for the API."""
    return {
        "phase": _lab.phase,
        "has_network": _lab.network is not None,
        "hidden_size": _lab.hidden_size,
        "persona": _lab.persona,
        "model_hash": _lab.model_hash,
        "display_name": _lab.display_name,
        "active_program": _lab.active_program,
        "eval_history": _lab.eval_history[-500:],
        "training_sessions_done": _lab.training_sessions_done,
        "total_training_sessions": _lab.total_training_sessions,
        "current_loss": _lab.current_loss,
        "expert_games_generated": _lab.expert_games_generated,
        "expert_experiences": _lab.expert_experiences,
        "self_play_games": _lab.self_play_games,
        "self_play_sessions": _lab.self_play_sessions,
        "running": _lab.running,
        "error": _lab.error,
        "snapshots": _lab.snapshots,
        # Per-session self-play metrics
        "sp_avg_placement": _lab.sp_avg_placement,
        "sp_avg_reward": _lab.sp_avg_reward,
        "sp_avg_score": _lab.sp_avg_score,
        "sp_bid_rate": _lab.sp_bid_rate,
        "sp_klop_rate": _lab.sp_klop_rate,
        "sp_solo_rate": _lab.sp_solo_rate,
        "sp_games_per_second": _lab.sp_games_per_second,
        "sp_avg_placement_history": _lab.sp_avg_placement_history[-500:],
        "sp_avg_reward_history": _lab.sp_avg_reward_history[-500:],
        "sp_avg_score_history": _lab.sp_avg_score_history[-500:],
        "sp_loss_history": _lab.sp_loss_history[-500:],
        "sp_bid_rate_history": _lab.sp_bid_rate_history[-500:],
        "sp_klop_rate_history": _lab.sp_klop_rate_history[-500:],
        "sp_solo_rate_history": _lab.sp_solo_rate_history[-500:],
        "sp_min_score_history": _lab.sp_min_score_history[-500:],
        "sp_max_score_history": _lab.sp_max_score_history[-500:],
        # PBT
        "pbt_enabled": _lab.pbt_enabled,
        "pbt_generation": _lab.pbt_generation,
        "pbt_total_generations": _lab.pbt_total_generations,
        "pbt_population_size": _lab.pbt_population_size,
        "pbt_member_index": _lab.pbt_member_index,
        "pbt_member_total": _lab.pbt_member_total,
        "population": _lab.pbt_population,
        "generation_history": _lab.pbt_generation_history[-500:],
        "population_events": _lab.pbt_events[-500:],
        "training_summary": _lab.training_summary,
        "opponent_stats": _lab.opponent_stats,
        "contract_stats": _lab.contract_stats,
        "tarok_count_bids": _lab.tarok_count_bids,
    }


def reset_lab():
    """Reset lab state."""
    global _lab, _lab_task
    if _lab_task and not _lab_task.done():
        _lab_task.cancel()
    _lab = LabState()
    _lab_task = None


def create_lab_network(hidden_size: int = 256):
    """Create a fresh network with a generated persona and store it in lab state."""
    global _lab
    _lab.network = _create_network(hidden_size)
    _lab.hidden_size = hidden_size
    _lab.loaded_from_checkpoint = False
    _lab.persona = _generate_persona()
    _lab.model_hash = _model_hash(_lab.network)
    _lab.display_name = _display_name(_lab.persona, _lab.model_hash)
    _lab.eval_history = []
    _lab.expert_games_generated = 0
    _lab.expert_experiences = 0
    _lab.self_play_games = 0
    _lab.self_play_sessions = 0
    _lab.sp_avg_placement = 0.0
    _lab.sp_avg_reward = 0.0
    _lab.sp_avg_score = 0.0
    _lab.sp_bid_rate = 0.0
    _lab.sp_klop_rate = 0.0
    _lab.sp_solo_rate = 0.0
    _lab.sp_games_per_second = 0.0
    _lab.sp_avg_placement_history = []
    _lab.sp_avg_reward_history = []
    _lab.sp_avg_score_history = []
    _lab.sp_loss_history = []
    _lab.sp_bid_rate_history = []
    _lab.sp_klop_rate_history = []
    _lab.sp_solo_rate_history = []
    _lab.sp_min_score_history = []
    _lab.sp_max_score_history = []
    _lab.training_sessions_done = 0
    _lab.current_loss = 0
    _lab.error = None
    _lab.phase = "idle"
    _lab.active_program = ""
    _lab.snapshots = []
    _lab.pbt_enabled = False
    _lab.pbt_generation = 0
    _lab.pbt_total_generations = 0
    _lab.pbt_population_size = 0
    _lab.pbt_member_index = 0
    _lab.pbt_member_total = 0
    _lab.pbt_population = []
    _lab.pbt_generation_history = []
    _lab.pbt_events = []


def load_lab_checkpoint(choice: str) -> dict:
    """Load a checkpoint into the training lab state."""
    global _lab

    path = _resolve_checkpoint_path(choice)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    hidden_size = checkpoint.get("hidden_size") or state_dict["shared.0.weight"].shape[0]
    oracle_critic = any(key.startswith("critic_backbone.") for key in state_dict)

    network = TarokNet(hidden_size=hidden_size, oracle_critic=oracle_critic)
    network.load_state_dict(state_dict)

    persona = dict(checkpoint.get("persona") or _generate_persona())
    persona.setdefault("age", 0)

    _lab.network = network
    _lab.hidden_size = hidden_size
    _lab.loaded_from_checkpoint = True
    _lab.persona = persona
    _lab.eval_history = checkpoint.get("eval_history", [])
    _lab.training_step = 0
    _lab.total_training_steps = 0
    _lab.current_loss = 0
    _lab.expert_games_generated = 0
    _lab.expert_experiences = 0
    _lab.training_sessions_done = 0
    _lab.total_training_sessions = 0
    _lab.self_play_games = 0
    _lab.self_play_sessions = 0
    _lab.active_program = ""
    _lab.running = False
    _lab.error = None
    _lab.phase = "idle"
    _lab.snapshots = []
    _lab.pbt_enabled = False
    _lab.pbt_generation = 0
    _lab.pbt_total_generations = 0
    _lab.pbt_population_size = 0
    _lab.pbt_member_index = 0
    _lab.pbt_member_total = 0
    _lab.pbt_population = []
    _lab.pbt_generation_history = []
    _lab.pbt_events = []

    _lab.model_hash = checkpoint.get("model_hash") or _model_hash(_lab.network)
    _lab.display_name = (
        checkpoint.get("display_name")
        or checkpoint.get("model_name")
        or _display_name(_lab.persona, _lab.model_hash)
    )

    return {
        "filename": str(path.relative_to(CHECKPOINTS_DIR.resolve())),
        "display_name": _lab.display_name,
        "hidden_size": _lab.hidden_size,
    }


async def start_lab_training(
    expert_games: int = 500_000,
    expert_source: str = "v2v3v5",
    eval_bots: list[str] | None = None,
    training_epochs: int = 3,
    eval_games: int = 500,
    num_rounds: int = 10,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    chunk_size: int = 50_000,
):
    """Start the imitation learning pipeline."""
    global _lab, _lab_task

    if eval_bots is None:
        eval_bots = ["v1", "v2", "v3", "v5"]

    if _lab.network is None:
        create_lab_network()

    _lab.running = True
    _lab.error = None
    _lab.phase = "evaluating"

    _lab_task = asyncio.create_task(
        _run_lab_session(
            expert_games=expert_games,
            expert_source=expert_source,
            eval_bots=eval_bots,
            training_epochs=training_epochs,
            eval_games=eval_games,
            num_rounds=num_rounds,
            batch_size=batch_size,
            learning_rate=learning_rate,
            chunk_size=chunk_size,
        )
    )


async def start_self_play(
    num_sessions: int = 50,
    games_per_session: int = 20,
    eval_games: int = 100,
    eval_bots: list[str] | None = None,
    eval_interval: int = 5,
    learning_rate: float = 3e-4,
    stockskis_ratio: float = 0.0,
    fsp_ratio: float = 0.3,
    hof_ratio: float = 0.0,
    pbt_enabled: bool = False,
    population_size: int = 4,
    exploit_top_ratio: float = 0.25,
    exploit_bottom_ratio: float = 0.25,
    mutation_scale: float = 1.0,
    time_limit_minutes: float = 5.0,
):
    """Start the self-play PPO training pipeline."""
    global _lab, _lab_task

    if eval_bots is None:
        eval_bots = ["v1", "v2", "v3", "v5"]

    if _lab.network is None:
        create_lab_network()

    _lab.running = True
    _lab.error = None
    _lab.pbt_enabled = pbt_enabled
    _lab.pbt_generation = 0
    _lab.pbt_total_generations = 0
    _lab.pbt_population_size = population_size if _lab.pbt_enabled else 0
    _lab.pbt_member_index = 0
    _lab.pbt_member_total = 0
    _lab.pbt_population = []
    _lab.pbt_generation_history = []
    _lab.pbt_events = []

    if _lab.pbt_enabled:
        _lab_task = asyncio.create_task(
            _run_island_pbt_session(
                num_sessions=num_sessions,
                games_per_session=games_per_session,
                eval_games=eval_games,
                eval_bots=eval_bots,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                population_size=population_size,
                mutation_scale=mutation_scale,
                fsp_ratio=fsp_ratio,
                time_limit_minutes=time_limit_minutes,
            )
        )
    else:
        _lab_task = asyncio.create_task(
            _run_self_play_session(
                num_sessions=num_sessions,
                games_per_session=games_per_session,
                eval_games=eval_games,
                eval_bots=eval_bots,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                stockskis_ratio=stockskis_ratio,
                fsp_ratio=fsp_ratio,
                hof_ratio=hof_ratio,
            )
        )


def stop_lab():
    """Stop training."""
    global _lab, _lab_task, _island_stop_event, _island_processes
    _lab.running = False
    if _island_stop_event is not None:
        _island_stop_event.set()
    if _lab_task and not _lab_task.done():
        _lab_task.cancel()
    _lab.phase = "idle"
