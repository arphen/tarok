"""PPO Trainer — session-based self-play training with multi-decision heads.

Training is organized into *sessions* of N games (default 20).  The agent
learns to maximise cumulative score across a session, which naturally teaches
strategic bidding: when to pass (berac/klop), when to bid conservatively
(tri/dva), and when to go for high-value solo contracts.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tarok.adapters.ai.agent import RLAgent, Experience
from tarok.adapters.ai.encoding import (
    DecisionType,
    BID_ACTION_SIZE,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
    CARD_ACTION_SIZE,
    ANNOUNCE_ACTION_SIZE,
)
from tarok.adapters.ai.network import TarokNet
from tarok.adapters.ai.network_bank import NetworkBank
from tarok.adapters.ai.stockskis_player import StockSkisPlayer
from tarok.use_cases.game_loop import GameLoop

try:
    from tarok.adapters.ai.rust_game_loop import RustGameLoop
    _HAS_RUST = True
except Exception:
    _HAS_RUST = False


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


# Contract names we individually track
_TRACKED_CONTRACTS = ["klop", "three", "two", "one", "solo_three", "solo_two", "solo_one", "solo", "berac"]


@dataclass
class TrainingMetrics:
    episode: int = 0
    total_episodes: int = 0
    session: int = 0
    total_sessions: int = 0
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
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
    win_rate_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    bid_rate_history: list[float] = field(default_factory=list)
    klop_rate_history: list[float] = field(default_factory=list)
    solo_rate_history: list[float] = field(default_factory=list)
    contract_win_rate_history: dict[str, list[float]] = field(
        default_factory=lambda: {c: [] for c in _TRACKED_CONTRACTS}
    )
    session_avg_score_history: list[float] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    # Avg finishing place of StockŠkis opponents per session (1=best, 4=worst)
    stockskis_place_history: list[float] = field(default_factory=list)
    # Tarok count vs bid: for each tarok count (0..12), track which contract was played
    tarok_count_bids: dict[int, dict[str, int]] = field(
        default_factory=lambda: {i: {} for i in range(13)}
    )

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "total_episodes": self.total_episodes,
            "session": self.session,
            "total_sessions": self.total_sessions,
            "avg_reward": round(self.avg_reward, 2),
            "avg_loss": round(self.avg_loss, 4),
            "win_rate": round(self.win_rate, 4),
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
            "win_rate_history": self.win_rate_history[-500:],
            "loss_history": self.loss_history[-500:],
            "bid_rate_history": self.bid_rate_history[-500:],
            "klop_rate_history": self.klop_rate_history[-500:],
            "solo_rate_history": self.solo_rate_history[-500:],
            "contract_win_rate_history": {
                k: v[-500:] for k, v in self.contract_win_rate_history.items()
            },
            "session_avg_score_history": self.session_avg_score_history[-500:],
            "stockskis_place_history": self.stockskis_place_history[-500:],
            "snapshots": self.snapshots,
            "tarok_count_bids": {
                str(k): v for k, v in self.tarok_count_bids.items()
            },
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
        # --- Rust engine ---
        use_rust_engine: bool = False,
        warmup_games: int = 0,
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
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # All agents share the same network for self-play
        self.shared_network = agents[0].network
        self.optimizer = optim.Adam(self.shared_network.parameters(), lr=lr)

        # Sync all agents to use the same network
        for agent in agents:
            agent.network = self.shared_network

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
            ).to(self.device)

        # --- StockŠkis opponents ---
        self.stockskis_ratio = stockskis_ratio
        self._stockskis_opponents: list[StockSkisPlayer] | None = None
        if stockskis_ratio > 0:
            self._stockskis_opponents = [
                StockSkisPlayer(name=f"StockŠkis-{i}", strength=stockskis_strength)
                for i in range(3)
            ]

        self.metrics = TrainingMetrics()
        self._running = False
        self._metrics_callback: list = []
        self._rng = random.Random()

        # --- Rust engine ---
        self.use_rust_engine = use_rust_engine and _HAS_RUST
        self.warmup_games = warmup_games

    def add_metrics_callback(self, callback) -> None:
        self._metrics_callback.append(callback)

    def _enter_stockskis_mode(self) -> list:
        """Replace opponents (agents 1-3) with StockŠkis heuristic bots.

        Returns the original agents list so it can be restored later.
        """
        assert self._stockskis_opponents is not None
        original = list(self.agents)
        for i in range(1, 4):
            self.agents[i] = self._stockskis_opponents[i - 1]  # type: ignore[assignment]
        return original

    def _exit_stockskis_mode(self, original_agents: list) -> None:
        """Restore the original agents after a StockŠkis game."""
        for i in range(4):
            self.agents[i] = original_agents[i]

    def _enter_fsp_mode(self) -> None:
        """Switch opponents (agents 1-3) to historical weights for FSP."""
        snap = self.network_bank.sample()
        if snap is None or self.opponent_network is None:
            return
        self.opponent_network.load_state_dict(snap)
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

        recent_rewards: list[float] = []
        recent_wins: list[float] = []
        recent_bids: list[float] = []
        recent_klops: list[float] = []
        recent_solos: list[float] = []
        # Per-game records for rolling contract stats: (contract_name, declarer_p0, raw_score)
        recent_games: list[tuple[str, bool, int]] = []
        start_time = time.time()
        game_count = 0
        # Running sums for O(1) rolling-window metrics
        _rsum = 0.0; _wsum = 0.0; _bsum = 0.0; _ksum = 0.0; _ssum = 0.0

        for agent in self.agents:
            agent.set_training(True)

        snapshot_interval = max(1, num_sessions // 10)

        for session_idx in range(num_sessions):
            if not self._running:
                break

            self.metrics.session = session_idx + 1
            all_experiences: list[Experience] = []
            session_scores: list[int] = []
            session_stockskis_places: list[float] = []

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
                # FSP mode: use historical network weights (only if not StockŠkis)
                use_fsp = (
                    not use_stockskis
                    and self.fsp_ratio > 0
                    and self.network_bank.is_ready
                    and self._rng.random() < self.fsp_ratio
                )
                external_opponents = use_fsp or use_stockskis

                original_agents = None
                if use_stockskis:
                    original_agents = self._enter_stockskis_mode()
                elif use_fsp:
                    self._enter_fsp_mode()

                # Clear experiences from previous game
                for agent in self.agents:
                    agent.clear_experiences()

                # Play one game (Rust engine or Python engine)
                if self.use_rust_engine and not use_stockskis:
                    game = RustGameLoop(self.agents)
                else:
                    game = GameLoop(self.agents)
                state, scores = await game.run(dealer=(game_count + g) % 4)

                # Restore agents after external-opponent game
                if use_stockskis:
                    self._exit_stockskis_mode(original_agents)
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
                r = raw_score / 100.0
                w = 1.0 if raw_score > 0 else 0.0
                b = 1.0 if agent0_bids else 0.0
                k = 1.0 if is_klop else 0.0
                s = 1.0 if is_solo else 0.0
                recent_rewards.append(r); recent_wins.append(w)
                recent_bids.append(b); recent_klops.append(k); recent_solos.append(s)
                recent_games.append((contract_name, declarer_p0, raw_score))
                _rsum += r; _wsum += w; _bsum += b; _ksum += k; _ssum += s

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
                        if raw_score > 0:
                            cs.def_won += 1
                        cs.def_total_score += raw_score

                # Evict oldest entry once we exceed the window
                window = self.games_per_session * 10
                if len(recent_rewards) > window:
                    _rsum -= recent_rewards[-window - 1]
                    _wsum -= recent_wins[-window - 1]
                    _bsum -= recent_bids[-window - 1]
                    _ksum -= recent_klops[-window - 1]
                    _ssum -= recent_solos[-window - 1]
                    # Also evict from rolling contract stats
                    old_cn, old_decl, old_score = recent_games[-window - 1]
                    if old_cn in self.metrics.contract_stats:
                        old_cs = self.metrics.contract_stats[old_cn]
                        if old_decl:
                            old_cs.decl_played -= 1
                            if old_score > 0:
                                old_cs.decl_won -= 1
                            old_cs.decl_total_score -= old_score
                        else:
                            old_cs.def_played -= 1
                            if old_score > 0:
                                old_cs.def_won -= 1
                            old_cs.def_total_score -= old_score

                # Update live metrics (O(1) — no recomputation)
                self.metrics.episode = game_count
                n = min(len(recent_rewards), window)
                self.metrics.avg_reward = _rsum / n
                self.metrics.win_rate = _wsum / n
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

            # Per-session average score
            if session_scores:
                self.metrics.session_avg_score_history.append(
                    round(sum(session_scores) / len(session_scores), 2)
                )

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

            # --- Append per-session history for charts ---
            self.metrics.reward_history.append(self.metrics.avg_reward)
            self.metrics.win_rate_history.append(self.metrics.win_rate)
            self.metrics.loss_history.append(self.metrics.avg_loss)
            self.metrics.bid_rate_history.append(self.metrics.bid_rate)
            self.metrics.klop_rate_history.append(self.metrics.klop_rate)
            self.metrics.solo_rate_history.append(self.metrics.solo_rate)

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
            device=str(self.device),
            include_oracle=self.shared_network.oracle_critic_enabled,
            progress_callback=sync_progress,
        )

        # Save a warmup checkpoint
        self._save_checkpoint(0, is_snapshot=True, custom_name="tarok_agent_warmup.pt")

    def _ppo_update(self, experiences: list[Experience]) -> dict[str, float]:
        """Perform PPO update on collected experiences, grouped by decision type.

        Uses Generalized Advantage Estimation (GAE) computed per-game in
        temporal order *before* grouping by decision type.
        """
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

            states = torch.stack([e.state for e in dt_exps]).to(self.device)
            actions = torch.tensor([e.action for e in dt_exps], dtype=torch.long).to(self.device)
            old_log_probs = torch.stack([e.log_prob for e in dt_exps]).to(self.device)

            # Oracle states for critic (if PTIE is active)
            has_oracle = dt_exps[0].oracle_state is not None
            oracle_states = (
                torch.stack([e.oracle_state for e in dt_exps]).to(self.device)
                if has_oracle else None
            )

            # Use pre-computed GAE advantages and returns
            advantages = torch.tensor(
                [gae_advantages[id(e)] for e in dt_exps], dtype=torch.float32
            ).to(self.device)
            returns = torch.tensor(
                [gae_returns[id(e)] for e in dt_exps], dtype=torch.float32
            ).to(self.device)

            # Normalise advantages
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Legal masks (all-ones placeholder — actions were legal at play time)
            legal_masks = torch.ones(len(dt_exps), action_size, dtype=torch.float32).to(self.device)

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

                    value_loss = nn.functional.mse_loss(new_values, b_returns)

                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy.mean()
                    )

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
            "episode": episode,
            "session": self.metrics.session,
            "win_rate": round(self.metrics.win_rate, 4),
            "avg_reward": round(self.metrics.avg_reward, 2),
            "games_per_second": round(self.metrics.games_per_second, 1),
        }
        return info

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.shared_network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def stop(self) -> None:
        self._running = False
