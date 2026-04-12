"""RunPPOTraining — async producer-consumer PPO training loop.

Producer thread: runs RustBatchGameRunner continuously with a frozen
network copy, pushing experiences into a thread-safe ExperienceBuffer.

Consumer thread: waits for enough experiences, computes GAE, groups by
DecisionType, runs PPO update epochs, updates canonical weights, signals
producer to refresh its frozen copy.

When num_producers > 1 (default), spawns separate worker **processes** so
each gets its own GIL.  All producers infer on CPU; the consumer does PPO
gradient updates on GPU (MPS / CUDA).
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
import time

import torch
import torch.nn as nn
import torch.optim as optim

from training_lab.entities.config import TrainingConfig
from training_lab.entities.encoding import DecisionType, ACTION_SIZES, CARD_ACTION_SIZE
from training_lab.entities.experience import Experience, ExperienceBatch
from training_lab.entities.experience_buffer import ExperienceBuffer
from training_lab.entities.metrics import SessionMetrics, TrainingProgress
from training_lab.entities.network import TarokNet
from training_lab.infra.mp_producer import MultiProcessProducer, auto_num_producers
from training_lab.ports.checkpoint_store import CheckpointStorePort
from training_lab.ports.compute_backend import ComputeBackendPort
from training_lab.ports.game_simulator import GameSimulatorPort
from training_lab.ports.metrics_sink import MetricsSinkPort
from training_lab.ports.progress import ProgressPort

log = logging.getLogger(__name__)


class RunPPOTraining:
    """Async producer-consumer PPO training.

    Decouples self-play from gradient updates for better GPU utilization.
    """

    def __init__(
        self,
        simulator: GameSimulatorPort,
        compute: ComputeBackendPort,
        store: CheckpointStorePort,
        config: TrainingConfig,
        metrics_sink: MetricsSinkPort | None = None,
        progress: ProgressPort | None = None,
        resume_state_dict: dict | None = None,
    ):
        self.simulator = simulator
        self.compute = compute
        self.store = store
        self.config = config
        self.metrics_sink = metrics_sink
        self.progress = progress

        self.network = TarokNet(
            hidden_size=config.hidden_size,
            oracle_critic=config.oracle_critic,
        )
        if resume_state_dict is not None:
            self.network.load_state_dict(resume_state_dict)
            log.info("Resumed from existing checkpoint")
        self.network = self.compute.prepare_network(self.network)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.buffer = ExperienceBuffer(
            capacity=config.buffer_capacity,
            max_staleness=config.max_staleness,
        )

        self._policy_version = 0
        self._stop_event = threading.Event()
        self._refresh_event = threading.Event()
        self._running = False

    def _resolve_num_producers(self) -> int:
        n = self.config.num_producers
        if n <= 0:
            n = auto_num_producers()
        return n

    def run(self) -> dict:
        """Run the full training loop. Returns final stats."""
        self._running = True
        self._stop_event.clear()

        num_producers = self._resolve_num_producers()
        use_mp = num_producers > 1

        t0 = time.time()
        total_games = 0
        total_experiences = 0
        mp_producer: MultiProcessProducer | None = None

        try:
            if use_mp:
                # --- Multi-process path: N workers with separate GILs ---
                conc_per_worker = max(
                    16, self.config.producer_concurrency // num_producers,
                )
                mp_producer = MultiProcessProducer(
                    hidden_size=self.config.hidden_size,
                    oracle_critic=self.config.oracle_critic,
                    concurrency_per_worker=conc_per_worker,
                    games_per_batch=self.config.games_per_session,
                    explore_rate=self.config.explore_rate,
                    initial_state_dict=self.network.state_dict(),
                    experience_buffer=self.buffer,
                    num_workers=num_producers,
                )
                mp_producer.start()
                log.info(
                    "Multi-process mode: %d producers × %d concurrent games "
                    "(total %d parallel games)",
                    num_producers, conc_per_worker,
                    num_producers * conc_per_worker,
                )
            else:
                # --- Single-thread fallback (num_producers=1) ---
                producer_thread = threading.Thread(
                    target=self._producer_loop,
                    name="ppo-producer",
                    daemon=True,
                )
                producer_thread.start()

            # Consumer loop (main thread)
            for session in range(self.config.num_sessions):
                if self._stop_event.is_set():
                    break

                # Wait for enough experiences
                batch = self.buffer.pull_batch(
                    min_size=self.config.min_experiences,
                    timeout=120.0,
                )
                if batch is None:
                    log.warning("Timeout waiting for experiences, session %d", session)
                    continue

                # Discard stale experiences
                self.buffer.discard_stale(self._policy_version)

                # Extract raw experiences
                experiences = [t.experience for t in batch]
                total_experiences += len(experiences)

                # Compute rewards and GAE per game
                game_experiences = self._group_by_game(experiences)
                all_exps_with_returns = []
                session_reward = 0.0
                n_games = 0

                for game_id, game_exps in game_experiences.items():
                    # Rewards are assigned at game end via the simulator
                    # For now, use the final experience's done flag
                    processed = self._compute_gae(game_exps)
                    all_exps_with_returns.extend(processed)
                    session_reward += sum(e.reward for e in game_exps)
                    n_games += 1

                total_games += n_games

                if not all_exps_with_returns:
                    continue

                # PPO update
                metrics = self._ppo_update(all_exps_with_returns)

                # Bump policy version and signal producer
                self._policy_version += 1
                if self._policy_version % self.config.network_refresh_interval == 0:
                    if mp_producer is not None:
                        mp_producer.refresh_weights(
                            self.network.state_dict(), self._policy_version,
                        )
                    else:
                        self._refresh_event.set()

                # Record metrics
                elapsed = time.time() - t0
                session_metrics = SessionMetrics(
                    session_id=session,
                    games_played=n_games,
                    total_reward=session_reward,
                    avg_reward=session_reward / max(n_games, 1),
                    policy_loss=metrics["policy_loss"],
                    value_loss=metrics["value_loss"],
                    entropy=metrics["entropy"],
                    total_loss=metrics["total_loss"],
                    games_per_sec=total_games / max(elapsed, 1),
                    experiences_count=len(experiences),
                    explore_rate=self.config.explore_rate,
                )

                if self.metrics_sink:
                    self.metrics_sink.record(session_metrics)

                if self.progress:
                    self.progress.report(TrainingProgress(
                        phase="ppo",
                        current_session=session,
                        total_sessions=self.config.num_sessions,
                        policy_version=self._policy_version,
                        elapsed_seconds=elapsed,
                        buffer_size=len(self.buffer),
                        is_running=True,
                    ))

                # Checkpoint
                if session > 0 and session % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(session)

                if session % 10 == 0:
                    log.info(
                        "Session %d/%d | games=%d | exps=%d | loss=%.4f | reward=%.2f | %.1f games/s",
                        session, self.config.num_sessions, n_games,
                        len(experiences), metrics["total_loss"],
                        session_reward / max(n_games, 1),
                        total_games / max(elapsed, 1),
                    )

        finally:
            self._stop_event.set()
            self._running = False
            if mp_producer is not None:
                mp_producer.stop()

        elapsed = time.time() - t0
        return {
            "total_sessions": self.config.num_sessions,
            "total_games": total_games,
            "total_experiences": total_experiences,
            "policy_version": self._policy_version,
            "elapsed_secs": round(elapsed, 1),
        }

    def stop(self) -> None:
        """Signal the training loop to stop."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Producer thread
    # ------------------------------------------------------------------

    def _producer_loop(self) -> None:
        """Continuously play games and push experiences into the buffer."""
        frozen_network = copy.deepcopy(self.network)
        frozen_network.eval()

        while not self._stop_event.is_set():
            # Play a batch of games
            results = self.simulator.play_batch(
                network=frozen_network,
                n_games=self.config.games_per_session,
                explore_rate=self.config.explore_rate,
            )

            # Assign rewards from game scores
            for result in results:
                if result.experiences:
                    scores = result.scores
                    # Normalize scores to [-1, 1] range
                    max_score = max(abs(s) for s in scores) if scores else 1.0
                    max_score = max(max_score, 1.0)

                    for exp in result.experiences:
                        player = exp.game_id % 4  # which player's perspective
                        exp.reward = scores[player % len(scores)] / max_score
                        exp.done = (exp.step_in_game == len(result.experiences) - 1)

                    self.buffer.push_game(result.experiences, self._policy_version)

            # Check if consumer wants us to refresh
            if self._refresh_event.is_set():
                self._refresh_event.clear()
                frozen_network.load_state_dict(self.network.state_dict())
                frozen_network.eval()

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _group_by_game(self, experiences: list[Experience]) -> dict[int, list[Experience]]:
        games: dict[int, list[Experience]] = {}
        for exp in experiences:
            games.setdefault(exp.game_id, []).append(exp)
        for game_exps in games.values():
            game_exps.sort(key=lambda e: e.step_in_game)
        return games

    def _compute_gae(self, experiences: list[Experience]) -> list[dict]:
        """Compute Generalized Advantage Estimation for a single game."""
        gamma = self.config.gamma
        lam = self.config.gae_lambda
        n = len(experiences)
        if n == 0:
            return []

        values = [e.value.item() if hasattr(e.value, 'item') else e.value for e in experiences]
        rewards = [e.reward for e in experiences]

        # GAE
        advantages = [0.0] * n
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = values[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae

        returns = [advantages[t] + values[t] for t in range(n)]

        result = []
        for i, exp in enumerate(experiences):
            result.append({
                "experience": exp,
                "advantage": advantages[i],
                "return": returns[i],
            })
        return result

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, data: list[dict]) -> dict:
        """Run PPO update epochs on the collected experiences."""
        cfg = self.config

        # Prepare tensors
        states = self.compute.stack_to_device([d["experience"].state for d in data])
        actions = self.compute.tensor_to_device(
            [d["experience"].action for d in data], dtype=torch.long,
        )
        old_log_probs = self.compute.stack_to_device(
            [d["experience"].log_prob for d in data],
        )
        advantages_t = self.compute.tensor_to_device(
            [d["advantage"] for d in data],
        )
        returns_t = self.compute.tensor_to_device(
            [d["return"] for d in data],
        )

        # Build legal masks
        masks_list = []
        for d in data:
            m = d["experience"].legal_mask
            if m is not None:
                if m.shape[0] < CARD_ACTION_SIZE:
                    padded = torch.zeros(CARD_ACTION_SIZE)
                    padded[:m.shape[0]] = m
                    m = padded
            else:
                m = torch.ones(CARD_ACTION_SIZE)
            masks_list.append(m)
        legal_masks = self.compute.stack_to_device(masks_list)

        decision_types = [d["experience"].decision_type for d in data]

        # Normalize advantages
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Oracle states if available
        oracle_states = None
        if any(d["experience"].oracle_state is not None for d in data):
            oracle_states = self.compute.stack_to_device([
                d["experience"].oracle_state if d["experience"].oracle_state is not None
                else torch.zeros_like(states[0])
                for d in data
            ])

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        N = len(data)
        self.network.train()

        for _epoch in range(cfg.ppo_epochs):
            perm = torch.randperm(N, device=states.device)

            for start in range(0, N, cfg.mini_batch_size):
                end = min(start + cfg.mini_batch_size, N)
                idx = perm[start:end]

                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages_t[idx]
                b_ret = returns_t[idx]
                b_masks = legal_masks[idx]
                b_dtypes = [decision_types[i] for i in idx.cpu().tolist()]
                b_oracle = oracle_states[idx] if oracle_states is not None else None

                # Forward through network (grouped by decision type)
                all_logits, values = self.network.forward_batch(
                    b_states, b_dtypes, b_oracle,
                )

                # Vectorised masked log-prob and entropy (no Python loop)
                masked_logits = all_logits.clone()
                masked_logits[b_masks == 0] = float("-inf")
                log_probs_full = torch.log_softmax(masked_logits, dim=-1)
                new_log_probs = log_probs_full.gather(1, b_actions.unsqueeze(1)).squeeze(1)
                probs = torch.softmax(masked_logits, dim=-1)
                log_probs_safe = log_probs_full.clamp(min=-30.0)
                entropy = -(probs * log_probs_safe).sum(dim=-1).mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, b_ret)

                # Total loss
                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    - cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "total_loss": (total_policy_loss + total_value_loss) / max(num_updates, 1),
            "num_updates": num_updates,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, session: int) -> None:
        from training_lab.entities.checkpoint import Checkpoint

        state_dict = self.network.state_dict()
        h = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            h.update(state_dict[key].cpu().numpy().tobytes()[:64])
        model_hash = h.hexdigest()[:8]

        checkpoint = Checkpoint(
            model_hash=model_hash,
            train_step=session,
            phase_label=f"S{session}",
            hidden_size=self.config.hidden_size,
            oracle_critic=self.config.oracle_critic,
        )
        self.store.save(state_dict, checkpoint)
        log.info("Saved checkpoint at session %d (hash=%s)", session, model_hash)
