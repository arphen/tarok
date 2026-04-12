"""Multi-process experience production — bypasses the GIL.

Spawns N worker processes, each running RustBatchGameRunner on CPU.
Experiences flow back to the main process via a mp.Queue.
A collector thread drains the queue into the shared ExperienceBuffer
so the consumer (PPO update) loop is unchanged.

On Apple Silicon the GPU is reserved for PPO gradients only;
producers do inference on CPU (fast for a 758K-param network).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from training_lab.entities.experience import Experience
    from training_lab.entities.experience_buffer import ExperienceBuffer

log = logging.getLogger(__name__)


def auto_num_producers() -> int:
    """Pick a reasonable default: half the CPU cores minus 1 for the consumer."""
    cpus = os.cpu_count() or 4
    return max(1, cpus // 2 - 1)


# -----------------------------------------------------------------------
# Worker process (top-level function so it's picklable with "spawn")
# -----------------------------------------------------------------------


def _producer_worker(
    worker_id: int,
    hidden_size: int,
    oracle_critic: bool,
    concurrency: int,
    games_per_batch: int,
    explore_rate: float,
    initial_weights: dict,
    experience_queue: mp.Queue,
    weight_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    """Self-play worker — runs entirely on CPU in its own process/GIL."""
    # Lazy imports so the main process doesn't pay startup cost twice
    from training_lab.adapters.compute.cpu_backend import CpuBackend
    from training_lab.adapters.engine.rust_batch_runner import RustBatchGameRunner
    from training_lab.entities.network import TarokNet

    backend = CpuBackend()
    network = TarokNet(hidden_size=hidden_size, oracle_critic=oracle_critic)
    network.load_state_dict(initial_weights)
    network = backend.prepare_network(network)
    network.eval()

    runner = RustBatchGameRunner(
        compute=backend,
        concurrency=concurrency,
        oracle=oracle_critic,
    )

    batch_counter = 0
    log.info("Producer worker %d started (concurrency=%d)", worker_id, concurrency)

    while not stop_event.is_set():
        # Non-blocking weight refresh
        try:
            while True:
                new_weights = weight_queue.get_nowait()
                network.load_state_dict(new_weights)
                network.eval()
        except queue.Empty:
            pass

        results = runner.play_batch(
            network=network,
            n_games=games_per_batch,
            explore_rate=explore_rate,
        )

        # Unique game-id offset so different workers never collide.
        # Worker k's batch b gets ids [k*10^8 + b*games, k*10^8 + (b+1)*games).
        id_offset = worker_id * 100_000_000 + batch_counter * games_per_batch

        all_states = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_dtypes = []
        all_rewards = []
        all_dones = []
        all_game_ids = []
        all_steps = []
        all_legal_masks = []
        all_oracle = []

        for result in results:
            if not result.experiences:
                continue
            scores = result.scores
            max_score = max(abs(s) for s in scores) if scores else 1.0
            max_score = max(max_score, 1.0)
            n_exps = len(result.experiences)

            for exp in result.experiences:
                player = exp.game_id % 4
                exp.reward = scores[player % len(scores)] / max_score
                exp.done = (exp.step_in_game == n_exps - 1)
                exp.game_id += id_offset

                all_states.append(exp.state)
                all_actions.append(exp.action)
                all_log_probs.append(exp.log_prob)
                all_values.append(
                    exp.value if exp.value.dim() > 0 else exp.value.unsqueeze(0)
                )
                all_dtypes.append(exp.decision_type.value)
                all_rewards.append(exp.reward)
                all_dones.append(exp.done)
                all_game_ids.append(exp.game_id)
                all_steps.append(exp.step_in_game)
                all_legal_masks.append(exp.legal_mask)
                all_oracle.append(exp.oracle_state)

        if not all_states:
            batch_counter += 1
            continue

        # Pack into a single dict of batched tensors for cheap IPC
        batch = {
            "states": torch.stack(all_states),
            "actions": torch.tensor(all_actions, dtype=torch.long),
            "log_probs": torch.stack(all_log_probs),
            "values": torch.stack(all_values).squeeze(-1),
            "decision_types": all_dtypes,
            "rewards": all_rewards,
            "dones": all_dones,
            "game_ids": all_game_ids,
            "steps": all_steps,
            "legal_masks": all_legal_masks,
            "oracle_states": all_oracle,
        }

        try:
            experience_queue.put(batch, timeout=10.0)
        except queue.Full:
            pass  # consumer is behind; drop this batch rather than blocking

        batch_counter += 1

    log.info("Producer worker %d stopped", worker_id)


# -----------------------------------------------------------------------
# Unbatching (runs in collector thread inside the main process)
# -----------------------------------------------------------------------


def _unbatch(batch: dict) -> list["Experience"]:
    """Unpack a serialised batch dict back into Experience objects."""
    from training_lab.entities.encoding import DecisionType
    from training_lab.entities.experience import Experience

    n = len(batch["rewards"])
    exps: list[Experience] = []
    for i in range(n):
        exps.append(Experience(
            state=batch["states"][i],
            action=batch["actions"][i].item(),
            log_prob=batch["log_probs"][i],
            value=batch["values"][i],
            decision_type=DecisionType(batch["decision_types"][i]),
            reward=batch["rewards"][i],
            done=batch["dones"][i],
            oracle_state=batch["oracle_states"][i],
            legal_mask=batch["legal_masks"][i],
            game_id=batch["game_ids"][i],
            step_in_game=batch["steps"][i],
        ))
    return exps


# -----------------------------------------------------------------------
# MultiProcessProducer — manages workers + collector thread
# -----------------------------------------------------------------------


class MultiProcessProducer:
    """Spawn N worker processes for GIL-free parallel self-play.

    Usage::

        mp_prod = MultiProcessProducer(config, initial_weights, buffer, n=4)
        mp_prod.start()
        ...  # consumer loop reads from buffer as usual
        mp_prod.stop()
    """

    def __init__(
        self,
        hidden_size: int,
        oracle_critic: bool,
        concurrency_per_worker: int,
        games_per_batch: int,
        explore_rate: float,
        initial_state_dict: dict,
        experience_buffer: "ExperienceBuffer",
        num_workers: int = 4,
    ):
        self.num_workers = num_workers
        self.hidden_size = hidden_size
        self.oracle_critic = oracle_critic
        self.concurrency = concurrency_per_worker
        self.games_per_batch = games_per_batch
        self.explore_rate = explore_rate
        # Move to shared memory so spawn-ed workers get it cheaply
        self.initial_state_dict = {
            k: v.share_memory_() for k, v in initial_state_dict.items()
        }
        self.buffer = experience_buffer

        self._exp_queue: mp.Queue | None = None
        self._weight_queues: list[mp.Queue] = []
        self._stop_event: mp.Event | None = None
        self._workers: list[mp.Process] = []
        self._collector_thread: threading.Thread | None = None
        self._collector_stop = threading.Event()
        self._policy_version = 0

    # -- public API --

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._exp_queue = ctx.Queue(maxsize=self.num_workers * 4)
        self._stop_event = ctx.Event()

        for i in range(self.num_workers):
            wq: mp.Queue = ctx.Queue(maxsize=2)
            self._weight_queues.append(wq)
            p = ctx.Process(
                target=_producer_worker,
                args=(
                    i,
                    self.hidden_size,
                    self.oracle_critic,
                    self.concurrency,
                    self.games_per_batch,
                    self.explore_rate,
                    self.initial_state_dict,
                    self._exp_queue,
                    wq,
                    self._stop_event,
                ),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Collector thread drains the mp.Queue → ExperienceBuffer
        self._collector_stop.clear()
        self._collector_thread = threading.Thread(
            target=self._collector_loop,
            daemon=True,
            name="mp-collector",
        )
        self._collector_thread.start()
        log.info(
            "Started %d producer processes (%d concurrent games each)",
            self.num_workers, self.concurrency,
        )

    def refresh_weights(self, new_state_dict: dict, policy_version: int) -> None:
        """Push updated weights to all workers (non-blocking best-effort)."""
        self._policy_version = policy_version
        # Share memory so workers don't need a full copy
        shared = {k: v.cpu().share_memory_() for k, v in new_state_dict.items()}
        for wq in self._weight_queues:
            # Drain stale weights first
            try:
                while not wq.empty():
                    wq.get_nowait()
            except Exception:
                pass
            try:
                wq.put_nowait(shared)
            except queue.Full:
                pass  # worker hasn't consumed previous update yet

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        self._collector_stop.set()
        for p in self._workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        self._workers.clear()
        if self._collector_thread:
            self._collector_thread.join(timeout=2.0)
        log.info("All producer workers stopped")

    # -- internal --

    def _collector_loop(self) -> None:
        """Drain the mp.Queue into the ExperienceBuffer."""
        assert self._exp_queue is not None
        while not self._collector_stop.is_set():
            try:
                batch = self._exp_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            exps = _unbatch(batch)
            self.buffer.push_game(exps, self._policy_version)
