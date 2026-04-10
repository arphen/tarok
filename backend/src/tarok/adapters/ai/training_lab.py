"""Training Lab — interactive neural network training and evaluation.

Provides endpoints for:
1. Creating a fresh neural network with a generated persona
2. Evaluating it against v1/v2/v3 heuristic bots (real games)
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
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

from tarok.adapters.ai.network import TarokNet


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
from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.imitation import imitation_pretrain
from tarok.adapters.ai.stockskis_player import StockSkisPlayer
from tarok.adapters.ai.stockskis_v2 import StockSkisPlayerV2
from tarok.adapters.ai.stockskis_v3 import StockSkisPlayerV3
from tarok.adapters.ai.stockskis_v4 import StockSkisPlayerV4
from tarok.adapters.api.spectator_observer import SpectatorObserver
from tarok.use_cases.game_loop import GameLoop

try:
    from tarok.adapters.ai.batch_game_runner import BatchGameRunner, GameResult, GameExperience
    from tarok.adapters.ai.compute import ComputeBackend, create_backend
    import tarok_engine as _te_check
    _HAS_RUST = _te_check is not None
except Exception:
    _HAS_RUST = False

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


def _ensure_hof_dir():
    HOF_DIR.mkdir(parents=True, exist_ok=True)


def save_to_hof(network: TarokNet, persona: dict, eval_history: list[dict], phase_label: str = "") -> dict:
    """Save a model snapshot to the Hall of Fame directory."""
    _ensure_hof_dir()
    h = _model_hash(network)
    display = _display_name(persona, h)
    filename = f"hof_{persona['first_name']}_{persona['last_name']}_age{persona['age']}_{h}.pt"

    latest_eval = eval_history[-1] if eval_history else {}

    data = {
        "model_state_dict": network.state_dict(),
        "persona": persona,
        "model_hash": h,
        "display_name": display,
        "model_name": display,
        "phase_label": phase_label,
        "eval_history": eval_history,
        "hidden_size": network.shared[0].out_features,
        "metrics": {
            "win_rate": latest_eval.get("vs_v1", 0),
            "vs_v1": latest_eval.get("vs_v1", 0),
            "vs_v2": latest_eval.get("vs_v2", 0),
            "vs_v3": latest_eval.get("vs_v3", 0),
            "avg_score_v1": latest_eval.get("avg_score_v1", 0),
        },
        "saved_at": time.time(),
    }

    torch.save(data, HOF_DIR / filename)

    # Update manifest
    manifest_path = HOF_DIR / "manifest.json"
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest.append({
        "filename": filename,
        "display_name": display,
        "persona": persona,
        "model_hash": h,
        "phase_label": phase_label,
        "vs_v1": latest_eval.get("vs_v1", 0),
        "vs_v2": latest_eval.get("vs_v2", 0),
        "vs_v3": latest_eval.get("vs_v3", 0),
        "saved_at": data["saved_at"],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {"filename": filename, "display_name": display, "model_hash": h}


def list_hof() -> list[dict]:
    """List all Hall of Fame models."""
    manifest_path = HOF_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text())


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
    sp_win_rate: float = 0.0
    sp_avg_reward: float = 0.0
    sp_avg_score: float = 0.0
    sp_bid_rate: float = 0.0
    sp_klop_rate: float = 0.0
    sp_solo_rate: float = 0.0
    sp_games_per_second: float = 0.0
    # History arrays (one entry per session)
    sp_win_rate_history: list[float] = field(default_factory=list)
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
        eval_bots = ["v1", "v2", "v3"]
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
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.trainer import PPOTrainer

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
    """Create 3 heuristic bot opponents of the given version."""
    if version == "v2":
        return [StockSkisPlayerV2(name=f"V2-{i}") for i in range(3)]
    elif version == "v3":
        return [StockSkisPlayerV3(name=f"V3-{i}") for i in range(3)]
    elif version == "v4":
        return [StockSkisPlayerV4(name=f"V4-{i}") for i in range(3)]
    else:
        return [StockSkisPlayer(name=f"V1-{i}", strength=1.0) for i in range(3)]


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

    loop = asyncio.new_event_loop()
    try:
        for g in range(num_games):
            players = [agent] + opponents
            game = GameLoop(players, rng=random.Random(g))
            _state, scores = loop.run_until_complete(game.run(dealer=g % 4))

            raw_score = scores.get(0, 0)
            opp_avg = sum(scores.get(i, 0) for i in range(1, 4)) / 3
            total_diff += raw_score - opp_avg

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

    win_rate = wins / max(num_games, 1)
    avg_score = total_diff / max(num_games, 1)
    return {
        "win_rate": round(win_rate, 4),
        "avg_score": round(avg_score, 2),
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
    from tarok.adapters.ai.trainer import PPOTrainer

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
        eval_bots = ["v1", "v2", "v3"]
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
        eval_bots = ["v1", "v2", "v3"]
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
                ),
                daemon=True,
            )
            p.start()
            _island_processes.append(p)

        _lab.pbt_population = [
            {"index": i, "label": f"Island {i}", "status": "starting",
             "fitness": 0, "games": 0, "hparams": {},
             "batch_avg_reward": 0, "batch_win_rate": 0,
             **{f"vs_{v}": 0 for v in eval_bots},
             **{f"avg_score_{v}": 0 for v in eval_bots},
             "loss": 0, "copied_from": None, "mutations": 0,
             "survival_count": 0, "model_hash": "",
             "games_per_sec": 0, "generation": 0}
            for i in range(population_size)
        ]
        _lab.total_training_sessions = int(time_limit_minutes)

        # Poll island stats files for dashboard updates
        t_start = time.perf_counter()
        while _lab.running:
            elapsed = time.perf_counter() - t_start
            stats = read_all_island_stats(CHECKPOINTS_DIR)
            total_games = sum(s.get("total_games", 0) for s in stats)
            total_gps = sum(s.get("games_per_sec", 0) for s in stats)

            _lab.self_play_games = total_games
            _lab.sp_games_per_second = round(total_gps, 1)
            _lab.training_sessions_done = min(int(elapsed / 60), int(time_limit_minutes))

            # Update per-island population display
            for s in stats:
                idx = s.get("island_id", 0)
                if idx < len(_lab.pbt_population):
                    island_hp = s.get("hparams", {})
                    _lab.pbt_population[idx].update({
                        "status": s.get("status", "running"),
                        "fitness": s.get("best_fitness", 0),
                        "games": s.get("total_games", 0),
                        "batch_avg_reward": s.get("avg_score", 0),
                        "batch_win_rate": s.get("session_win_rate", 0),
                        "vs_v1": s.get("vs_v1", 0),
                        "vs_v3": s.get("vs_v3", 0),
                        "loss": s.get("loss", 0),
                        "games_per_sec": s.get("games_per_sec", 0),
                        "generation": s.get("generation", 0),
                        "hparams": island_hp,
                        "model_hash": f"island-{idx}",
                    })

            # Best across all islands
            if stats:
                best = max(stats, key=lambda s: s.get("best_fitness", 0))
                _lab.sp_win_rate = best.get("win_rate", 0)
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

        # Load best model from HoF back into lab state
        hof_data = _load_best_from_hof(HOF_DIR)
        if hof_data is not None and _lab.network is not None:
            _lab.network.load_state_dict(hof_data["model_state_dict"])
            _lab.model_hash = _model_hash(_lab.network)

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
                        _lab.sp_win_rate = msg["win_rate"]
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
):
    """Self-play PPO training pipeline:
    1. Create PPOTrainer wrapping the lab network
    2. Run sessions of self-play with PPO updates
    3. Every eval_interval sessions, eval vs all bots and save
    """
    global _lab

    try:
        from tarok.adapters.ai.trainer import PPOTrainer

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

        # Detect if batched runner is available (Rust engine required)
        can_batch = _HAS_RUST
        if can_batch:
            compute = create_backend(dev)
            _lab.network = compute.prepare_network(_lab.network)

        for session_idx in range(num_sessions):
            if not _lab.running:
                break

            # Update LR and entropy schedules
            trainer._update_schedule(session_idx, num_sessions)

            _lab.phase = "self_play"
            _lab.training_sessions_done = session_idx + 1
            _lab.self_play_sessions = session_idx + 1
            await asyncio.sleep(0)

            all_experiences: list = []
            session_scores: list[int] = []
            session_wins = 0
            session_bids = 0
            session_klops = 0
            session_solos = 0
            session_start = time.time()

            if can_batch:
                # ── Batched self-play (all games at once) ──
                runner = BatchGameRunner(
                    network=_lab.network,
                    concurrency=min(32, games_per_session),
                    oracle=_lab.network.oracle_critic_enabled,
                    compute=compute,
                )
                game_offset = session_idx * games_per_session
                results = runner.run(
                    total_games=games_per_session,
                    explore_rate=0.1,
                    dealer_offset=game_offset,
                    game_id_offset=game_offset,
                )

                for result in results:
                    scores = result.scores
                    raw_score = scores.get(0, 0)

                    # Convert GameExperience → Experience with terminal reward
                    from tarok.adapters.ai.agent import Experience
                    game_exps: list = []
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
                    if game_exps:
                        game_exps[-1].reward = raw_score / 100.0
                        game_exps[-1].done = True
                    all_experiences.extend(game_exps)

                    session_scores.append(raw_score)
                    _lab.self_play_games += 1

                    # Track per-game stats
                    _, win = _score_game(result.py_state, scores, player_idx=0)
                    session_wins += int(win)
                    if result.is_klop:
                        session_klops += 1
                    if result.declarer_p0:
                        session_bids += 1
                    if result.is_solo and result.declarer_p0:
                        session_solos += 1

                # Yield to API
                await asyncio.sleep(0)

            else:
                # ── Fallback: sequential GameLoop (no Rust engine) ──
                for agent in agents:
                    agent.set_training(True)
                    agent.clear_experiences()

                for g in range(games_per_session):
                    if not _lab.running:
                        break

                    for agent in agents:
                        agent.clear_experiences()

                    game = GameLoop(agents)
                    state, scores = await game.run(dealer=(session_idx * games_per_session + g) % 4)

                    raw_score = scores.get(0, 0)
                    for i, agent in enumerate(agents):
                        reward = scores.get(i, 0) / 100.0
                        agent.finalize_game(reward)
                        all_experiences.extend(agent.experiences)

                    session_scores.append(raw_score)
                    _lab.self_play_games += 1

                    _, win = _score_game(state, scores, player_idx=0)
                    session_wins += int(win)
                    is_klop = state.contract is not None and state.contract.is_klop
                    did_bid = state.declarer == 0
                    is_solo = state.contract is not None and state.contract.is_solo and state.declarer == 0
                    if did_bid:
                        session_bids += 1
                    if is_klop:
                        session_klops += 1
                    if is_solo:
                        session_solos += 1

                    if (g + 1) % 5 == 0 or g == games_per_session - 1:
                        await asyncio.sleep(0)

            if not _lab.running or not all_experiences:
                break

            # PPO update
            info = trainer._ppo_update(all_experiences)
            _lab.current_loss = info.get("policy_loss", 0)

            # Update per-session metrics
            n = max(len(session_scores), 1)
            elapsed = max(time.time() - session_start, 0.001)
            _lab.sp_win_rate = session_wins / n
            _lab.sp_avg_reward = sum(s / 100.0 for s in session_scores) / n
            _lab.sp_avg_score = sum(session_scores) / n
            _lab.sp_bid_rate = session_bids / n
            _lab.sp_klop_rate = session_klops / n
            _lab.sp_solo_rate = session_solos / n
            _lab.sp_games_per_second = len(session_scores) / elapsed

            _lab.sp_win_rate_history.append(round(_lab.sp_win_rate, 4))
            _lab.sp_avg_reward_history.append(round(_lab.sp_avg_reward, 4))
            _lab.sp_avg_score_history.append(round(_lab.sp_avg_score, 2))
            _lab.sp_loss_history.append(round(_lab.current_loss, 4))
            _lab.sp_bid_rate_history.append(round(_lab.sp_bid_rate, 4))
            _lab.sp_klop_rate_history.append(round(_lab.sp_klop_rate, 4))
            _lab.sp_solo_rate_history.append(round(_lab.sp_solo_rate, 4))
            _lab.sp_min_score_history.append(float(min(session_scores)) if session_scores else 0.0)
            _lab.sp_max_score_history.append(float(max(session_scores)) if session_scores else 0.0)

            _lab.persona["age"] = _lab.persona.get("age", 0) + 1

            # Periodic evaluation
            if (session_idx + 1) % eval_interval == 0 or session_idx == num_sessions - 1:
                _lab.phase = "evaluating"
                await asyncio.sleep(0)

                results = await _evaluate_vs_selected_bots(_lab.network, eval_games, eval_bots)

                step = len(_lab.eval_history)
                entry = _eval_results_to_history_entry(
                    results,
                    step=step,
                    label=f"SP Session {session_idx + 1}",
                    program="self_play",
                    loss=_lab.current_loss,
                    experiences=_lab.expert_experiences,
                    games=_lab.self_play_games,
                )
                _lab.eval_history.append(entry)

                _lab.persona["age"] = _lab.persona.get("age", 0) + 1
                _update_persona_hash()
                info = save_to_hof(
                    _lab.network, _lab.persona, _lab.eval_history,
                    phase_label=f"selfplay-s{session_idx + 1}",
                )
                _lab.snapshots.append(info)

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
        "sp_win_rate": _lab.sp_win_rate,
        "sp_avg_reward": _lab.sp_avg_reward,
        "sp_avg_score": _lab.sp_avg_score,
        "sp_bid_rate": _lab.sp_bid_rate,
        "sp_klop_rate": _lab.sp_klop_rate,
        "sp_solo_rate": _lab.sp_solo_rate,
        "sp_games_per_second": _lab.sp_games_per_second,
        "sp_win_rate_history": _lab.sp_win_rate_history[-500:],
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
    _lab.sp_win_rate = 0.0
    _lab.sp_avg_reward = 0.0
    _lab.sp_avg_score = 0.0
    _lab.sp_bid_rate = 0.0
    _lab.sp_klop_rate = 0.0
    _lab.sp_solo_rate = 0.0
    _lab.sp_games_per_second = 0.0
    _lab.sp_win_rate_history = []
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
        eval_bots = ["v1", "v2", "v3"]

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
        eval_bots = ["v1", "v2", "v3"]

    if _lab.network is None:
        create_lab_network()

    _lab.running = True
    _lab.error = None
    _lab.pbt_enabled = pbt_enabled or population_size > 1
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
