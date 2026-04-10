"""Island worker — self-contained training loop for island-model PBT.

Each island is a standalone OS process that:
1. Seeds from the best Hall-of-Fame weights (or random init)
2. Runs self-play PPO in a tight loop
3. Periodically saves improved weights back to HoF
4. Writes stats to a JSON file for the dashboard
5. Stops when the shared stop_event is set

No IPC during training — only filesystem coordination.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import random
import time
from multiprocessing import Event
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATS_DIR_NAME = "island_stats"


def _stats_dir(checkpoints_dir: Path) -> Path:
    d = checkpoints_dir / _STATS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Hyperparameter mutation (standalone copy to avoid importing training_lab)
# ---------------------------------------------------------------------------

_HPARAM_SPACE = {
    "lr": {"low": 1e-5, "high": 5e-3, "log": True, "integer": False, "sigma": 0.18},
    "gamma": {"low": 0.9, "high": 0.999, "log": False, "integer": False, "sigma": 0.015},
    "gae_lambda": {"low": 0.85, "high": 0.99, "log": False, "integer": False, "sigma": 0.02},
    "clip_epsilon": {"low": 0.05, "high": 0.35, "log": False, "integer": False, "sigma": 0.05},
    "value_coef": {"low": 0.1, "high": 1.2, "log": False, "integer": False, "sigma": 0.12},
    "entropy_coef": {"low": 0.001, "high": 0.08, "log": True, "integer": False, "sigma": 0.2},
    "epochs_per_update": {"low": 2, "high": 8, "log": False, "integer": True, "sigma": 1.0},
    "batch_size": {"low": 32, "high": 256, "log": False, "integer": True, "sigma": 24.0},
    "explore_rate": {"low": 0.02, "high": 0.25, "log": False, "integer": False, "sigma": 0.03},
}


def _clip_hparam(name: str, value: float) -> Any:
    spec = _HPARAM_SPACE[name]
    clipped = max(spec["low"], min(spec["high"], value))
    return int(round(clipped)) if spec["integer"] else float(clipped)


def _mutate_hparams(base: dict, rng: random.Random, scale: float = 1.0) -> dict:
    mutated = dict(base)
    changed = 0
    for name, spec in _HPARAM_SPACE.items():
        if name not in mutated:
            continue
        if rng.random() > 0.55:
            continue
        changed += 1
        value = float(mutated[name])
        sigma = spec["sigma"] * max(scale, 1e-3)
        if spec["log"]:
            value = 10 ** (math.log10(max(value, spec["low"])) + rng.gauss(0.0, sigma))
        else:
            value += rng.gauss(0.0, sigma)
        mutated[name] = _clip_hparam(name, value)
    if changed == 0:
        forced = rng.choice([k for k in _HPARAM_SPACE if k in mutated])
        spec = _HPARAM_SPACE[forced]
        value = float(mutated[forced])
        if spec["log"]:
            value = 10 ** (math.log10(max(value, spec["low"])) + rng.gauss(0.0, spec["sigma"]))
        else:
            value += rng.gauss(0.0, spec["sigma"])
        mutated[forced] = _clip_hparam(forced, value)
    return mutated


def _default_hparams(lr: float = 3e-4) -> dict:
    return {
        "lr": lr,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "epochs_per_update": 4,
        "batch_size": 64,
        "explore_rate": 0.1,
    }


# ---------------------------------------------------------------------------
# HoF interaction (minimal, filesystem-only)
# ---------------------------------------------------------------------------

def _load_best_from_hof(hof_dir: Path) -> dict | None:
    """Load the best model from the Hall of Fame by vs_v3 win rate."""
    manifest_path = hof_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if not manifest:
        return None
    best_entry = max(manifest, key=lambda e: e.get("vs_v3", 0))
    pt_path = hof_dir / best_entry["filename"]
    if not pt_path.exists():
        return None
    try:
        return torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _save_to_hof(
    network_state: dict,
    hidden_size: int,
    island_id: int,
    generation: int,
    eval_results: dict,
    hof_dir: Path,
) -> None:
    """Save a model snapshot to the Hall of Fame."""
    import hashlib

    hof_dir.mkdir(parents=True, exist_ok=True)

    h = hashlib.md5(
        str(sorted((k, v.sum().item()) for k, v in network_state.items())).encode()
    ).hexdigest()[:8]

    filename = f"hof_island{island_id}_gen{generation}_{h}.pt"
    data = {
        "model_state_dict": network_state,
        "persona": {"first_name": f"Island{island_id}", "last_name": "Worker", "age": generation},
        "model_hash": h,
        "display_name": f"Island {island_id} gen {generation} #{h}",
        "model_name": f"Island {island_id} gen {generation} #{h}",
        "phase_label": f"island-{island_id}-g{generation}",
        "eval_history": [],
        "hidden_size": hidden_size,
        "metrics": {
            "vs_v1": eval_results.get("vs_v1", 0),
            "vs_v2": eval_results.get("vs_v2", 0),
            "vs_v3": eval_results.get("vs_v3", 0),
        },
        "saved_at": time.time(),
    }
    torch.save(data, hof_dir / filename)

    # Append to manifest (with file lock via temp file)
    manifest_path = hof_dir / "manifest.json"
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = []
    manifest.append({
        "filename": filename,
        "display_name": data["display_name"],
        "persona": data["persona"],
        "model_hash": h,
        "phase_label": data["phase_label"],
        "vs_v1": eval_results.get("vs_v1", 0),
        "vs_v2": eval_results.get("vs_v2", 0),
        "vs_v3": eval_results.get("vs_v3", 0),
        "saved_at": data["saved_at"],
    })
    # Atomic write
    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(manifest_path)


# ---------------------------------------------------------------------------
# Evaluation (synchronous, runs inside the island process)
# ---------------------------------------------------------------------------

def _evaluate_vs_bots(network, num_games: int, version: str) -> dict:
    """Evaluate a network vs heuristic bots. Fully synchronous."""
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.stockskis_player import StockSkisPlayer
    from tarok.adapters.ai.stockskis_v2 import StockSkisPlayerV2
    from tarok.adapters.ai.stockskis_v3 import StockSkisPlayerV3
    from tarok.use_cases.game_loop import GameLoop

    network.eval()
    dev = next(network.parameters()).device
    hidden_size = network.shared[0].out_features
    agent = RLAgent(name="Island-NN", hidden_size=hidden_size, device=str(dev), explore_rate=0.0)
    agent.network = network
    agent.set_training(False)

    if version == "v2":
        opponents = [StockSkisPlayerV2(name=f"V2-{i}") for i in range(3)]
    elif version == "v3":
        opponents = [StockSkisPlayerV3(name=f"V3-{i}") for i in range(3)]
    else:
        opponents = [StockSkisPlayer(name=f"V1-{i}", strength=1.0) for i in range(3)]

    wins = 0
    total_diff = 0

    loop = asyncio.new_event_loop()
    try:
        for g in range(num_games):
            game = GameLoop([agent] + opponents, rng=random.Random(g))
            state, scores = loop.run_until_complete(game.run(dealer=g % 4))
            raw_score = scores.get(0, 0)
            opp_avg = sum(scores.get(i, 0) for i in range(1, 4)) / 3
            total_diff += raw_score - opp_avg

            is_klop = state.contract is not None and state.contract.is_klop
            if is_klop:
                won = raw_score > 0
            elif state.declarer == 0:
                won = raw_score > 0
            else:
                declarer_score = scores.get(state.declarer, 0) if state.declarer is not None else 0
                won = declarer_score < 0
            if won:
                wins += 1
            agent.clear_experiences()
    finally:
        loop.close()

    return {
        "win_rate": round(wins / max(num_games, 1), 4),
        "avg_score": round(total_diff / max(num_games, 1), 2),
    }


# ---------------------------------------------------------------------------
# Stats file (dashboard communication)
# ---------------------------------------------------------------------------

def _write_stats(stats_dir: Path, island_id: int, stats: dict) -> None:
    """Write island stats to a JSON file for the coordinator to read."""
    path = stats_dir / f"island_{island_id}.json"
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(stats))
        tmp.replace(path)
    except Exception:
        pass


def read_all_island_stats(checkpoints_dir: Path) -> list[dict]:
    """Read stats from all active islands. Called by the coordinator."""
    d = checkpoints_dir / _STATS_DIR_NAME
    if not d.exists():
        return []
    results = []
    for f in sorted(d.glob("island_*.json")):
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            continue
    return results


def clear_island_stats(checkpoints_dir: Path) -> None:
    """Remove all island stats files."""
    d = checkpoints_dir / _STATS_DIR_NAME
    if d.exists():
        for f in d.glob("island_*.json"):
            try:
                f.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Island worker — the main entry point for each process
# ---------------------------------------------------------------------------

def island_worker(
    island_id: int,
    stop_event: Event,
    checkpoints_dir: str,
    hidden_size: int = 256,
    games_per_session: int = 20,
    eval_games: int = 50,
    eval_interval: int = 10,
    seed_state_dict: dict | None = None,
    hparams: dict | None = None,
    mutation_scale: float = 1.0,
) -> None:
    """Self-contained training loop for one island.

    Runs in its own OS process.  Communicates only via:
    - Hall of Fame directory (read/write model weights)
    - Stats directory (write JSON for dashboard)
    - stop_event (read-only signal to shut down)
    """
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.network import TarokNet
    from tarok.adapters.ai.trainer import PPOTrainer

    ckpt_dir = Path(checkpoints_dir)
    hof_dir = ckpt_dir / "hall_of_fame"
    stats_dir = _stats_dir(ckpt_dir)
    rng = random.Random(time.time() + island_id * 1000)

    # --- Initialize network ---
    net = TarokNet(hidden_size=hidden_size)
    if seed_state_dict is not None:
        net.load_state_dict(seed_state_dict)
    else:
        hof_data = _load_best_from_hof(hof_dir)
        if hof_data is not None:
            net.load_state_dict(hof_data["model_state_dict"])
    net = net.to("cpu")

    # --- Initialize hparams ---
    hp = hparams or _default_hparams()
    hp = _mutate_hparams(hp, rng, mutation_scale)

    # --- Create trainer ---
    agents = [
        RLAgent(name=f"P{s}", hidden_size=hidden_size, device="cpu",
                explore_rate=float(hp.get("explore_rate", 0.1)))
        for s in range(4)
    ]
    agents[0].network = net
    for a in agents[1:]:
        a.network = net

    trainer = PPOTrainer(
        agents=agents,
        lr=float(hp["lr"]),
        gamma=float(hp["gamma"]),
        gae_lambda=float(hp["gae_lambda"]),
        clip_epsilon=float(hp["clip_epsilon"]),
        value_coef=float(hp["value_coef"]),
        entropy_coef=float(hp["entropy_coef"]),
        epochs_per_update=int(hp["epochs_per_update"]),
        batch_size=int(hp["batch_size"]),
        games_per_session=games_per_session,
        device="cpu",
        stockskis_ratio=0.0,
        fsp_ratio=0.0,
        use_rust_engine=True,
        save_dir=str(ckpt_dir / "island_tmp" / f"island_{island_id}"),
        batch_concurrency=32,
        value_clip=0.2,
    )
    trainer._running = True

    # --- Training loop ---
    generation = 0
    total_games = 0
    total_wins = 0.0
    total_scores = 0.0
    best_eval: dict = {}
    best_fitness = -1.0
    t_start = time.perf_counter()
    session_times: list[float] = []

    loop = asyncio.new_event_loop()
    try:
        while not stop_event.is_set():
            generation += 1
            session_start = time.perf_counter()

            # --- Self-play session ---
            all_experiences = []
            session_games = 0
            session_wins = 0.0
            session_scores = 0.0

            for sess in range(eval_interval):
                if stop_event.is_set():
                    break
                exps, stats, sk_exps, sk_stats = loop.run_until_complete(
                    trainer._play_session_batched(sess, total_games, time.time())
                )
                all_experiences.extend(exps)
                all_experiences.extend(sk_exps)
                for stat in stats + sk_stats:
                    raw_score = stat["raw_score"]
                    session_scores += raw_score
                    session_wins += 1.0 if raw_score > 0 else 0.0
                    session_games += 1
                    total_games += 1
                trainer.metrics.session += 1

            if stop_event.is_set():
                break

            # --- PPO update ---
            loss = 0.0
            if all_experiences:
                loss_info = trainer._ppo_update(all_experiences)
                loss = loss_info.get("policy_loss", 0.0)

            total_wins += session_wins
            total_scores += session_scores
            session_time = time.perf_counter() - session_start
            session_times.append(session_time)
            elapsed = time.perf_counter() - t_start
            gps = total_games / max(elapsed, 0.01)

            # --- Periodic eval ---
            net.eval()
            v1 = _evaluate_vs_bots(net, eval_games, "v1")
            v3 = _evaluate_vs_bots(net, eval_games, "v3")
            net.train()

            fitness = 0.6 * v3["win_rate"] + 0.25 * v1["win_rate"] + 0.15 * max(0.0, min(1.0, (session_wins / max(session_games, 1))))

            # --- Save to HoF if improved ---
            if fitness > best_fitness:
                best_fitness = fitness
                best_eval = {"vs_v1": v1["win_rate"], "vs_v3": v3["win_rate"]}
                _save_to_hof(
                    net.state_dict(), hidden_size, island_id, generation,
                    {"vs_v1": v1["win_rate"], "vs_v2": 0, "vs_v3": v3["win_rate"]},
                    hof_dir,
                )

            # --- Check HoF for better weights (migration) ---
            if generation % 5 == 0:
                hof_data = _load_best_from_hof(hof_dir)
                if hof_data is not None:
                    hof_v3 = hof_data.get("metrics", {}).get("vs_v3", 0)
                    if hof_v3 > best_fitness + 0.05:
                        # Adopt better weights from another island
                        net.load_state_dict(hof_data["model_state_dict"])
                        for a in agents:
                            a.network = net
                        # Re-mutate hparams
                        hp = _mutate_hparams(hp, rng, mutation_scale)
                        for pg in trainer.optimizer.param_groups:
                            pg["lr"] = float(hp["lr"])
                        trainer.gamma = float(hp["gamma"])
                        trainer.gae_lambda = float(hp["gae_lambda"])
                        trainer.clip_epsilon = float(hp["clip_epsilon"])
                        trainer.value_coef = float(hp["value_coef"])
                        trainer.entropy_coef = float(hp["entropy_coef"])
                        trainer.epochs_per_update = int(hp["epochs_per_update"])
                        trainer.batch_size = int(hp["batch_size"])
                        for a in agents:
                            a.explore_rate = float(hp.get("explore_rate", 0.1))

            # --- Write stats for dashboard ---
            _write_stats(stats_dir, island_id, {
                "island_id": island_id,
                "generation": generation,
                "total_games": total_games,
                "games_per_sec": round(gps, 1),
                "session_games": session_games,
                "session_time": round(session_time, 2),
                "win_rate": round(total_wins / max(total_games, 1), 4),
                "session_win_rate": round(session_wins / max(session_games, 1), 4),
                "avg_score": round(total_scores / max(total_games, 1), 2),
                "loss": round(loss, 4),
                "vs_v1": v1["win_rate"],
                "vs_v3": v3["win_rate"],
                "fitness": round(fitness, 4),
                "best_fitness": round(best_fitness, 4),
                "elapsed": round(elapsed, 1),
                "hparams": {k: round(v, 6) if isinstance(v, float) else v for k, v in hp.items()},
                "status": "running",
            })

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        loop.close()

        # Write final stats
        elapsed = time.perf_counter() - t_start
        _write_stats(stats_dir, island_id, {
            "island_id": island_id,
            "generation": generation,
            "total_games": total_games,
            "games_per_sec": round(total_games / max(elapsed, 0.01), 1),
            "win_rate": round(total_wins / max(total_games, 1), 4),
            "vs_v1": best_eval.get("vs_v1", 0),
            "vs_v3": best_eval.get("vs_v3", 0),
            "fitness": round(best_fitness, 4),
            "best_fitness": round(best_fitness, 4),
            "elapsed": round(elapsed, 1),
            "hparams": {k: round(v, 6) if isinstance(v, float) else v for k, v in hp.items()},
            "status": "stopped",
        })
