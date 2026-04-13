"""Tournament endpoints — single match and multi-tournament simulation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.bot_registry import get_registry
from tarok.adapters.ai.random_agent import RandomPlayer
from tarok.adapters.ai.rust_game_loop import RustGameLoop as GameLoop

router = APIRouter(prefix="/api/tournament", tags=["tournament"])

log = logging.getLogger(__name__)


# ---- Models ----

class TournamentMatchRequest(BaseModel):
    agents: list[dict]  # [{name, type, checkpoint?}] — exactly 4
    num_games: int = 5


class MultiTournamentRequest(BaseModel):
    agents: list[dict]  # [{name, type, checkpoint?}] — 4-8 entries
    num_tournaments: int = 5
    games_per_round: int = 5


# ---- Globals ----

_multi_tournament_task: asyncio.Task | None = None
_multi_tournament_progress: dict | None = None


# ---- Helpers ----

def _build_agent(cfg: dict, idx: int):
    """Instantiate a single agent from a config dict."""
    agent_type = str(cfg.get("type", "rl")).strip().lower()
    agent_name = cfg.get("name", f"Agent-{idx}")
    checkpoint = cfg.get("checkpoint")

    registry = get_registry()

    # Check registry first (random, stockskis_v*, etc.)
    if registry.has(agent_type):
        return registry.create(agent_type, name=agent_name)

    # Bare "stockskis" → latest version
    if agent_type == "stockskis":
        versions = registry.stockskis_versions
        if versions:
            return registry.create(f"stockskis_v{max(versions)}", name=agent_name)

    # RL agent (checkpoint)
    ckpt_path = None
    if checkpoint:
        candidate = Path("checkpoints") / checkpoint
        root_candidate = Path("../checkpoints") / checkpoint
        if candidate.exists():
            ckpt_path = candidate
        elif root_candidate.exists():
            ckpt_path = root_candidate
    else:
        default = Path("checkpoints/tarok_agent_latest.pt")
        if default.exists():
            ckpt_path = default

    if ckpt_path:
        agent = RLAgent.from_checkpoint(ckpt_path, name=agent_name)
    else:
        agent = RLAgent(name=agent_name)
    agent.set_training(False)
    return agent


# ---- Endpoints ----

@router.post("/match")
async def tournament_match(req: TournamentMatchRequest):
    """Run N games between 4 agents and return cumulative scores."""
    agents = [_build_agent(cfg, i) for i, cfg in enumerate(req.agents[:4])]
    while len(agents) < 4:
        agents.append(RandomPlayer(name=f"Fill-{len(agents)}"))

    cumulative: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    game_results: list[dict[str, int]] = []

    for _ in range(max(1, min(req.num_games, 100))):
        loop = GameLoop(agents)
        _state, scores = await loop.run()
        for pid, pts in scores.items():
            cumulative[pid] += pts
        game_results.append({str(k): v for k, v in scores.items()})

    ranked = sorted(cumulative.items(), key=lambda x: x[1], reverse=True)
    return {
        "cumulative": {str(k): v for k, v in cumulative.items()},
        "ranked": [{"seat": k, "name": agents[k].name, "score": v} for k, v in ranked],
        "games": game_results,
    }


# ---- Multi-tournament simulation ----

async def _simulate_single_tournament(
    agent_configs: list[dict],
    games_per_round: int,
) -> dict[str, dict]:
    """Run one double-elimination tournament, return per-agent placement stats."""
    import random as _random

    padded = list(agent_configs)
    while len(padded) < 8:
        padded.append({"name": f"Random-{len(padded)}", "type": "random"})
    _random.shuffle(padded)

    async def _play_match(cfgs: list[dict]) -> list[dict]:
        """Play a match, return configs ranked best→worst."""
        agents = [_build_agent(c, i) for i, c in enumerate(cfgs[:4])]
        while len(agents) < 4:
            agents.append(RandomPlayer(name=f"Fill-{len(agents)}"))

        cumulative: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for _ in range(max(1, min(games_per_round, 100))):
            loop = GameLoop(agents)
            _state, scores = await loop.run()
            for pid, pts in scores.items():
                cumulative[pid] += pts
            await asyncio.sleep(0)

        ranked_seats = sorted(cumulative, key=lambda s: cumulative[s], reverse=True)
        return [cfgs[s] for s in ranked_seats]

    # WB R1
    wb_r1_a = await _play_match(padded[0:4])
    wb_r1_b = await _play_match(padded[4:8])

    wb_r1_top = wb_r1_a[:2] + wb_r1_b[:2]  # advance
    wb_r1_bot = wb_r1_a[2:] + wb_r1_b[2:]  # drop to losers

    # LB R1
    lb_r1 = await _play_match(wb_r1_bot)
    lb_r1_top = lb_r1[:2]

    # WB Final
    wb_final = await _play_match(wb_r1_top)
    wb_final_top = wb_final[:2]
    wb_final_bot = wb_final[2:]

    # LB Final
    lb_final_entries = lb_r1_top + wb_final_bot
    lb_final = await _play_match(lb_final_entries)
    lb_final_top = lb_final[:2]

    # Grand Final
    gf_entries = wb_final_top + lb_final_top
    gf_ranked = await _play_match(gf_entries)

    # Assign placements
    placements: dict[str, int] = {}
    for i, cfg in enumerate(gf_ranked):
        placements[cfg["name"]] = i + 1
    for i, cfg in enumerate(lb_final[2:]):
        placements[cfg["name"]] = 5 + i
    for i, cfg in enumerate(lb_r1[2:]):
        placements[cfg["name"]] = 7 + i

    return placements


@router.post("/simulate")
async def simulate_multi_tournament(req: MultiTournamentRequest):
    """Run multiple tournaments and return aggregate standings."""
    global _multi_tournament_task, _multi_tournament_progress

    if _multi_tournament_task and not _multi_tournament_task.done():
        return {"status": "already_running"}

    num = max(1, min(req.num_tournaments, 100))
    agent_configs = req.agents[:8]

    _multi_tournament_progress = {
        "status": "running",
        "current": 0,
        "total": num,
        "standings": {},
    }

    async def _run():
        global _multi_tournament_progress
        # Per-agent stats: {name: {wins, top2, total_placement, placements: [...]}}
        standings: dict[str, dict] = {}
        for cfg in agent_configs:
            standings[cfg["name"]] = {
                "name": cfg["name"],
                "type": cfg.get("type", "rl"),
                "checkpoint": cfg.get("checkpoint", ""),
                "wins": 0,
                "top2": 0,
                "top4": 0,
                "total_placement": 0,
                "tournaments_played": 0,
                "placements": [],
            }
        # Also init for random fillers
        for i in range(len(agent_configs), 8):
            fill_name = f"Random-{i}"
            standings[fill_name] = {
                "name": fill_name,
                "type": "random",
                "checkpoint": "",
                "wins": 0,
                "top2": 0,
                "top4": 0,
                "total_placement": 0,
                "tournaments_played": 0,
                "placements": [],
            }

        try:
            for t in range(num):
                try:
                    placements = await _simulate_single_tournament(agent_configs, req.games_per_round)
                except Exception:
                    log.exception("Tournament %d/%d failed, skipping", t + 1, num)
                    _multi_tournament_progress = {
                        "status": "running",
                        "current": t + 1,
                        "total": num,
                        "standings": {
                            name: {**s, "avg_placement": round(s["total_placement"] / max(s["tournaments_played"], 1), 2)}
                            for name, s in standings.items()
                        },
                    }
                    await asyncio.sleep(0)
                    continue

                for name, place in placements.items():
                    if name not in standings:
                        continue
                    s = standings[name]
                    s["tournaments_played"] += 1
                    s["total_placement"] += place
                    s["placements"].append(place)
                    if place == 1:
                        s["wins"] += 1
                    if place <= 2:
                        s["top2"] += 1
                    if place <= 4:
                        s["top4"] += 1

                _multi_tournament_progress = {
                    "status": "running",
                    "current": t + 1,
                    "total": num,
                    "standings": {
                        name: {**s, "avg_placement": round(s["total_placement"] / max(s["tournaments_played"], 1), 2)}
                        for name, s in standings.items()
                    },
                }
                await asyncio.sleep(0)

            # Final result
            final_standings = {
                name: {**s, "avg_placement": round(s["total_placement"] / max(s["tournaments_played"], 1), 2)}
                for name, s in standings.items()
            }
            _multi_tournament_progress = {
                "status": "done",
                "current": num,
                "total": num,
                "standings": final_standings,
            }
        except asyncio.CancelledError:
            log.info("Multi-tournament cancelled")
            _multi_tournament_progress["status"] = "cancelled"
        except Exception:
            log.exception("Multi-tournament failed")
            _multi_tournament_progress["status"] = "error"

    _multi_tournament_task = asyncio.create_task(_run())
    return {"status": "started", "num_tournaments": num}


@router.get("/simulate/progress")
async def multi_tournament_progress():
    if _multi_tournament_progress:
        return _multi_tournament_progress
    return {"status": "idle", "current": 0, "total": 0, "standings": {}}


@router.post("/simulate/stop")
async def stop_multi_tournament():
    global _multi_tournament_task
    if _multi_tournament_task and not _multi_tournament_task.done():
        _multi_tournament_task.cancel()
    return {"status": "stopped"}
