"""Bot Arena — mass analytics simulation via Rust self-play engine."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from tarok.adapters.api.checkpoint_utils import resolve_checkpoint

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/arena", tags=["arena"])

log = logging.getLogger(__name__)


# ---- Models ----


class ArenaRequest(BaseModel):
    agents: list[dict]  # [{name, type, checkpoint?}] — exactly 4
    total_games: int = 100000
    session_size: int = 50  # games per session for progress tracking


# ---- Globals ----

_arena_task: asyncio.Task | None = None
_arena_progress: dict | None = None
_arena_history_path = Path(__file__).resolve().parents[5] / "data" / "arena_results.json"
_ARENA_ROOT_CKPT_DIR = Path("../data/checkpoints")


def _arena_resolve_checkpoint(token: str) -> str | None:
    """Resolve a checkpoint token to a path string, using shared logic."""
    path = resolve_checkpoint(token)
    return str(path) if path else None


# Contract name lookup (must match engine-rs Contract enum order)
_ARENA_CONTRACT_NAMES = [
    "KLOP",
    "THREE",
    "TWO",
    "ONE",
    "SOLO_THREE",
    "SOLO_TWO",
    "SOLO_ONE",
    "SOLO",
    "BERAC",
    "BARVNI_VALAT",
]
_VALAT_CONTRACT_IDX = 9  # BARVNI_VALAT


def _serialize_trace(t: dict) -> dict:
    """Convert a raw numpy trace dict to JSON-serialisable form."""
    return {
        "bids": [list(b) for b in t["bids"]],
        "king_call": list(t["king_call"]) if t["king_call"] is not None else None,
        "talon_pick": list(t["talon_pick"]) if t["talon_pick"] is not None else None,
        "put_down": list(t["put_down"]),
        "cards_played": [list(c) for c in t["cards_played"]],
        "dealer": int(t["dealer"]),
    }


# ---- History persistence ----


def _load_arena_history() -> dict:
    if not _arena_history_path.exists():
        return {"runs": []}
    try:
        with open(_arena_history_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"runs": []}
        runs = data.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        return {"runs": runs}
    except Exception:
        return {"runs": []}


def _save_arena_history(data: dict) -> None:
    _arena_history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _arena_history_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(_arena_history_path)


def _persist_arena_run(
    req_agents: list[dict], total_games: int, session_size: int, payload: dict
) -> None:
    analytics = payload.get("analytics") or {}
    checkpoints = sorted(
        {
            str(a.get("checkpoint", "")).strip()
            for a in req_agents
            if str(a.get("type", "")).strip().lower() == "rl"
            and str(a.get("checkpoint", "")).strip()
        }
    )
    run = {
        "run_id": f"arena-{time.time_ns()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": payload.get("status", "done"),
        "games_done": int(payload.get("games_done", 0)),
        "total_games": int(total_games),
        "session_size": int(session_size),
        "agents": [
            {
                "name": str(a.get("name", "")),
                "type": str(a.get("type", "")),
                "checkpoint": str(a.get("checkpoint", "")),
            }
            for a in req_agents
        ],
        "checkpoints": checkpoints,
        "analytics": analytics,
    }
    history = _load_arena_history()
    runs = history.get("runs", [])
    runs.append(run)
    # Keep recent history bounded.
    if len(runs) > 200:
        runs = runs[-200:]
    history["runs"] = runs
    _save_arena_history(history)


# ---- Leaderboard ----


def _build_checkpoint_leaderboard(runs: list[dict]) -> list[dict]:
    agg: dict[str, dict] = {}

    for run in runs:
        analytics = run.get("analytics") or {}
        players = analytics.get("players") or []
        agents = run.get("agents") or []

        for idx, player in enumerate(players):
            if idx >= len(agents):
                continue
            agent = agents[idx]
            if str(agent.get("type", "")).strip().lower() != "rl":
                continue

            checkpoint = str(agent.get("checkpoint", "")).strip() or "latest"
            games_played = int(player.get("games_played", 0) or 0)
            declared_count = int(player.get("declared_count", 0) or 0)
            declared_won = int(player.get("declared_won", 0) or 0)
            declared_lost = max(0, declared_count - declared_won)

            row = agg.setdefault(
                checkpoint,
                {
                    "checkpoint": checkpoint,
                    "appearances": 0,
                    "runs": set(),
                    "games": 0,
                    "placement_weighted": 0.0,
                    "bid_wins": 0,
                    "taroks_weighted": 0.0,
                    "declared_games": 0,
                    "declared_wins": 0,
                    "declared_win_score_weighted": 0.0,
                    "declared_loss_score_weighted": 0.0,
                    "times_called": 0,
                    "latest_run_at": "",
                },
            )

            row["appearances"] += 1
            row["runs"].add(str(run.get("run_id", "")))
            row["games"] += games_played
            row["placement_weighted"] += (
                float(player.get("avg_placement", 0.0) or 0.0) * games_played
            )
            row["bid_wins"] += int(player.get("bid_won_count", 0) or 0)
            row["taroks_weighted"] += (
                float(player.get("avg_taroks_in_hand", 0.0) or 0.0) * games_played
            )
            row["declared_games"] += declared_count
            row["declared_wins"] += declared_won
            row["declared_win_score_weighted"] += (
                float(player.get("avg_declared_win_score", 0.0) or 0.0) * declared_won
            )
            row["declared_loss_score_weighted"] += (
                float(player.get("avg_declared_loss_score", 0.0) or 0.0) * declared_lost
            )
            row["times_called"] += int(player.get("times_called", 0) or 0)
            created_at = str(run.get("created_at", ""))
            if created_at > row["latest_run_at"]:
                row["latest_run_at"] = created_at

    rows: list[dict] = []
    for checkpoint, data in agg.items():
        games = max(1, data["games"])
        declared_games = data["declared_games"]
        declared_wins = data["declared_wins"]
        declared_lost = max(0, declared_games - declared_wins)

        rows.append(
            {
                "checkpoint": checkpoint,
                "appearances": data["appearances"],
                "runs": len(data["runs"]),
                "games": data["games"],
                "avg_placement": round(data["placement_weighted"] / games, 3),
                "bid_wins": data["bid_wins"],
                "bid_win_rate_per_game": round(data["bid_wins"] / games * 100, 2),
                "avg_taroks_in_hand": round(data["taroks_weighted"] / games, 3),
                "declared_games": declared_games,
                "declared_win_rate": round((declared_wins / max(1, declared_games)) * 100, 2),
                "avg_declared_win_score": round(
                    data["declared_win_score_weighted"] / max(1, declared_wins), 3
                ),
                "avg_declared_loss_score": round(
                    data["declared_loss_score_weighted"] / max(1, declared_lost), 3
                ),
                "times_called": data["times_called"],
                "latest_run_at": data["latest_run_at"],
            }
        )

    rows.sort(key=lambda r: (r["avg_placement"], -r["games"], r["checkpoint"]))
    return rows


# ---- Seat mapping / TorchScript export ----


def _agent_type_to_seat_label(agent_type: str) -> str | None:
    """Map frontend agent type string to Rust seat_config label."""
    t = agent_type.strip().lower()
    if t in ("stockskis", "stockskis_v5"):
        return "bot_v5"
    if t == "stockskis_v6":
        return "bot_v6"
    if t in ("stockskis_m6", "bot_m6"):
        return "bot_m6"
    if t == "rl":
        return "nn"
    return None


def _export_checkpoint_to_torchscript(checkpoint_path: str) -> str:
    """Load a regular PyTorch checkpoint and export a TorchScript model."""
    import tempfile
    import torch
    from tarok_model.network import TarokNetV4

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    model_arch = ckpt.get("model_arch")
    if model_arch != "v4":
        raise ValueError(
            f"Unsupported checkpoint architecture '{model_arch}'. Only 'v4' checkpoints are supported."
        )
    hidden_size = sd["shared.0.weight"].shape[0]
    has_oracle = any(k.startswith("critic_backbone.") for k in sd)

    net = TarokNetV4(hidden_size, oracle_critic=has_oracle)
    net.load_state_dict(sd, strict=True)
    net.eval()

    class _AllHeads(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor):
            h = self.base.shared(x)
            h = self.base.res_blocks(h)
            cf = self.base._extract_card_features(x)
            a = self.base.card_attention(cf)
            h = self.base.fuse(torch.cat([h, a], dim=-1))
            return (
                self.base.bid_head(h),
                self.base.king_head(h),
                self.base.talon_head(h),
                self.base.card_head(h),
                self.base.critic(h).squeeze(-1),
            )

    wrapper = _AllHeads(net)
    wrapper.eval()
    traced = torch.jit.trace(wrapper, torch.randn(1, 450), check_trace=False)

    path = tempfile.mktemp(suffix=".pt", prefix="tarok_arena_ts_")
    traced.save(path)
    return path


# ---- Analytics builder ----


def _build_arena_analytics(
    player_stats,
    contract_stats,
    taroks_per_contract,
    games_done,
    total_games,
    session_size=50,
    notable_games=None,
):
    """Build the analytics payload from accumulated stats."""
    players = []
    for ps in player_stats:
        gp = max(ps["games_played"], 1)
        n_sessions = max(gp / session_size, 1)
        players.append(
            {
                "name": ps["name"],
                "type": ps["type"],
                "games_played": ps["games_played"],
                "total_score": ps["total_score"],
                "avg_score": round(ps["total_score"] / n_sessions, 1),
                "placements": ps["placements"],
                "avg_placement": round(ps["placement_sum"] / max(n_sessions, 1), 2),
                "win_rate": round(ps["wins"] / max(n_sessions, 1) * 100, 2),
                "positive_rate": round(ps["positive_games"] / gp * 100, 2),
                "bids_made": ps["bids_made"],
                "declared_count": ps["declared_count"],
                "declared_won": ps["declared_won"],
                "bid_won_count": ps["bid_won_count"],
                "declared_win_rate": round(
                    ps["declared_won"] / max(ps["declared_count"], 1) * 100, 2
                ),
                "avg_declared_win_score": round(
                    ps["declared_win_score_total"] / max(ps["declared_win_games"], 1), 2
                ),
                "avg_declared_loss_score": round(
                    ps["declared_loss_score_total"] / max(ps["declared_loss_games"], 1), 2
                ),
                "defended_count": ps["defended_count"],
                "defended_won": ps["defended_won"],
                "defended_win_rate": round(
                    ps["defended_won"] / max(ps["defended_count"], 1) * 100, 2
                ),
                "announcements_made": {},
                "kontra_count": 0,
                "times_called": ps["times_called"],
                "avg_taroks_in_hand": round(
                    ps["taroks_in_hand_total"] / max(ps["taroks_in_hand_count"], 1), 2
                ),
                "best_game": {
                    "score": ps["best_game_score"],
                    "game_idx": ps["best_game_idx"],
                    "hands": ps["best_game_hands"],
                    "talon": ps["best_game_talon"],
                    "trace": ps["best_game_trace"],
                },
                "worst_game": {
                    "score": ps["worst_game_score"],
                    "game_idx": ps["worst_game_idx"],
                    "hands": ps["worst_game_hands"],
                    "talon": ps["worst_game_talon"],
                    "trace": ps["worst_game_trace"],
                },
                "avg_win_score": round(
                    ps["positive_score_total"] / max(ps["positive_games"], 1), 2
                ),
                "avg_loss_score": round(
                    ps["negative_score_total"] / max(ps["negative_game_count"], 1), 2
                ),
                "score_history": ps["score_history"],
                "taroks_per_contract": {
                    name: round(tc["total_taroks"] / max(tc["count"], 1), 2)
                    for name, tc in ps["taroks_per_contract"].items()
                },
                "contract_stats": {
                    name: {
                        "declared": cs["declared"],
                        "won": cs["won"],
                        "win_rate": round(cs["won"] / max(cs["declared"], 1) * 100, 2),
                        "avg_score": round(cs["total_score"] / max(cs["declared"], 1), 2),
                    }
                    for name, cs in ps["contract_stats"].items()
                },
            }
        )

    contracts = {}
    for name, cs in contract_stats.items():
        played = max(cs["played"], 1)
        contracts[name] = {
            "played": cs["played"],
            "decl_win_rate": round(cs["decl_won"] / played * 100, 2),
            "avg_decl_score": round(cs["total_decl_score"] / played, 2),
            "avg_def_score": round(cs["total_def_score"] / played, 2),
        }

    tpc = {}
    for name, tc in taroks_per_contract.items():
        cnt = max(tc["count"], 1)
        tpc[name] = round(tc["total_taroks"] / cnt, 2)

    return {
        "games_done": games_done,
        "total_games": total_games,
        "players": players,
        "contracts": contracts,
        "taroks_per_contract": tpc,
        "notable_games": notable_games,
    }


# ---- Endpoints ----


@router.get("/history")
async def arena_history(checkpoint: str | None = None):
    history = _load_arena_history()
    runs = history.get("runs", [])
    if checkpoint:
        ck = checkpoint.strip()
        runs = [r for r in runs if ck in (r.get("checkpoints") or [])]
    return {"runs": runs}


@router.get("/leaderboard/checkpoints")
async def arena_checkpoint_leaderboard():
    history = _load_arena_history()
    runs = history.get("runs", [])
    leaderboard = _build_checkpoint_leaderboard(runs)
    return {"leaderboard": leaderboard}


@router.post("/start")
async def start_arena(req: ArenaRequest):
    """Run a large-scale bot arena via Rust self-play engine.

    All seat types (bot_v5, bot_v6, bot_m6, nn) go through run_self_play
    which supports any mix of bot and NN seats.
    """
    global _arena_task, _arena_progress

    if _arena_task and not _arena_task.done():
        return {"status": "already_running"}

    total = max(1, min(req.total_games, 500_000))
    session_size = max(1, min(req.session_size, 1000))
    agent_configs = req.agents[:4]
    while len(agent_configs) < 4:
        agent_configs.append({"name": f"StockŠkis-{len(agent_configs)}", "type": "stockskis"})

    # Map agent types to Rust seat labels.
    seat_labels = []
    agent_names = []
    agent_types_raw = []
    has_nn = False
    nn_checkpoint_path: str | None = None
    for i, cfg in enumerate(agent_configs):
        atype = str(cfg.get("type", "stockskis")).strip().lower()
        aname = cfg.get("name", f"Agent-{i}")
        label = _agent_type_to_seat_label(atype)
        if label is None:
            return {
                "status": "error",
                "message": f"Agent '{aname}' type '{atype}' is not supported. Use stockskis / stockskis_v5 / stockskis_v6 / stockskis_m6 / rl.",
            }
        if label == "nn":
            has_nn = True
            ckpt = cfg.get("checkpoint", "")
            if ckpt:
                nn_checkpoint_path = _arena_resolve_checkpoint(ckpt)
            if nn_checkpoint_path is None:
                # Fall back to training_run persona
                fallback = _ARENA_ROOT_CKPT_DIR / "training_run" / "_current.pt"
                if fallback.exists():
                    nn_checkpoint_path = str(fallback)
        seat_labels.append(label)
        agent_names.append(aname)
        agent_types_raw.append(atype)

    seat_config = ",".join(seat_labels)

    # For NN: export checkpoint to TorchScript
    ts_model_path: str | None = None
    if has_nn:
        if nn_checkpoint_path is None:
            return {
                "status": "error",
                "message": "No checkpoint found for RL agent. Place a checkpoint in checkpoints/ or select one.",
            }
        try:
            ts_model_path = _export_checkpoint_to_torchscript(nn_checkpoint_path)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to export checkpoint to TorchScript: {e}",
            }

    _arena_progress = {
        "status": "running",
        "games_done": 0,
        "total_games": total,
        "analytics": None,
    }

    async def _run_arena():
        global _arena_progress
        import numpy as np

        try:
            import tarok_engine as te  # type: ignore[assignment]
        except ImportError:
            log.error("tarok_engine not available — cannot run arena")
            _arena_progress["status"] = "error"
            return

        # Per-player accumulators
        player_stats = []
        for i in range(4):
            player_stats.append(
                {
                    "name": agent_names[i],
                    "type": agent_types_raw[i],
                    "total_score": 0,
                    "games_played": 0,
                    "placements": {1: 0, 2: 0, 3: 0, 4: 0},
                    "placement_sum": 0.0,
                    "wins": 0.0,
                    "positive_games": 0,
                    "bids_made": {},
                    "declared_count": 0,
                    "declared_won": 0,
                    "bid_won_count": 0,
                    "declared_win_score_total": 0,
                    "declared_loss_score_total": 0,
                    "declared_win_games": 0,
                    "declared_loss_games": 0,
                    "defended_count": 0,
                    "defended_won": 0,
                    "times_called": 0,
                    "taroks_in_hand_total": 0,
                    "taroks_in_hand_count": 0,
                    "positive_score_total": 0,
                    "negative_game_count": 0,
                    "negative_score_total": 0,
                    "best_game_score": None,
                    "worst_game_score": None,
                    "best_game_idx": None,
                    "worst_game_idx": None,
                    "best_game_hands": None,
                    "best_game_talon": None,
                    "best_game_trace": None,
                    "worst_game_hands": None,
                    "worst_game_talon": None,
                    "worst_game_trace": None,
                    "score_history": [],
                    "taroks_per_contract": {},
                    "contract_stats": {},
                }
            )
        contract_stats: dict = {}
        taroks_per_contract: dict = {}
        notable_games: dict = {
            "best_non_valat": None,
            "worst_non_valat": None,
            "best_valat": None,
            "worst_valat": None,
            "by_contract": {},
        }
        games_done = 0
        sessions_played = 0
        session_scores = [0, 0, 0, 0]
        games_in_session = 0

        def _accumulate_scores(
            scores: "np.ndarray", n_batch: int, initial_hands=None, initial_talon=None, traces=None
        ) -> None:
            nonlocal games_in_session, session_scores, sessions_played

            for pid in range(4):
                col = scores[:, pid]
                ps = player_stats[pid]
                ps["total_score"] += int(col.sum())
                ps["games_played"] += n_batch

                pos_mask = col > 0
                neg_mask = ~pos_mask
                n_pos = int(pos_mask.sum())
                n_neg = int(neg_mask.sum())
                ps["positive_games"] += n_pos
                ps["positive_score_total"] += int(col[pos_mask].sum()) if n_pos else 0
                ps["negative_game_count"] += n_neg
                ps["negative_score_total"] += int(col[neg_mask].sum()) if n_neg else 0

                batch_max_idx = int(col.argmax())
                batch_min_idx = int(col.argmin())
                batch_max = int(col[batch_max_idx])
                batch_min = int(col[batch_min_idx])
                if ps["best_game_score"] is None or batch_max > ps["best_game_score"]:
                    ps["best_game_score"] = batch_max
                    ps["best_game_idx"] = games_done + batch_max_idx
                    if initial_hands is not None and initial_talon is not None:
                        ps["best_game_hands"] = initial_hands[batch_max_idx].tolist()
                        ps["best_game_talon"] = initial_talon[batch_max_idx].tolist()
                    if traces is not None:
                        ps["best_game_trace"] = _serialize_trace(traces[batch_max_idx])
                if ps["worst_game_score"] is None or batch_min < ps["worst_game_score"]:
                    ps["worst_game_score"] = batch_min
                    ps["worst_game_idx"] = games_done + batch_min_idx
                    if initial_hands is not None and initial_talon is not None:
                        ps["worst_game_hands"] = initial_hands[batch_min_idx].tolist()
                        ps["worst_game_talon"] = initial_talon[batch_min_idx].tolist()
                    if traces is not None:
                        ps["worst_game_trace"] = _serialize_trace(traces[batch_min_idx])

            # Session tracking: score_history + placement
            batch_offset = 0
            while batch_offset < n_batch:
                remaining = session_size - games_in_session
                chunk_end = min(batch_offset + remaining, n_batch)
                chunk = scores[batch_offset:chunk_end]
                for pid in range(4):
                    session_scores[pid] += int(chunk[:, pid].sum())
                games_in_session += chunk_end - batch_offset
                batch_offset = chunk_end
                if games_in_session >= session_size:
                    indexed = sorted(range(4), key=lambda p: session_scores[p], reverse=True)
                    for rank_idx, pid in enumerate(indexed):
                        place = rank_idx + 1
                        player_stats[pid]["placements"][place] += 1
                        player_stats[pid]["placement_sum"] += place
                        if place == 1:
                            player_stats[pid]["wins"] += 1.0
                    sessions_played += 1
                    for pid in range(4):
                        player_stats[pid]["score_history"].append(session_scores[pid])
                    session_scores = [0, 0, 0, 0]
                    games_in_session = 0

        batch_size = min(2_000 if has_nn else 10_000, total)

        def _process_batch_analytics(
            scores,
            contracts,
            declarers,
            partners,
            bid_contracts,
            taroks_in_hand,
            n_batch,
            initial_hands=None,
            initial_talon=None,
            traces=None,
        ):
            _accumulate_scores(
                scores,
                n_batch,
                initial_hands=initial_hands,
                initial_talon=initial_talon,
                traces=traces,
            )

            # Taroks in hand
            if taroks_in_hand is not None:
                for pid in range(4):
                    player_stats[pid]["taroks_in_hand_total"] += int(taroks_in_hand[:, pid].sum())
                    player_stats[pid]["taroks_in_hand_count"] += n_batch

            # Taroks per contract
            if taroks_in_hand is not None and declarers is not None and contracts is not None:
                non_klop = declarers >= 0
                if non_klop.any():
                    nk_idx = np.where(non_klop)[0]
                    nk_decl = declarers[nk_idx]
                    nk_contracts = contracts[nk_idx]
                    nk_taroks = taroks_in_hand[nk_idx, nk_decl]
                    for cu8 in np.unique(nk_contracts):
                        cname = (
                            _ARENA_CONTRACT_NAMES[int(cu8)]
                            if int(cu8) < len(_ARENA_CONTRACT_NAMES)
                            else "UNKNOWN"
                        )
                        c_mask = nk_contracts == cu8
                        if cname not in taroks_per_contract:
                            taroks_per_contract[cname] = {"total_taroks": 0, "count": 0}
                        taroks_per_contract[cname]["total_taroks"] += int(nk_taroks[c_mask].sum())
                        taroks_per_contract[cname]["count"] += int(c_mask.sum())

                    for pid in range(4):
                        pid_mask = nk_decl == pid
                        if not pid_mask.any():
                            continue
                        pid_contracts = nk_contracts[pid_mask]
                        pid_taroks = nk_taroks[pid_mask]
                        ptpc = player_stats[pid]["taroks_per_contract"]
                        for cu8 in np.unique(pid_contracts):
                            cname = (
                                _ARENA_CONTRACT_NAMES[int(cu8)]
                                if int(cu8) < len(_ARENA_CONTRACT_NAMES)
                                else "UNKNOWN"
                            )
                            c_mask = pid_contracts == cu8
                            if cname not in ptpc:
                                ptpc[cname] = {"total_taroks": 0, "count": 0}
                            ptpc[cname]["total_taroks"] += int(pid_taroks[c_mask].sum())
                            ptpc[cname]["count"] += int(c_mask.sum())

            # Bid tracking
            if bid_contracts is not None:
                for pid in range(4):
                    bids_col = bid_contracts[:, pid]
                    valid_mask = bids_col >= 0
                    if valid_mask.any():
                        unique, counts = np.unique(bids_col[valid_mask], return_counts=True)
                        bids = player_stats[pid]["bids_made"]
                        for b, c in zip(unique, counts):
                            bname = (
                                _ARENA_CONTRACT_NAMES[int(b)]
                                if int(b) < len(_ARENA_CONTRACT_NAMES)
                                else f"CONTRACT_{b}"
                            )
                            bids[bname] = bids.get(bname, 0) + int(c)

            # Contract play counts
            unique_contracts, contract_counts = np.unique(contracts, return_counts=True)
            for cu8, cnt in zip(unique_contracts, contract_counts):
                cname = (
                    _ARENA_CONTRACT_NAMES[int(cu8)]
                    if int(cu8) < len(_ARENA_CONTRACT_NAMES)
                    else "UNKNOWN"
                )
                if cname not in contract_stats:
                    contract_stats[cname] = {
                        "played": 0,
                        "decl_won": 0,
                        "total_decl_score": 0,
                        "total_def_score": 0,
                    }
                contract_stats[cname]["played"] += int(cnt)

            # Declarer/defender stats for non-Klop games
            non_klop_mask = declarers >= 0
            if non_klop_mask.any():
                nk_idx = np.where(non_klop_mask)[0]
                nk_decl = declarers[nk_idx]
                nk_part = partners[nk_idx]
                nk_scores = scores[nk_idx]
                nk_contracts = contracts[nk_idx]

                decl_scores_arr = nk_scores[np.arange(len(nk_idx)), nk_decl]
                decl_won_mask = decl_scores_arr > 0

                has_partner = nk_part >= 0
                partner_scores_arr = np.zeros(len(nk_idx), dtype=np.int32)
                if has_partner.any():
                    hp_idx = np.where(has_partner)[0]
                    partner_scores_arr[hp_idx] = nk_scores[hp_idx, nk_part[hp_idx]]

                def_total_arr = nk_scores.sum(axis=1) - decl_scores_arr - partner_scores_arr

                for pid in range(4):
                    ps = player_stats[pid]
                    is_decl = nk_decl == pid
                    ps["bid_won_count"] += int(is_decl.sum())

                    d_won = is_decl & decl_won_mask
                    d_lost = is_decl & ~decl_won_mask
                    ps["declared_win_games"] += int(d_won.sum())
                    ps["declared_win_score_total"] += (
                        int(decl_scores_arr[d_won].sum()) if d_won.any() else 0
                    )
                    ps["declared_loss_games"] += int(d_lost.sum())
                    ps["declared_loss_score_total"] += (
                        int(decl_scores_arr[d_lost].sum()) if d_lost.any() else 0
                    )

                    ps["times_called"] += int((nk_part == pid).sum())

                    in_team = is_decl | (nk_part == pid)
                    ps["declared_count"] += int(in_team.sum())
                    ps["declared_won"] += int((in_team & decl_won_mask).sum())
                    is_def = ~in_team
                    ps["defended_count"] += int(is_def.sum())
                    ps["defended_won"] += int((is_def & ~decl_won_mask).sum())

                    if is_decl.any():
                        pid_contracts = nk_contracts[is_decl]
                        pid_scores = decl_scores_arr[is_decl]
                        pcs = ps["contract_stats"]
                        for cu8 in np.unique(pid_contracts):
                            cname = (
                                _ARENA_CONTRACT_NAMES[int(cu8)]
                                if int(cu8) < len(_ARENA_CONTRACT_NAMES)
                                else "UNKNOWN"
                            )
                            c_mask = pid_contracts == cu8
                            c_scores = pid_scores[c_mask]
                            if cname not in pcs:
                                pcs[cname] = {"declared": 0, "won": 0, "total_score": 0}
                            pcs[cname]["declared"] += int(c_mask.sum())
                            pcs[cname]["won"] += int((c_scores > 0).sum())
                            pcs[cname]["total_score"] += int(c_scores.sum())

                for cu8 in np.unique(nk_contracts):
                    cname = (
                        _ARENA_CONTRACT_NAMES[int(cu8)]
                        if int(cu8) < len(_ARENA_CONTRACT_NAMES)
                        else "UNKNOWN"
                    )
                    cs = contract_stats[cname]
                    c_mask = nk_contracts == cu8
                    c_decl = decl_scores_arr[c_mask]
                    cs["decl_won"] += int((c_decl > 0).sum())
                    cs["total_decl_score"] += int(c_decl.sum())
                    cs["total_def_score"] += int(def_total_arr[c_mask].sum())

            # ---- Notable games (valat / non-valat / by-contract) ----
            game_max_scores = np.max(scores, axis=1)
            game_min_scores = np.min(scores, axis=1)
            game_max_pids = np.argmax(scores, axis=1)
            game_min_pids = np.argmin(scores, axis=1)
            valat_mask = (contracts == _VALAT_CONTRACT_IDX) | (
                np.max(np.abs(scores), axis=1) >= 250
            )

            def _make_notable(gi: int, pid: int, sc: int) -> dict:
                cu = int(contracts[gi])
                return {
                    "score": sc,
                    "game_idx": games_done + gi,
                    "player_idx": pid,
                    "player_name": agent_names[pid],
                    "contract": _ARENA_CONTRACT_NAMES[cu]
                    if cu < len(_ARENA_CONTRACT_NAMES)
                    else "UNKNOWN",
                    "hands": initial_hands[gi].tolist() if initial_hands is not None else None,
                    "talon": initial_talon[gi].tolist() if initial_talon is not None else None,
                    "trace": _serialize_trace(traces[gi]) if traces is not None else None,
                }

            # Best / worst non-valat
            nv = ~valat_mask
            if nv.any():
                nv_idx = np.where(nv)[0]
                bi = nv_idx[int(game_max_scores[nv_idx].argmax())]
                wi = nv_idx[int(game_min_scores[nv_idx].argmin())]
                bs, ws = int(game_max_scores[bi]), int(game_min_scores[wi])
                if (
                    notable_games["best_non_valat"] is None
                    or bs > notable_games["best_non_valat"]["score"]
                ):
                    notable_games["best_non_valat"] = _make_notable(
                        int(bi), int(game_max_pids[bi]), bs
                    )
                if (
                    notable_games["worst_non_valat"] is None
                    or ws < notable_games["worst_non_valat"]["score"]
                ):
                    notable_games["worst_non_valat"] = _make_notable(
                        int(wi), int(game_min_pids[wi]), ws
                    )

            # Best / worst valat
            vm = valat_mask
            if vm.any():
                v_idx = np.where(vm)[0]
                bi = v_idx[int(game_max_scores[v_idx].argmax())]
                wi = v_idx[int(game_min_scores[v_idx].argmin())]
                bs, ws = int(game_max_scores[bi]), int(game_min_scores[wi])
                if notable_games["best_valat"] is None or bs > notable_games["best_valat"]["score"]:
                    notable_games["best_valat"] = _make_notable(int(bi), int(game_max_pids[bi]), bs)
                if (
                    notable_games["worst_valat"] is None
                    or ws < notable_games["worst_valat"]["score"]
                ):
                    notable_games["worst_valat"] = _make_notable(
                        int(wi), int(game_min_pids[wi]), ws
                    )

            # Best sample per contract type
            for cu8 in np.unique(contracts):
                cname = (
                    _ARENA_CONTRACT_NAMES[int(cu8)]
                    if int(cu8) < len(_ARENA_CONTRACT_NAMES)
                    else "UNKNOWN"
                )
                c_idx = np.where(contracts == cu8)[0]
                ci = c_idx[int(game_max_scores[c_idx].argmax())]
                sc = int(game_max_scores[ci])
                prev = notable_games["by_contract"].get(cname)
                if prev is None or sc > prev["score"]:
                    notable_games["by_contract"][cname] = _make_notable(
                        int(ci), int(game_max_pids[ci]), sc
                    )

        try:
            while games_done < total:
                n_batch = min(batch_size, total - games_done)

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda n=n_batch: te.run_self_play(
                        n_games=n,
                        concurrency=min(64, n),
                        model_path=ts_model_path,
                        explore_rate=0.0,
                        seat_config=seat_config,
                        include_replay_data=True,
                    ),
                )

                scores = np.asarray(result["scores"])
                contracts = np.asarray(result["contracts"])
                declarers = np.asarray(result["declarers"])
                partners = np.asarray(result["partners"])
                bid_contracts = (
                    np.asarray(result["bid_contracts"]) if "bid_contracts" in result else None
                )
                taroks_in_hand = (
                    np.asarray(result["taroks_in_hand"]) if "taroks_in_hand" in result else None
                )
                batch_initial_hands = (
                    np.asarray(result["initial_hands"]) if "initial_hands" in result else None
                )
                batch_initial_talon = (
                    np.asarray(result["initial_talon"]) if "initial_talon" in result else None
                )
                batch_traces = result.get("traces")

                _process_batch_analytics(
                    scores,
                    contracts,
                    declarers,
                    partners,
                    bid_contracts,
                    taroks_in_hand,
                    n_batch,
                    initial_hands=batch_initial_hands,
                    initial_talon=batch_initial_talon,
                    traces=batch_traces,
                )

                games_done += n_batch
                # Flush last partial session
                if games_done == total and games_in_session > 0:
                    indexed = sorted(range(4), key=lambda p: session_scores[p], reverse=True)
                    for rank_idx, pid in enumerate(indexed):
                        place = rank_idx + 1
                        player_stats[pid]["placements"][place] += 1
                        player_stats[pid]["placement_sum"] += place
                        if place == 1:
                            player_stats[pid]["wins"] += 1.0
                    sessions_played += 1
                    for pid in range(4):
                        player_stats[pid]["score_history"].append(session_scores[pid])

                analytics = _build_arena_analytics(
                    player_stats,
                    contract_stats,
                    taroks_per_contract,
                    games_done,
                    total,
                    session_size,
                    notable_games,
                )
                _arena_progress = {
                    "status": "running",
                    "games_done": games_done,
                    "total_games": total,
                    "analytics": analytics,
                }
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            log.info("Arena cancelled at game %d/%d", games_done, total)
            _arena_progress["status"] = "cancelled"
            _arena_progress["analytics"] = _build_arena_analytics(
                player_stats,
                contract_stats,
                taroks_per_contract,
                games_done,
                total,
                session_size,
                notable_games,
            )
            _persist_arena_run(req.agents[:4], total, session_size, _arena_progress)
            return
        except Exception:
            log.exception("Arena failed at game %d/%d", games_done, total)
            _arena_progress["status"] = "error"
            _arena_progress["analytics"] = _build_arena_analytics(
                player_stats,
                contract_stats,
                taroks_per_contract,
                games_done,
                total,
                session_size,
                notable_games,
            )
            _persist_arena_run(req.agents[:4], total, session_size, _arena_progress)
            return
        finally:
            if ts_model_path:
                import os

                try:
                    os.unlink(ts_model_path)
                except OSError:
                    pass

        _arena_progress = {
            "status": "done",
            "games_done": games_done,
            "total_games": total,
            "analytics": _build_arena_analytics(
                player_stats,
                contract_stats,
                taroks_per_contract,
                games_done,
                total,
                session_size,
                notable_games,
            ),
        }
        _persist_arena_run(req.agents[:4], total, session_size, _arena_progress)

    _arena_task = asyncio.create_task(_run_arena())
    return {"status": "started", "total_games": total, "session_size": session_size}


@router.get("/progress")
async def arena_progress():
    if _arena_progress:
        return _arena_progress
    history = _load_arena_history()
    runs = history.get("runs", [])
    if runs:
        last = runs[-1]
        return {
            "status": last.get("status", "done"),
            "games_done": last.get("games_done", 0),
            "total_games": last.get("total_games", 0),
            "analytics": last.get("analytics"),
        }
    return {"status": "idle", "games_done": 0, "total_games": 0, "analytics": None}


@router.post("/stop")
async def stop_arena():
    global _arena_task
    if _arena_task and not _arena_task.done():
        _arena_task.cancel()
    return {"status": "stopped"}
