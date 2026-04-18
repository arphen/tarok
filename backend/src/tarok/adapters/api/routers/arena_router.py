"""Bot Arena — HTTP endpoints and async run-loop orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from tarok.adapters import arena_history
from tarok.adapters.api.checkpoint_utils import resolve_checkpoint
from tarok.use_cases.arena import (
    VALAT_CONTRACT_IDX,
    agent_type_to_seat_label,
    build_analytics,
    build_checkpoint_leaderboard,
    contract_name,
    export_checkpoint_to_torchscript,
    serialize_trace,
)

router = APIRouter(prefix="/api/arena", tags=["arena"])
log = logging.getLogger(__name__)

_ARENA_ROOT_CKPT_DIR = Path("../data/checkpoints")

_arena_task: asyncio.Task | None = None
_arena_progress: dict | None = None


class ArenaRequest(BaseModel):
    agents: list[dict]  # [{name, type, checkpoint?}] — exactly 4
    total_games: int = 100000
    session_size: int = 50  # games per session for progress tracking


# ---- Endpoints ----


@router.get("/history")
async def arena_history_endpoint(checkpoint: str | None = None):
    history = arena_history.load()
    runs = history.get("runs", [])
    if checkpoint:
        ck = checkpoint.strip()
        runs = [r for r in runs if ck in (r.get("checkpoints") or [])]
    return {"runs": runs}


@router.get("/leaderboard/checkpoints")
async def arena_checkpoint_leaderboard():
    history = arena_history.load()
    return {"leaderboard": build_checkpoint_leaderboard(history.get("runs", []))}


@router.get("/progress")
async def arena_progress():
    if _arena_progress:
        return _arena_progress
    history = arena_history.load()
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


@router.post("/start")
async def start_arena(req: ArenaRequest):
    """Run a large-scale bot arena via Rust self-play engine."""
    global _arena_task, _arena_progress

    if _arena_task and not _arena_task.done():
        return {"status": "already_running"}

    total = max(1, min(req.total_games, 500_000))
    session_size = max(1, min(req.session_size, 1000))
    agent_configs = req.agents[:4]
    while len(agent_configs) < 4:
        agent_configs.append({"name": f"StockŠkis-{len(agent_configs)}", "type": "stockskis"})

    seat_labels = []
    agent_names = []
    agent_types_raw = []
    has_nn = False
    nn_checkpoint_path: str | None = None

    for i, cfg in enumerate(agent_configs):
        atype = str(cfg.get("type", "stockskis")).strip().lower()
        aname = cfg.get("name", f"Agent-{i}")
        label = agent_type_to_seat_label(atype)
        if label is None:
            return {
                "status": "error",
                "message": f"Agent '{aname}' type '{atype}' is not supported. Use stockskis / stockskis_v5 / stockskis_v6 / stockskis_m6 / rl.",
            }
        if label == "nn":
            has_nn = True
            ckpt = cfg.get("checkpoint", "")
            if ckpt:
                path = resolve_checkpoint(ckpt)
                nn_checkpoint_path = str(path) if path else None
            if nn_checkpoint_path is None:
                fallback = _ARENA_ROOT_CKPT_DIR / "training_run" / "_current.pt"
                if fallback.exists():
                    nn_checkpoint_path = str(fallback)
        seat_labels.append(label)
        agent_names.append(aname)
        agent_types_raw.append(atype)

    seat_config = ",".join(seat_labels)

    ts_model_path: str | None = None
    if has_nn:
        if nn_checkpoint_path is None:
            return {"status": "error", "message": "No checkpoint found for RL agent."}
        try:
            ts_model_path = export_checkpoint_to_torchscript(nn_checkpoint_path)
        except Exception as e:
            return {"status": "error", "message": f"Failed to export checkpoint: {e}"}

    _arena_progress = {
        "status": "running",
        "games_done": 0,
        "total_games": total,
        "analytics": None,
    }

    _arena_task = asyncio.create_task(
        _run_arena(
            total=total,
            session_size=session_size,
            agent_configs=agent_configs,
            agent_names=agent_names,
            agent_types_raw=agent_types_raw,
            seat_config=seat_config,
            has_nn=has_nn,
            ts_model_path=ts_model_path,
        )
    )
    return {"status": "started", "total_games": total, "session_size": session_size}


# ---- Run loop ----


async def _run_arena(
    *,
    total: int,
    session_size: int,
    agent_configs: list[dict],
    agent_names: list[str],
    agent_types_raw: list[str],
    seat_config: str,
    has_nn: bool,
    ts_model_path: str | None,
) -> None:
    global _arena_progress

    try:
        import tarok_engine as te  # type: ignore[assignment]
    except ImportError:
        log.error("tarok_engine not available — cannot run arena")
        assert _arena_progress is not None
        _arena_progress["status"] = "error"
        return

    player_stats = [_empty_player_stats(agent_names[i], agent_types_raw[i]) for i in range(4)]
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
    session_scores = [0, 0, 0, 0]
    games_in_session = 0

    def _flush_session() -> None:
        nonlocal session_scores, games_in_session
        indexed = sorted(range(4), key=lambda p: session_scores[p], reverse=True)
        for rank_idx, pid in enumerate(indexed):
            place = rank_idx + 1
            player_stats[pid]["placements"][place] += 1
            player_stats[pid]["placement_sum"] += place
            if place == 1:
                player_stats[pid]["wins"] += 1.0
        for pid in range(4):
            player_stats[pid]["score_history"].append(session_scores[pid])
        session_scores[:] = [0, 0, 0, 0]
        games_in_session = 0

    def _accumulate_scores(
        scores, n_batch, initial_hands=None, initial_talon=None, traces=None
    ) -> None:
        nonlocal games_in_session, session_scores

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
                if initial_hands is not None:
                    ps["best_game_hands"] = initial_hands[batch_max_idx].tolist()
                    ps["best_game_talon"] = initial_talon[batch_max_idx].tolist()
                if traces is not None:
                    ps["best_game_trace"] = serialize_trace(traces[batch_max_idx])
            if ps["worst_game_score"] is None or batch_min < ps["worst_game_score"]:
                ps["worst_game_score"] = batch_min
                ps["worst_game_idx"] = games_done + batch_min_idx
                if initial_hands is not None:
                    ps["worst_game_hands"] = initial_hands[batch_min_idx].tolist()
                    ps["worst_game_talon"] = initial_talon[batch_min_idx].tolist()
                if traces is not None:
                    ps["worst_game_trace"] = serialize_trace(traces[batch_min_idx])

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
                _flush_session()

    def _process_batch(
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
    ) -> None:
        _accumulate_scores(
            scores, n_batch, initial_hands=initial_hands, initial_talon=initial_talon, traces=traces
        )

        if taroks_in_hand is not None:
            for pid in range(4):
                player_stats[pid]["taroks_in_hand_total"] += int(taroks_in_hand[:, pid].sum())
                player_stats[pid]["taroks_in_hand_count"] += n_batch

        if taroks_in_hand is not None and declarers is not None and contracts is not None:
            non_klop = declarers >= 0
            if non_klop.any():
                nk_idx = np.where(non_klop)[0]
                nk_decl = declarers[nk_idx]
                nk_contracts = contracts[nk_idx]
                nk_taroks = taroks_in_hand[nk_idx, nk_decl]
                for cu8 in np.unique(nk_contracts):
                    cname = contract_name(int(cu8))
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
                        cname = contract_name(int(cu8))
                        c_mask = pid_contracts == cu8
                        if cname not in ptpc:
                            ptpc[cname] = {"total_taroks": 0, "count": 0}
                        ptpc[cname]["total_taroks"] += int(pid_taroks[c_mask].sum())
                        ptpc[cname]["count"] += int(c_mask.sum())

        if bid_contracts is not None:
            for pid in range(4):
                bids_col = bid_contracts[:, pid]
                valid_mask = bids_col >= 0
                if valid_mask.any():
                    unique, counts = np.unique(bids_col[valid_mask], return_counts=True)
                    bids = player_stats[pid]["bids_made"]
                    for b, c in zip(unique, counts):
                        bname = contract_name(int(b))
                        bids[bname] = bids.get(bname, 0) + int(c)

        unique_contracts, contract_counts = np.unique(contracts, return_counts=True)
        for cu8, cnt in zip(unique_contracts, contract_counts):
            cname = contract_name(int(cu8))
            if cname not in contract_stats:
                contract_stats[cname] = {
                    "played": 0,
                    "decl_won": 0,
                    "total_decl_score": 0,
                    "total_def_score": 0,
                }
            contract_stats[cname]["played"] += int(cnt)

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
                        cname = contract_name(int(cu8))
                        c_mask = pid_contracts == cu8
                        c_scores = pid_scores[c_mask]
                        if cname not in pcs:
                            pcs[cname] = {"declared": 0, "won": 0, "total_score": 0}
                        pcs[cname]["declared"] += int(c_mask.sum())
                        pcs[cname]["won"] += int((c_scores > 0).sum())
                        pcs[cname]["total_score"] += int(c_scores.sum())

            for cu8 in np.unique(nk_contracts):
                cname = contract_name(int(cu8))
                cs = contract_stats[cname]
                c_mask = nk_contracts == cu8
                c_decl = decl_scores_arr[c_mask]
                cs["decl_won"] += int((c_decl > 0).sum())
                cs["total_decl_score"] += int(c_decl.sum())
                cs["total_def_score"] += int(def_total_arr[c_mask].sum())

        # Notable games
        game_max_scores = np.max(scores, axis=1)
        game_min_scores = np.min(scores, axis=1)
        game_max_pids = np.argmax(scores, axis=1)
        game_min_pids = np.argmin(scores, axis=1)
        valat_mask = (contracts == VALAT_CONTRACT_IDX) | (np.max(np.abs(scores), axis=1) >= 250)

        def _make_notable(gi: int, pid: int, sc: int) -> dict:
            cu = int(contracts[gi])
            return {
                "score": sc,
                "game_idx": games_done + gi,
                "player_idx": pid,
                "player_name": agent_names[pid],
                "contract": contract_name(cu),
                "hands": initial_hands[gi].tolist() if initial_hands is not None else None,
                "talon": initial_talon[gi].tolist() if initial_talon is not None else None,
                "trace": serialize_trace(traces[gi]) if traces is not None else None,
            }

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
                notable_games["best_non_valat"] = _make_notable(int(bi), int(game_max_pids[bi]), bs)
            if (
                notable_games["worst_non_valat"] is None
                or ws < notable_games["worst_non_valat"]["score"]
            ):
                notable_games["worst_non_valat"] = _make_notable(
                    int(wi), int(game_min_pids[wi]), ws
                )

        vm = valat_mask
        if vm.any():
            v_idx = np.where(vm)[0]
            bi = v_idx[int(game_max_scores[v_idx].argmax())]
            wi = v_idx[int(game_min_scores[v_idx].argmin())]
            bs, ws = int(game_max_scores[bi]), int(game_min_scores[wi])
            if notable_games["best_valat"] is None or bs > notable_games["best_valat"]["score"]:
                notable_games["best_valat"] = _make_notable(int(bi), int(game_max_pids[bi]), bs)
            if notable_games["worst_valat"] is None or ws < notable_games["worst_valat"]["score"]:
                notable_games["worst_valat"] = _make_notable(int(wi), int(game_min_pids[wi]), ws)

        for cu8 in np.unique(contracts):
            cname = contract_name(int(cu8))
            c_idx = np.where(contracts == cu8)[0]
            ci = c_idx[int(game_max_scores[c_idx].argmax())]
            sc = int(game_max_scores[ci])
            prev = notable_games["by_contract"].get(cname)
            if prev is None or sc > prev["score"]:
                notable_games["by_contract"][cname] = _make_notable(
                    int(ci), int(game_max_pids[ci]), sc
                )

    def _current_analytics() -> dict:
        return build_analytics(
            player_stats,
            contract_stats,
            taroks_per_contract,
            games_done,
            total,
            session_size,
            notable_games,
        )

    batch_size = min(2_000 if has_nn else 10_000, total)

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

            _process_batch(
                scores=np.asarray(result["scores"]),
                contracts=np.asarray(result["contracts"]),
                declarers=np.asarray(result["declarers"]),
                partners=np.asarray(result["partners"]),
                bid_contracts=np.asarray(result["bid_contracts"])
                if "bid_contracts" in result
                else None,
                taroks_in_hand=np.asarray(result["taroks_in_hand"])
                if "taroks_in_hand" in result
                else None,
                n_batch=n_batch,
                initial_hands=np.asarray(result["initial_hands"])
                if "initial_hands" in result
                else None,
                initial_talon=np.asarray(result["initial_talon"])
                if "initial_talon" in result
                else None,
                traces=result.get("traces"),
            )

            games_done += n_batch
            if games_done == total and games_in_session > 0:
                _flush_session()

            _arena_progress = {
                "status": "running",
                "games_done": games_done,
                "total_games": total,
                "analytics": _current_analytics(),
            }
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        log.info("Arena cancelled at game %d/%d", games_done, total)
        payload = {
            "status": "cancelled",
            "games_done": games_done,
            "analytics": _current_analytics(),
        }
        _arena_progress = {**payload, "total_games": total}
        arena_history.persist_run(agent_configs, total, session_size, payload)
        return
    except Exception:
        log.exception("Arena failed at game %d/%d", games_done, total)
        payload = {"status": "error", "games_done": games_done, "analytics": _current_analytics()}
        _arena_progress = {**payload, "total_games": total}
        arena_history.persist_run(agent_configs, total, session_size, payload)
        return
    finally:
        if ts_model_path:
            try:
                os.unlink(ts_model_path)
            except OSError:
                pass

    payload = {"status": "done", "games_done": games_done, "analytics": _current_analytics()}
    _arena_progress = {**payload, "total_games": total}
    arena_history.persist_run(agent_configs, total, session_size, payload)


# ---- Helpers ----


def _empty_player_stats(name: str, agent_type: str) -> dict:
    return {
        "name": name,
        "type": agent_type,
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
