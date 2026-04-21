"""Arena use-case: analytics, leaderboard, seat helpers, checkpoint export."""

from __future__ import annotations

# Contract name lookup (must match engine-rs Contract enum order)
CONTRACT_NAMES = [
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
VALAT_CONTRACT_IDX = 9  # BARVNI_VALAT


def contract_name(idx: int) -> str:
    return CONTRACT_NAMES[idx] if idx < len(CONTRACT_NAMES) else "UNKNOWN"


def serialize_trace(t: dict) -> dict:
    """Convert a raw numpy trace dict to JSON-serialisable form."""
    return {
        "bids": [list(b) for b in t["bids"]],
        "king_call": list(t["king_call"]) if t["king_call"] is not None else None,
        "talon_pick": list(t["talon_pick"]) if t["talon_pick"] is not None else None,
        "put_down": list(t["put_down"]),
        "cards_played": [list(c) for c in t["cards_played"]],
        "dealer": int(t["dealer"]),
    }


def agent_type_to_seat_label(agent_type: str) -> str | None:
    """Map frontend agent type string to Rust seat_config label."""
    t = agent_type.strip().lower()
    if t in ("stockskis", "stockskis_v5"):
        return "bot_v5"
    if t in ("stockskis_lapajne", "lapajne", "bot_lapajne"):
        return "bot_lapajne"
    if t == "stockskis_v6":
        return "bot_v6"
    if t in ("stockskis_m6", "bot_m6"):
        return "bot_m6"
    if t in ("stockskis_m8", "bot_m8"):
        return "bot_m8"
    if t in ("stockskis_pozrl", "bot_pozrl", "pozrl"):
        return "bot_pozrl"
    if t in ("lustrek", "stockskis_lustrek", "bot_lustrek"):
        return "bot_lustrek"
    if t == "rl":
        return "nn"
    return None


def export_checkpoint_to_torchscript(checkpoint_path: str) -> str:
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

    from tarok_model.encoding import STATE_SIZE

    wrapper = _AllHeads(net)
    wrapper.eval()
    traced = torch.jit.trace(wrapper, torch.randn(1, STATE_SIZE), check_trace=False)

    with tempfile.NamedTemporaryFile(suffix=".pt", prefix="tarok_arena_ts_", delete=False) as f:
        path = f.name
    traced.save(path)
    return path


def build_analytics(
    player_stats: list[dict],
    contract_stats: dict,
    taroks_per_contract: dict,
    games_done: int,
    total_games: int,
    session_size: int = 50,
    notable_games: dict | None = None,
) -> dict:
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

    tpc = {
        name: round(tc["total_taroks"] / max(tc["count"], 1), 2)
        for name, tc in taroks_per_contract.items()
    }

    return {
        "games_done": games_done,
        "total_games": total_games,
        "players": players,
        "contracts": contracts,
        "taroks_per_contract": tpc,
        "notable_games": notable_games,
    }


def build_checkpoint_leaderboard(runs: list[dict]) -> list[dict]:
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
