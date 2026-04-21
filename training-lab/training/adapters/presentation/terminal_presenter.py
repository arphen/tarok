"""Adapter: terminal presenter — all human-facing output lives here."""

from __future__ import annotations

from pathlib import Path

from training.entities import TrainingConfig, ModelIdentity, TrainingRun
from training.entities.iteration_hyperparams import IterationHyperparams
from training.ports import PresenterPort

# ── Visual constants ──────────────────────────────────────────────────────
_W = 72  # total line width
_THIN = "─"
_BOLD = "━"
_BOX_H = "═"
_ANSI_RESET = "\033[0m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_CYAN = "\033[36m"


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _bar(fraction: float, width: int = 30) -> str:
    filled = int(width * fraction)
    head = "▸" if filled < width else ""
    return "▕" + "█" * filled + head + "░" * max(0, width - filled - len(head)) + "▏"


def _banner(text: str) -> str:
    pad = _W - 4 - len(text)
    return f"╔{'═' * (_W - 2)}╗\n║  {text}{' ' * max(0, pad)}║\n╚{'═' * (_W - 2)}╝"


def _section(text: str) -> str:
    return f"┌{'─' * (_W - 2)}┐\n│  {text}{' ' * max(0, _W - 4 - len(text))}│\n└{'─' * (_W - 2)}┘"


def _fmt_scalar(value: float) -> str:
    if value == 0:
        return "0"
    mag = abs(value)
    if mag < 1e-3 or mag >= 1e3:
        return f"{value:.1e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _fmt_schedule(max_value: float, min_value: float, schedule: str) -> str:
    if schedule == "constant":
        return f"{_fmt_scalar(max_value)} ({schedule})"
    return f"{_fmt_scalar(max_value)} -> {_fmt_scalar(min_value)} ({schedule})"


def _fmt_imitation_schedule(config: TrainingConfig) -> str:
    if config.imitation_schedule == "gaussian_elo":
        return (
            f"peak={_fmt_scalar(config.imitation_coef)} @ elo={config.imitation_center_elo:.0f} "
            f"(width={config.imitation_width_elo:.0f})"
        )
    return _fmt_schedule(config.imitation_coef, config.imitation_coef_min, config.imitation_schedule)


def _delta_arrow(delta: float, lower_is_better: bool = True) -> str:
    if delta == 0:
        return "  ─"
    if lower_is_better:
        return " ▲+" if delta < 0 else " ▼−"
    return " ▲+" if delta > 0 else " ▼−"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_ANSI_RESET}"


class TerminalPresenter(PresenterPort):

    # ── Setup ─────────────────────────────────────────────────────────────

    def on_model_loaded(self, identity: ModelIdentity, save_dir: str) -> None:
        if identity.is_new:
            print(_banner(f"NEW MODEL: {identity.name}"))
            print(f"  arch={identity.model_arch}  hidden={identity.hidden_size}  oracle={identity.oracle_critic}")
        else:
            print(f"  arch={identity.model_arch}  hidden={identity.hidden_size}  oracle={identity.oracle_critic}")
        print(f"  checkpoints → {save_dir}/")

    def on_device_selected(self, device: str) -> None:
        print(f"  device = {device}")
        print()

    # ── Benchmarks & plan ─────────────────────────────────────────────────

    def on_training_plan(self, config: TrainingConfig) -> None:
        print(_section(f"INITIAL BENCHMARK  ({config.bench_games} games, {config.effective_bench_seats}, greedy)"))

    def on_initial_benchmark(self, placement: float, n_games: int, seats: str, elapsed: float) -> None:
        print(f"  placement = {placement:.3f}   (1.0 = always 1st, 4.0 = always last)")
        print(f"  took {_format_time(elapsed)}")
        print()

    def on_training_loop_start(self, config: TrainingConfig) -> None:
        ckpts = ",".join(str(i) for i in config.benchmark_checkpoints)
        print(_banner("TRAINING PLAN"))
        print(f"  Run       : {config.iterations} iterations x {config.games:,} games")
        print(f"  Seats     : train={config.seats}")
        print(f"              bench={config.effective_bench_seats}")
        print(
            f"  Bench     : {config.bench_games:,} games @ [{ckpts}]  "
            f"metric={config.best_model_metric}  session={config.outplace_session_size}g"
        )
        print(
            f"  PPO       : epochs={config.ppo_epochs}  batch={config.batch_size:,}  "
            f"gamma={config.gamma:.3f}  gae={config.gae_lambda:.3f}  "
            f"clip={config.clip_epsilon:.3f}  policy={config.policy_coef:.3f}  value={config.value_coef:.3f}"
        )
        print("  Schedules :")
        print(f"              lr       {_fmt_schedule(config.lr, config.effective_lr_min, config.lr_schedule)}")
        print(
            f"              entropy  "
            f"{_fmt_schedule(config.entropy_coef, config.entropy_coef_min, config.entropy_schedule)}"
        )
        explore_rate_min = getattr(config, "explore_rate_min", config.explore_rate)
        explore_rate_schedule = getattr(config, "explore_rate_schedule", "constant")
        print(
            f"              explore  "
            f"{_fmt_schedule(config.explore_rate, explore_rate_min, explore_rate_schedule)}"
        )
        print(f"              oracle-distill {_fmt_imitation_schedule(config)}")
        if config.behavioral_clone_coef > 0.0 and config.behavioral_clone_games_per_iteration > 0:
            print(
                f"              behavior-clone "
                f"{_fmt_schedule(config.behavioral_clone_coef, config.behavioral_clone_coef_min, config.behavioral_clone_schedule)} "
                f"teacher={config.behavioral_clone_teacher} "
                f"games/iter={config.behavioral_clone_games_per_iteration:,}"
            )
        print(
            f"  Runtime   : device={config.device}  concurrency={config.concurrency}  "
            f"runner={config.iteration_runner_mode}  restart_every={config.iteration_runner_restart_every}"
        )
        if config.league is not None and config.league.enabled:
            print(
                f"  League    : enabled  opponents={len(config.league.opponents)}  "
                f"sampling={config.league.sampling}  min_nn={config.league.min_nn_per_game}  "
                f"snapshot_every={config.league.snapshot_interval}"
            )
            print(f"              elo_outplace_unit_weight={config.league.elo_outplace_unit_weight}")
        print()

    # ── Iteration loop ────────────────────────────────────────────────────

    def on_iteration_start(self, iteration: int, total: int, elapsed: float) -> None:
        frac = (iteration - 1) / total
        if iteration > 1:
            avg = elapsed / (iteration - 1)
            eta = avg * (total - iteration + 1)
            eta_str = f"ETA {_format_time(eta)}"
        else:
            eta_str = "ETA …"
        bar = _bar(frac)
        print(f"{_BOLD * _W}")
        print(f"  Iteration {iteration}/{total}  {bar}  {frac*100:3.0f}%   {eta_str}")
        print(f"{_THIN * _W}")

    def on_selfplay_start(self, config: TrainingConfig, effective_seats: str | None = None, hyperparams: IterationHyperparams | None = None) -> None:
        seats = effective_seats if effective_seats is not None else config.seats
        eps = hyperparams.explore_rate if hyperparams is not None else config.explore_rate
        print(f"  ① Self-play   {config.games:,} games  (ε={eps:.3f})")
        print(f"      seats: {seats}")
        print("      stats: ", end="", flush=True)

    def on_selfplay_done(self, n_total: int, n_learner: int, elapsed: float) -> None:
        if n_learner < n_total:
            print(f"{n_total:,} exp total, {n_learner:,} learner  [{_format_time(elapsed)}]")
        else:
            print(f"{n_total:,} exp  [{_format_time(elapsed)}]")

    def on_ppo_start(
        self,
        config: TrainingConfig,
        hyperparams: IterationHyperparams | None = None,
        iter_lr: float | None = None,
        iter_imitation_coef: float | None = None,
        iter_behavioral_clone_coef: float | None = None,
        iter_entropy_coef: float | None = None,
    ) -> None:
        # Build hyperparams from legacy kwargs if not passed directly.
        if hyperparams is None and any(
            v is not None for v in (iter_lr, iter_imitation_coef, iter_behavioral_clone_coef, iter_entropy_coef)
        ):
            hyperparams = IterationHyperparams(
                lr=iter_lr if iter_lr is not None else config.lr,
                imitation_coef=iter_imitation_coef if iter_imitation_coef is not None else config.imitation_coef,
                behavioral_clone_coef=iter_behavioral_clone_coef if iter_behavioral_clone_coef is not None else 0.0,
                entropy_coef=iter_entropy_coef if iter_entropy_coef is not None else config.entropy_coef,
                explore_rate=config.explore_rate,
            )
        hp = hyperparams
        lr_tag = f"  lr={hp.lr:.1e}" if hp is not None and config.lr_schedule != "constant" else ""
        il_tag = f"  il={hp.imitation_coef:.4f}" if hp is not None else ""
        bc_tag = f"  bc={hp.behavioral_clone_coef:.4f}" if hp is not None and hp.behavioral_clone_coef > 0 else ""
        ent_tag = f"  ent={hp.entropy_coef:.5f}" if hp is not None and config.entropy_schedule != "constant" else ""
        exp_tag = f"  ε={hp.explore_rate:.3f}" if hp is not None and getattr(config, "explore_rate_schedule", "constant") != "constant" else ""
        print(f"  ② PPO update   {config.ppo_epochs}ep  batch={config.batch_size}{lr_tag}{il_tag}{bc_tag}{ent_tag}{exp_tag}")
        print("      stats: ", end="", flush=True)

    def on_ppo_done(self, metrics: dict[str, float], elapsed: float) -> None:
        p = metrics["policy_loss"]
        v = metrics["value_loss"]
        e = metrics["entropy"]
        il = metrics.get("il_loss", 0.0)
        bc = metrics.get("bc_loss", 0.0)
        il_str = f"  il={il:.4f}" if il > 0.0 else ""
        bc_str = f"  bc={bc:.4f}" if bc > 0.0 else ""
        print(
            f"loss={metrics['total_loss']:.4f}  "
            f"(π={p:.4f}  v={v:.4f}  H={e:.4f}{il_str}{bc_str})  [{_format_time(elapsed)}]"
        )

    def on_benchmark_start(self, config: TrainingConfig) -> None:
        print(f"  ③ Benchmark    {config.bench_games:,} games  (greedy)")
        print(f"      seats: {config.effective_bench_seats}")
        print("      stats: ", end="", flush=True)

    def on_benchmark_done(self, placement: float, elapsed: float) -> None:
        print(f"placement={placement:.3f}  [{_format_time(elapsed)}]")

    def on_benchmark_skipped(self, iteration: int, config: TrainingConfig) -> None:
        print(f"  ③ Benchmark    skipped  (next: iter {_next_checkpoint(iteration, config.benchmark_checkpoints)})")

    def on_iteration_done(self, prev_placement: float, curr_placement: float, elapsed: float) -> None:
        delta = curr_placement - prev_placement
        if delta < -0.001:
            tag = f"▲ {abs(delta):.3f} better"
        elif delta > 0.001:
            tag = f"▼ {abs(delta):.3f} worse"
        else:
            tag = "─ same"
        print(f"{_THIN * _W}")
        print(f"  placement  {prev_placement:.3f} → {curr_placement:.3f}  ({tag})    took {_format_time(elapsed)}")

    # ── League ────────────────────────────────────────────────────────────

    def on_league_elo_updated(self, pool, elo_deltas: dict[str, float] | None = None) -> None:
        entries = pool.entries
        if not entries:
            return

        learner_delta = elo_deltas.get("__learner__", 0.0) if elo_deltas else 0.0

        rows: list[tuple[str, float, float | None, str]] = []
        rows.append(("★ LEARNER", pool.learner_elo, learner_delta, ""))
        for e in entries:
            recent = e.recent_outplace_rate
            extra = f"outplace {e.outplace_rate:.0%}  ({e.games_played:,} sessions)"
            rows.append((e.opponent.name, e.elo, recent, extra))
        rows.sort(key=lambda x: x[1], reverse=True)

        print(f"  {'┈' * (_W - 4)}")
        print(f"  {'#':>3}  {'Name':<22} {'Elo':>7}  {'Δ':>8}  long term")
        for idx, (name, elo, delta_or_recent, extra) in enumerate(rows, 1):
            if name == "★ LEARNER":
                d_str = f"{(delta_or_recent or 0.0):+.1f}"
                if (delta_or_recent or 0.0) > 0:
                    d_str = _color(d_str, _ANSI_GREEN)
                elif (delta_or_recent or 0.0) < 0:
                    d_str = _color(d_str, _ANSI_RED)
                name_str = _color(name, _ANSI_CYAN)
            else:
                if delta_or_recent is None:
                    d_str = ""
                else:
                    pct = delta_or_recent * 100.0
                    d_str = f"{pct:>6.1f}%"
                    if pct >= 50.0:
                        d_str = _color(d_str, _ANSI_GREEN)
                    else:
                        d_str = _color(d_str, _ANSI_RED)
                name_str = name
            print(f"  {idx:>3}  {name_str:<22} {elo:>7.1f}  {d_str:>8}  {extra}")

    def on_league_snapshot_added(self, iteration: int, path: str) -> None:
        print(f"  📸 snapshot saved → {path}")

    def confirm_league_state_reset(
        self,
        previous_profile: str,
        current_profile: str,
        league_pool_dir: str,
    ) -> bool:
        print("  ⚠ league profile mismatch detected")
        print(f"      previous: {previous_profile}")
        print(f"      current : {current_profile}")
        print(f"      state   : {league_pool_dir}")
        answer = input("  Reset league state for this config switch? [y/N]: ").strip().lower()
        return answer in {"y", "yes"}

    def on_initial_league_calibration_start(
        self,
        n_opponents: int,
        n_games_per_pair: int,
        anchor_name: str | None,
        anchor_elo: float,
    ) -> None:
        anchor_label = anchor_name if anchor_name is not None else "first opponent"
        print("  ⓪ League calibration (greedy, one-time)")
        print(
            f"      opponents={n_opponents}  games/pair={n_games_per_pair:,}  "
            f"anchor={anchor_label}@{anchor_elo:.0f}"
        )

    def on_initial_league_calibration_done(self, elapsed: float) -> None:
        print(f"      done [{_format_time(elapsed)}]")

    def on_initial_league_calibration_pair_result(
        self,
        pair_idx: int,
        total_pairs: int,
        left_name: str,
        right_name: str,
        left_wins: int,
        right_wins: int,
        draws: int,
        left_score: float,
    ) -> None:
        print(
            f"      [{pair_idx}/{total_pairs}] {left_name} vs {right_name}  "
            f"{left_wins}-{right_wins}-{draws}  score={left_score:.3f}"
        )

    def on_initial_league_calibration_mixed_result(
        self,
        run_idx: int,
        total_runs: int,
        target_name: str,
        seat_tokens: tuple[str, str, str],
        placements: tuple[float, float, float, float],
    ) -> None:
        target_place, opp1, opp2, opp3 = placements
        print(
            f"      [{run_idx}/{total_runs}] {target_name} vs ({seat_tokens[0]}, {seat_tokens[1]}, {seat_tokens[2]})  "
            f"{target_name}={target_place:.3f} vs ({opp1:.3f}, {opp2:.3f}, {opp3:.3f})"
        )

    def on_league_full_recalibration_start(
        self,
        snapshot_name: str,
        n_entries: int,
        n_games_per_entry: int,
    ) -> None:
        print(f"  ④ Full Elo recalibration  (triggered by {snapshot_name})")
        print(f"      entries={n_entries}  games/entry={n_games_per_entry:,}  mode=greedy")

    def on_league_full_recalibration_done(
        self,
        snapshot_name: str,
        elapsed: float,
        calibrated: bool,
    ) -> None:
        status = "done" if calibrated else "skipped"
        print(f"      {status} [{_format_time(elapsed)}]")

    def on_snapshot_calibration_start(
        self,
        snapshot_name: str,
        n_opponents: int,
        n_games_per_opponent: int,
    ) -> None:
        print(f"  ④ Snapshot calibration  {snapshot_name}")
        print(
            f"      opponents={n_opponents}  games/opponent={n_games_per_opponent:,}  mode=greedy"
        )

    def on_snapshot_calibration_done(
        self,
        snapshot_name: str,
        elapsed: float,
        calibrated: bool,
    ) -> None:
        status = "calibrated" if calibrated else "skipped"
        print(f"      {status} [{_format_time(elapsed)}]")

    def on_snapshot_calibration_match_setup(
        self,
        snapshot_name: str,
        seat_config: str,
        n_games: int,
    ) -> None:
        print(f"      match: {seat_config}")
        print(f"      games: {n_games:,}")

    def on_snapshot_calibration_pair_result(
        self,
        snapshot_name: str,
        opponent_name: str,
        snapshot_wins: int,
        opponent_wins: int,
        draws: int,
        snapshot_score: float,
        implied_snapshot_elo: float,
        opponent_elo: float,
    ) -> None:
        print(
            f"      {snapshot_name} vs {opponent_name}  "
            f"avg_place={snapshot_score:.3f}  implied={implied_snapshot_elo:.1f} from {opponent_elo:.1f}"
        )

    # ── Completion ────────────────────────────────────────────────────────

    def on_training_complete(self, run: TrainingRun) -> None:
        bar = _bar(1.0)
        print(f"{_BOLD * _W}")
        print(f"  DONE  {bar}  100%   Total: {_format_time(run.total_time)}")
        print(f"{_BOLD * _W}")
        print()
        _print_summary(run)
        _print_next_steps(run)


# ── Helpers ───────────────────────────────────────────────────────────────

def _next_checkpoint(current: int, checkpoints: list[int] | tuple[int, ...]) -> int | str:
    for c in sorted(checkpoints):
        if c > current:
            return c
    return "—"


def _print_summary(run: TrainingRun) -> None:
    print(_banner("RESULTS"))

    placements = run.placements
    results = run.results

    header = f"  {'Iter':>6}  {'Placement':>10}  {'Δ':>8}  {'Loss':>10}"
    print(header)
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 8}  {'─' * 10}")
    print(f"  {'init':>6}  {placements[0]:>10.3f}")
    for i, r in enumerate(results, 1):
        delta = placements[i] - placements[i - 1]
        arrow = "▲" if delta < 0 else "▼" if delta > 0 else " "
        print(f"  {i:>6}  {placements[i]:>10.3f}  {delta:>+7.3f}{arrow}  {r.loss:>10.4f}")

    print()
    overall_delta = placements[-1] - placements[0]
    direction = "IMPROVED" if overall_delta < 0 else "REGRESSED" if overall_delta > 0 else "UNCHANGED"
    print(f"  Overall: {placements[0]:.3f} → {placements[-1]:.3f}  ({overall_delta:+.3f})  {direction}")
    print(f"  Best placement: {run.best_placement:.3f}  (iter {run.best_iteration})")
    print(f"  Best loss:      {run.best_loss:.4f}  (iter {run.best_loss_iteration})")
    print(f"  Best saved to:  {run.config.save_dir}/best.pt  (metric={run.config.best_model_metric})")
    print()

    avg_iter = sum(r.total_time for r in results) / len(results) if results else 0
    print(f"  Total: {_format_time(run.total_time)}    Avg iter: {_format_time(avg_iter)}")
    print()


def _print_next_steps(run: TrainingRun) -> None:
    cfg = run.config
    save_dir = Path(cfg.save_dir)
    best_pt = save_dir / "best.pt"
    last_pt = save_dir / f"iter_{cfg.iterations:03d}.pt"
    model_pt = str(best_pt if best_pt.exists() else last_pt)

    config_name = ""
    if cfg.seats == "nn,bot_v5,bot_v5,bot_v5":
        config_name = "vs-3-bots"
    elif cfg.seats == "nn,bot_v3,bot_v3,bot_v3":
        config_name = "vs-3-v3"
    elif cfg.seats == "nn,bot_v6,bot_v6,bot_v6":
        config_name = "vs-3-v6"
    elif cfg.seats == "nn,nn,nn,nn":
        config_name = "self-play"

    print(_section("WHAT NEXT?"))
    print()

    more_iters = cfg.iterations * 2
    cmd1 = f"make train-iterate MODEL={model_pt}"
    if config_name and config_name != "vs-3-bots":
        cmd1 += f" CONFIG={config_name}"
    cmd1 += f' EXTRA="--iterations {more_iters}"'
    print(f"  [1] Keep training ({more_iters} more iters, same config):")
    print(f"      {cmd1}")
    print()

    next_configs = {
        "vs-3-bots": ("vs-3-v6", "harder v6 bots"),
        "": ("vs-3-v6", "harder v6 bots"),
        "vs-3-v6": ("self-play", "pure self-play"),
        "vs-2-bots": ("self-play", "pure self-play"),
        "vs-1-bot": ("self-play", "pure self-play"),
    }
    next_config, next_desc = next_configs.get(config_name, ("vs-3-v6", "harder v6 bots"))
    print(f"  [2] Escalate to {next_desc}:")
    print(f"      make train-iterate MODEL={model_pt} CONFIG={next_config}")
    print()

    fine_lr = cfg.lr / 3
    cmd3 = f"make train-iterate MODEL={model_pt}"
    if config_name and config_name != "vs-3-bots":
        cmd3 += f" CONFIG={config_name}"
    cmd3 += f' EXTRA="--iterations {cfg.iterations} --lr {fine_lr:.1e}"'
    print(f"  [3] Fine-tune with lower lr ({fine_lr:.1e}):")
    print(f"      {cmd3}")
    print()

    print(f"  [4] Play in browser:")
    print(f"      make run → select {model_pt}")
    print()

    print(f"  [5] Train from scratch:")
    print(f"      make train-new")
    print()
