"""Adapter: terminal presenter — all human-facing output lives here."""

from __future__ import annotations

from pathlib import Path

from training.entities import TrainingConfig, ModelIdentity, TrainingRun
from training.ports import PresenterPort

# ── Visual constants ──────────────────────────────────────────────────────
_W = 72  # total line width
_THIN = "─"
_BOLD = "━"
_BOX_H = "═"


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


def _delta_arrow(delta: float, lower_is_better: bool = True) -> str:
    if delta == 0:
        return "  ─"
    if lower_is_better:
        return " ▲+" if delta < 0 else " ▼−"
    return " ▲+" if delta > 0 else " ▼−"


class TerminalPresenter(PresenterPort):

    # ── Setup ─────────────────────────────────────────────────────────────

    def on_model_loaded(self, identity: ModelIdentity, save_dir: str) -> None:
        if identity.is_new:
            print(_banner(f"NEW MODEL: {identity.name}"))
            print(f"  arch={identity.model_arch}  hidden={identity.hidden_size}  oracle={identity.oracle_critic}")
            print(f"  checkpoints → {save_dir}/")
        else:
            print(f"  arch={identity.model_arch}  hidden={identity.hidden_size}  oracle={identity.oracle_critic}")

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
        lr_info = f"lr={config.lr}"
        if config.lr_schedule != "constant":
            lr_info += f"→{config.effective_lr_min} ({config.lr_schedule})"
        ckpts = ",".join(str(i) for i in config.benchmark_checkpoints)
        print(_banner("TRAINING PLAN"))
        print(f"  {config.iterations} iters × {config.games} games    "
              f"train={config.seats}    bench={config.effective_bench_seats}")
        print(f"  PPO {config.ppo_epochs}ep  batch={config.batch_size}  {lr_info}    "
              f"bench={config.bench_games}g @ [{ckpts}]")
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

    def on_selfplay_start(self, config: TrainingConfig, effective_seats: str | None = None) -> None:
        seats = effective_seats if effective_seats is not None else config.seats
        print(f"  ① Self-play   {config.games:,} games  (ε={config.explore_rate})")
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
        iter_lr: float | None = None,
        iter_imitation_coef: float | None = None,
        iter_entropy_coef: float | None = None,
    ) -> None:
        lr_tag = f"  lr={iter_lr:.1e}" if iter_lr is not None and config.lr_schedule != "constant" else ""
        il_tag = f"  il={iter_imitation_coef:.4f}" if iter_imitation_coef is not None else ""
        ent_tag = f"  ent={iter_entropy_coef:.5f}" if iter_entropy_coef is not None and config.entropy_schedule != "constant" else ""
        print(f"  ② PPO update   {config.ppo_epochs}ep  batch={config.batch_size}{lr_tag}{il_tag}{ent_tag}")
        print("      stats: ", end="", flush=True)

    def on_ppo_done(self, metrics: dict[str, float], elapsed: float) -> None:
        p = metrics["policy_loss"]
        v = metrics["value_loss"]
        e = metrics["entropy"]
        il = metrics.get("il_loss", 0.0)
        il_str = f"  il={il:.4f}" if il > 0.0 else ""
        print(f"loss={metrics['total_loss']:.4f}  (π={p:.4f}  v={v:.4f}  H={e:.4f}{il_str})  [{_format_time(elapsed)}]")

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

        rows: list[tuple[str, float, float, str]] = []
        rows.append(("★ LEARNER", pool.learner_elo, learner_delta, ""))
        for e in entries:
            delta = elo_deltas.get(e.opponent.name, 0.0) if elo_deltas else 0.0
            extra = f"outplace {e.outplace_rate:.0%}  ({e.games_played:,} games)"
            rows.append((e.opponent.name, e.elo, delta, extra))
        rows.sort(key=lambda x: x[1], reverse=True)

        print(f"  {'┈' * (_W - 4)}")
        print(f"  {'#':>3}  {'Name':<22} {'Elo':>7}  {'Δ':>6}  Details")
        for idx, (name, elo, delta, extra) in enumerate(rows, 1):
            d_str = f"{delta:+.1f}" if delta else ""
            print(f"  {idx:>3}  {name:<22} {elo:>7.1f}  {d_str:>6}  {extra}")

    def on_league_snapshot_added(self, iteration: int, path: str) -> None:
        print(f"  📸 snapshot saved → {path}")

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
