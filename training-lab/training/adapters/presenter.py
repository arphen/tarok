"""Adapter: terminal presenter — all human-facing output lives here."""

from __future__ import annotations

from pathlib import Path

from training.entities import TrainingConfig, ModelIdentity, TrainingRun
from training.ports import PresenterPort


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _progress_bar(fraction: float, width: int = 40) -> str:
    filled = int(width * fraction)
    return "[" + "=" * filled + ">" * min(1, width - filled) + "." * max(0, width - filled - 1) + "]"


class TerminalPresenter(PresenterPort):
    def on_model_loaded(self, identity: ModelIdentity, save_dir: str) -> None:
        if identity.is_new:
            print(
                f"Creating new model: {identity.name}  "
                f"(arch={identity.model_arch}, hidden={identity.hidden_size}, oracle={identity.oracle_critic})"
            )
            print(f"  Model '{identity.name}' initialized with random weights")
            print(f"  Checkpoints → {save_dir}/")
        else:
            print(
                f"  arch={identity.model_arch}, "
                f"hidden_size={identity.hidden_size}, oracle={identity.oracle_critic}"
            )

    def on_device_selected(self, device: str) -> None:
        print(f"Training device: {device}")
        print()

    def on_training_plan(self, config: TrainingConfig) -> None:
        print(f"{'=' * 70}")
        if config.should_benchmark_initial():
            print(f" INITIAL BENCHMARK  ({config.bench_games} games, {config.effective_bench_seats}, greedy)")
        else:
            print(" INITIAL BENCHMARK  (skipped by config)")
        print(f"{'=' * 70}")

    def on_initial_benchmark(self, placement: float, n_games: int, seats: str, elapsed: float) -> None:
        print(f"  Avg placement: {placement:.3f}  (1.0 = always 1st, 4.0 = always last)")
        print(f"  Benchmark took {_format_time(elapsed)}")
        print()

    def on_training_loop_start(self, config: TrainingConfig) -> None:
        print(f"{'=' * 70}")
        print(f" TRAINING PLAN")
        print(f"   {config.iterations} iterations × {config.games} games/iter")
        print(f"   train seats  = {config.seats}")
        print(f"   bench seats  = {config.effective_bench_seats}")
        lr_info = f"lr={config.lr}"
        if config.lr_schedule != "constant":
            lr_info += f" → {config.effective_lr_min} ({config.lr_schedule})"
        print(f"   PPO: {config.ppo_epochs} epochs, batch_size={config.batch_size}, {lr_info}")
        checkpoints = ",".join(str(i) for i in config.benchmark_checkpoints)
        print(f"   Benchmark: {config.bench_games} games at checkpoints [{checkpoints}] (0=initial)")
        print(f"{'=' * 70}")
        print()

    def on_iteration_start(self, iteration: int, total: int, elapsed: float) -> None:
        frac = (iteration - 1) / total
        if iteration > 1:
            avg = elapsed / (iteration - 1)
            eta = avg * (total - iteration + 1)
            eta_str = f"ETA {_format_time(eta)}"
        else:
            eta_str = "ETA calculating..."
        bar = _progress_bar(frac)
        print(f"─── Iteration {iteration}/{total}  {bar} {frac*100:.0f}%  {eta_str} ───")

    def on_selfplay_start(self, config: TrainingConfig, effective_seats: str | None = None) -> None:
        seats = effective_seats if effective_seats is not None else config.seats
        print(f"  [1/3] Self-play: {config.games} games ({seats}, explore={config.explore_rate})...", end="", flush=True)

    def on_selfplay_done(self, n_experiences: int, elapsed: float) -> None:
        print(f" {n_experiences} exps in {_format_time(elapsed)}")

    def on_ppo_start(
        self,
        config: TrainingConfig,
        iter_lr: float | None = None,
        iter_imitation_coef: float | None = None,
    ) -> None:
        lr_tag = f", lr={iter_lr:.1e}" if iter_lr is not None and config.lr_schedule != "constant" else ""
        il_tag = ""
        if iter_imitation_coef is not None:
            il_tag = f", il_coef={iter_imitation_coef:.4f}"
        print(
            f"  [2/3] PPO update ({config.ppo_epochs} epochs, batch={config.batch_size}{lr_tag}{il_tag})...",
            end="",
            flush=True,
        )

    def on_ppo_done(self, metrics: dict[str, float], elapsed: float) -> None:
        loss = metrics["total_loss"]
        p = metrics["policy_loss"]
        v = metrics["value_loss"]
        e = metrics["entropy"]
        il = metrics.get("il_loss", 0.0)
        il_str = f" il={il:.4f}" if il > 0.0 else ""
        print(f" loss={loss:.4f} (p={p:.4f} v={v:.4f} ent={e:.4f}{il_str}) in {_format_time(elapsed)}")

    def on_benchmark_start(self, config: TrainingConfig) -> None:
        print(f"  [3/3] Benchmark: {config.bench_games} games (greedy, {config.effective_bench_seats})...", end="", flush=True)

    def on_benchmark_done(self, placement: float, elapsed: float) -> None:
        print(f" placement={placement:.3f} in {_format_time(elapsed)}")

    def on_benchmark_skipped(self, iteration: int, config: TrainingConfig) -> None:
        print(f"  [3/3] Benchmark: skipped at iteration {iteration} (checkpoints={list(config.benchmark_checkpoints)})")

    def on_iteration_done(self, prev_placement: float, curr_placement: float, elapsed: float) -> None:
        delta = curr_placement - prev_placement
        direction = "▲ better!" if delta < 0 else "▼ worse" if delta > 0 else "─ same"
        print(f"  → placement {prev_placement:.3f} → {curr_placement:.3f}  ({delta:+.3f} {direction})")
        print(f"  → iteration took {_format_time(elapsed)}")
        print()

    def on_training_complete(self, run: TrainingRun) -> None:
        bar = _progress_bar(1.0)
        print(f"─── Done  {bar} 100%  Total: {_format_time(run.total_time)} ───")
        print()
        _print_summary(run)
        _print_next_steps(run)

    def on_league_elo_updated(self, pool, elo_deltas: dict[str, float] | None = None) -> None:
        entries = pool.entries
        if not entries:
            return
        ranked = sorted(entries, key=lambda e: e.elo, reverse=True)
        print("  [league] Elo standings:")
        for idx, e in enumerate(ranked, start=1):
            delta = 0.0
            if elo_deltas is not None:
                delta = elo_deltas.get(e.opponent.name, 0.0)
            print(
                f"    {idx:>2}. {e.opponent.name:<20} "
                f"{e.elo:>7.1f} ({delta:+.1f})  "
                f"wr={e.win_rate:.2f} gp={e.games_played}"
            )

    def on_league_snapshot_added(self, iteration: int, path: str) -> None:
        print(f"  [league] Snapshot added: iter_{iteration:03d} → {path}")

    def on_memory_stats(
        self,
        iteration: int,
        stats: dict[str, float],
        deltas: dict[str, float] | None = None,
    ) -> None:
        del iteration

        def _fmt(name: str, with_delta: bool = True) -> str:
            v = stats.get(name)
            if v is None:
                return ""
            if not with_delta or deltas is None or name not in deltas:
                return f"{name}={v:.0f}MB"
            d = deltas[name]
            return f"{name}={v:.0f}MB ({d:+.0f})"

        parts = [
            _fmt("rss_mb"),
            _fmt("footprint_mb"),
            _fmt("compressed_mb"),
            _fmt("py_heap_mb"),
            _fmt("py_heap_peak_mb", with_delta=False),
        ]
        if "mps_alloc_mb" in stats:
            parts.append(_fmt("mps_alloc_mb"))
        if "mps_driver_mb" in stats:
            parts.append(_fmt("mps_driver_mb"))
        if "cuda_alloc_mb" in stats:
            parts.append(_fmt("cuda_alloc_mb"))
        if "cuda_reserved_mb" in stats:
            parts.append(_fmt("cuda_reserved_mb"))

        parts = [p for p in parts if p]
        if parts:
            print("  [mem] " + " | ".join(parts))


def _print_summary(run: TrainingRun) -> None:
    print(f"{'=' * 70}")
    print(f" RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print()

    placements = run.placements
    results = run.results

    print(f"  {'Iter':>6s}  {'Placement':>10s}  {'Change':>8s}  {'Loss':>10s}")
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 8}  {'─' * 10}")
    print(f"  {'init':>6s}  {placements[0]:>10.3f}  {'':>8s}  {'':>10s}")
    for i, r in enumerate(results, 1):
        delta = placements[i] - placements[i - 1]
        arrow = "▲" if delta < 0 else "▼" if delta > 0 else "─"
        print(f"  {i:>6d}  {placements[i]:>10.3f}  {delta:>+7.3f}{arrow}  {r.loss:>10.4f}")

    print()
    overall_delta = placements[-1] - placements[0]
    direction = "IMPROVED" if overall_delta < 0 else "REGRESSED" if overall_delta > 0 else "UNCHANGED"
    print(f"  Overall: {placements[0]:.3f} → {placements[-1]:.3f}  ({overall_delta:+.3f})  {direction}")
    print(f"  Best placement: {run.best_placement:.3f} at iteration {run.best_iteration}")
    print(f"  Best loss:      {run.best_loss:.4f} at iteration {run.best_loss_iteration}")
    print()

    print(f"  Best model saved to {run.config.save_dir}/best.pt (metric={run.config.best_model_metric})")

    print()
    avg_iter = sum(r.total_time for r in results) / len(results) if results else 0
    print(f"  Total training time: {_format_time(run.total_time)}")
    print(f"  Avg iteration time:  {_format_time(avg_iter)}")
    print(f"  Checkpoints saved in {run.config.save_dir}/")


def _print_next_steps(run: TrainingRun) -> None:
    cfg = run.config
    save_dir = Path(cfg.save_dir)
    best_pt = save_dir / "best.pt"
    last_pt = save_dir / f"iter_{cfg.iterations:03d}.pt"
    model_pt = str(best_pt if best_pt.exists() else last_pt)

    config_name = ""
    # Try to infer config name from save_dir
    if cfg.seats == "nn,bot_v5,bot_v5,bot_v5":
        config_name = "vs-3-bots"
    elif cfg.seats == "nn,bot_v3,bot_v3,bot_v3":
        config_name = "vs-3-v3"
    elif cfg.seats == "nn,bot_v6,bot_v6,bot_v6":
        config_name = "vs-3-v6"
    elif cfg.seats == "nn,nn,nn,nn":
        config_name = "self-play"

    print()
    print(f"{'=' * 70}")
    print(f" WHAT NEXT?")
    print(f"{'=' * 70}")
    print()

    more_iters = cfg.iterations * 2
    cmd1 = f"make train-iterate MODEL={model_pt}"
    if config_name and config_name != "vs-3-bots":
        cmd1 += f" CONFIG={config_name}"
    cmd1 += f' EXTRA="--iterations {more_iters}"'
    print(f"  [1] Keep training ({more_iters} more iterations, same config):")
    print(f"      {cmd1}")
    print()

    next_configs = {
        "vs-3-bots": ("vs-3-v6", "harder v6 bots"),
        "": ("vs-3-v6", "harder v6 bots"),
        "vs-3-v6": ("self-play", "pure self-play (no bots)"),
        "vs-2-bots": ("self-play", "pure self-play (no bots)"),
        "vs-1-bot": ("self-play", "pure self-play (no bots)"),
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
    print(f"  [3] Fine-tune with lower learning rate ({fine_lr:.1e}):")
    print(f"      {cmd3}")
    print()

    print(f"  [4] Play in the browser:")
    print("      make run")
    print(f"      Then select this checkpoint in the UI: {model_pt}")
    print()

    print(f"  [5] Train a brand-new model from scratch:")
    print(f"      make train-new")
    print()
