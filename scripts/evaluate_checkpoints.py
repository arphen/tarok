#!/usr/bin/env python3
"""Evaluate every checkpoint under data/checkpoints/ in a big tournament.

For every training run directory under data/checkpoints/ we pick a
representative checkpoint (`best.pt` > newest `iter_*.pt` > `_current.pt`).
We also include every `*.pt` file directly under `hall_of_fame/`, and a
couple of heuristic baselines (`bot_v5`, `bot_m6`).

Each candidate is placed in seat 0 against a fixed opponent lineup
(default: `bot_m6,bot_v5,bot_m6`) and we run N games via the single
canonical `tarok_engine.run_self_play` path (see copilot-instructions.md).

Candidates whose checkpoint cannot be exported to TorchScript, or whose
seat crashes during play, are flagged as "broken" and recommended for
deletion — the script prints ready-to-paste shell commands for cleanup
and for promoting the top non-HoF models into `hall_of_fame/`.

Usage:
    backend/.venv/bin/python scripts/evaluate_checkpoints.py \
        --games 500 --session-size 50

Run with `--help` for the full flag list.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Make the backend package importable (for export_checkpoint_to_torchscript)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT / "model" / "src"))

# Importing torch before tarok_engine on macOS avoids @rpath/libtorch_cpu.dylib
# load failures in the PyO3 extension.
import torch  # noqa: F401  (side-effect: preload libtorch dylibs)

import numpy as np

import tarok_engine as te
from tarok.use_cases.arena import export_checkpoint_to_torchscript


CHECKPOINT_ROOT = ROOT / "data" / "checkpoints"
HOF_DIR = CHECKPOINT_ROOT / "hall_of_fame"

# Fixed baselines that don't need a checkpoint.
BOT_BASELINES = ("bot_m6", "bot_v5")


@dataclass
class Candidate:
    """Something we can seat at position 0 in run_self_play."""

    name: str               # display label
    kind: str               # "model" | "hof" | "bot"
    seat_token: str         # either a .pt path (torchscript-exported) or a bot label
    source_path: Path | None = None   # the original checkpoint path (for rm suggestions)
    source_dir: Path | None = None    # the run directory (for rm -rf suggestions)


@dataclass
class Result:
    candidate: Candidate
    games_played: int = 0
    total_score: int = 0
    wins: int = 0
    positive_games: int = 0
    error: str | None = None
    duration_s: float = 0.0
    ts_path: str | None = None  # torchscript temp file to clean up

    @property
    def avg_session_score(self) -> float:
        # Project convention: "avg score" = total / number_of_sessions.
        if self.games_played == 0:
            return 0.0
        n_sessions = max(self.games_played / max(self.session_size, 1), 1.0)
        return self.total_score / n_sessions

    session_size: int = 50  # filled in by runner


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def _pick_representative(run_dir: Path) -> Path | None:
    """Return the checkpoint we should use for this training run, or None."""
    best = run_dir / "best.pt"
    if best.is_file():
        return best
    iters = sorted(run_dir.glob("iter_*.pt"))
    if iters:
        return iters[-1]
    current = run_dir / "_current.pt"
    if current.is_file():
        return current
    return None


def discover_candidates(include_bots: bool = True) -> list[Candidate]:
    cands: list[Candidate] = []

    # Training-run checkpoints
    for run_dir in sorted(p for p in CHECKPOINT_ROOT.iterdir() if p.is_dir()):
        if run_dir.name == "hall_of_fame":
            continue
        ckpt = _pick_representative(run_dir)
        if ckpt is None:
            # No loadable checkpoint at all — recommend deletion up front.
            cands.append(
                Candidate(
                    name=f"{run_dir.name} (empty)",
                    kind="model",
                    seat_token="__MISSING__",
                    source_path=None,
                    source_dir=run_dir,
                )
            )
            continue
        cands.append(
            Candidate(
                name=run_dir.name,
                kind="model",
                seat_token=str(ckpt),
                source_path=ckpt,
                source_dir=run_dir,
            )
        )

    # Hall of Fame — every .pt file is its own candidate.
    if HOF_DIR.is_dir():
        for ckpt in sorted(HOF_DIR.glob("*.pt")):
            cands.append(
                Candidate(
                    name=f"HOF/{ckpt.name}",
                    kind="hof",
                    seat_token=str(ckpt),
                    source_path=ckpt,
                    source_dir=None,
                )
            )

    if include_bots:
        for bot in BOT_BASELINES:
            cands.append(
                Candidate(
                    name=bot,
                    kind="bot",
                    seat_token=bot,
                    source_path=None,
                    source_dir=None,
                )
            )

    return cands


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------


def prepare_seat_token(c: Candidate) -> tuple[str, str | None]:
    """Return (seat_token_for_run_self_play, ts_temp_path_or_None).

    For model/hof candidates we export the raw PyTorch checkpoint to a
    TorchScript file that NeuralNetPlayer can load.
    Raises on any load/export failure.
    """
    if c.kind == "bot":
        return c.seat_token, None
    if c.seat_token == "__MISSING__":
        raise FileNotFoundError("no checkpoint file present")
    ts_path = export_checkpoint_to_torchscript(c.seat_token)
    return ts_path, ts_path


def evaluate_candidate(
    c: Candidate,
    opponents: list[str],
    n_games: int,
    concurrency: int,
    session_size: int,
) -> Result:
    assert len(opponents) == 3, "opponents must list three seats"
    result = Result(candidate=c, session_size=session_size)
    t0 = time.perf_counter()
    try:
        seat_token, ts_path = prepare_seat_token(c)
        result.ts_path = ts_path
        seat_config = ",".join([seat_token, *opponents])
        # run_self_play requires model_path iff 'nn' or 'centaur' is in seat_config.
        # We always use explicit paths/bot labels instead, so we pass None.
        raw = te.run_self_play(
            n_games=n_games,
            concurrency=min(concurrency, n_games),
            model_path=None,
            explore_rate=0.0,
            seat_config=seat_config,
            include_replay_data=False,
            include_oracle_states=False,
        )
        scores = np.asarray(raw["scores"])  # shape (n_games, 4)
        seat0 = scores[:, 0]
        result.games_played = int(scores.shape[0])
        result.total_score = int(seat0.sum())
        result.wins = int((seat0 > seat0.max(initial=0) * 0 + 0).sum())  # placeholder, overwrite below
        # "win" in tarok = player has the highest score this game.
        game_winners = scores.argmax(axis=1)
        result.wins = int((game_winners == 0).sum())
        result.positive_games = int((seat0 > 0).sum())
    except Exception as e:  # noqa: BLE001  (we deliberately capture everything)
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        result.duration_s = time.perf_counter() - t0
        if result.ts_path and os.path.exists(result.ts_path):
            try:
                os.unlink(result.ts_path)
            except OSError:
                pass
    return result


# ---------------------------------------------------------------------------
# Reporting / recommendations
# ---------------------------------------------------------------------------


def _fmt_row(rank: int, r: Result) -> str:
    if r.error:
        return (
            f"{rank:>3}. {r.candidate.name:<40}  "
            f"[{r.candidate.kind:<5}]  BROKEN — {r.error[:60]}"
        )
    return (
        f"{rank:>3}. {r.candidate.name:<40}  "
        f"[{r.candidate.kind:<5}]  "
        f"avg_session={r.avg_session_score:+8.1f}  "
        f"win%={100*r.wins/max(r.games_played,1):5.1f}  "
        f"pos%={100*r.positive_games/max(r.games_played,1):5.1f}  "
        f"games={r.games_played}  "
        f"({r.duration_s:.1f}s)"
    )


def print_report(results: list[Result], promote_top_k: int) -> None:
    ok = [r for r in results if r.error is None]
    broken = [r for r in results if r.error is not None]

    ok.sort(key=lambda r: r.avg_session_score, reverse=True)

    print()
    print("=" * 100)
    print(f"Tournament results — {len(ok)} ranked, {len(broken)} broken")
    print("=" * 100)
    for i, r in enumerate(ok, 1):
        print(_fmt_row(i, r))
    if broken:
        print()
        print("Broken / unloadable candidates:")
        for r in broken:
            print(_fmt_row(0, r))

    # Recommendations
    print()
    print("=" * 100)
    print("Recommendations")
    print("=" * 100)

    # Figure out baseline score (mean of bot baselines, if present).
    bot_scores = [r.avg_session_score for r in ok if r.candidate.kind == "bot"]
    baseline = float(np.mean(bot_scores)) if bot_scores else 0.0
    print(f"Baseline (mean of bot baselines): {baseline:+.1f}")

    hof_scores = [r.avg_session_score for r in ok if r.candidate.kind == "hof"]
    hof_floor = min(hof_scores) if hof_scores else baseline
    print(f"HoF floor (worst HoF model):      {hof_floor:+.1f}")
    print()

    # Models to promote to HoF: top-K non-HoF, non-bot candidates that
    # also clear the current HoF floor.
    promote = [
        r for r in ok
        if r.candidate.kind == "model" and r.avg_session_score > hof_floor
    ][:promote_top_k]

    # Models to delete: broken ones OR models scoring well below baseline.
    delete_threshold = baseline - 20.0   # "meaningfully worse than bots"
    delete_dirs: list[Path] = []
    for r in broken:
        if r.candidate.source_dir is not None:
            delete_dirs.append(r.candidate.source_dir)
    for r in ok:
        if r.candidate.kind == "model" and r.avg_session_score < delete_threshold:
            if r.candidate.source_dir is not None:
                delete_dirs.append(r.candidate.source_dir)

    # Keep list = everything that isn't promoted and isn't deleted.
    promoted_dirs = {r.candidate.source_dir for r in promote if r.candidate.source_dir}
    delete_dirs = [d for d in delete_dirs if d not in promoted_dirs]

    if promote:
        print("# Promote these to Hall of Fame:")
        for r in promote:
            ckpt = r.candidate.source_path
            assert ckpt is not None
            dst_name = f"{r.candidate.name.lower()}.pt"
            print(
                f"cp {_q(ckpt)} {_q(HOF_DIR / dst_name)}   "
                f"# avg_session={r.avg_session_score:+.1f}"
            )
        print()

    if delete_dirs:
        print(
            f"# Delete these — broken or clearly worse than "
            f"bot baseline ({delete_threshold:+.1f}):"
        )
        for d in delete_dirs:
            print(f"rm -rf {_q(d)}")
        print()

    # HoF hygiene: suggest dropping HoF entries that are worse than bot baseline.
    bad_hof = [
        r for r in ok
        if r.candidate.kind == "hof" and r.avg_session_score < baseline - 10.0
    ]
    if bad_hof:
        print("# HoF entries weaker than bot baseline — consider pruning:")
        for r in bad_hof:
            assert r.candidate.source_path is not None
            print(
                f"rm {_q(r.candidate.source_path)}   "
                f"# avg_session={r.avg_session_score:+.1f}"
            )
        print()

    if not promote and not delete_dirs and not bad_hof:
        print("(no actions suggested — everything looks roughly in range)")


def _q(p: Path) -> str:
    s = str(p)
    return f"'{s}'" if any(ch.isspace() or not ch.isascii() for ch in s) else s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=500,
                    help="games per candidate (default 500)")
    ap.add_argument("--session-size", type=int, default=50,
                    help="games per session for avg_score (default 50)")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="Rust self-play concurrency (default 64)")
    ap.add_argument("--opponents", type=str, default="bot_m6,bot_v5,bot_m6",
                    help="comma-separated seat labels for the 3 opponents")
    ap.add_argument("--promote-top-k", type=int, default=3,
                    help="top-K models to suggest promoting to HoF (default 3)")
    ap.add_argument("--skip-bots", action="store_true",
                    help="don't evaluate bot_v5 / bot_m6 as candidates")
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on candidate name (for debugging)")
    args = ap.parse_args()

    opponents = [s.strip() for s in args.opponents.split(",") if s.strip()]
    if len(opponents) != 3:
        print(f"--opponents must be 3 seats, got {opponents}", file=sys.stderr)
        return 2

    candidates = discover_candidates(include_bots=not args.skip_bots)
    if args.only:
        candidates = [c for c in candidates if args.only.lower() in c.name.lower()]

    print(f"Discovered {len(candidates)} candidates:")
    for c in candidates:
        print(f"  - {c.name:<40} [{c.kind}]  {c.seat_token}")
    print()
    print(
        f"Running {args.games} games each vs opponents={opponents} "
        f"(session_size={args.session_size}, concurrency={args.concurrency})"
    )
    print()

    results: list[Result] = []
    for i, c in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {c.name} ...", flush=True)
        r = evaluate_candidate(
            c, opponents, args.games, args.concurrency, args.session_size
        )
        results.append(r)
        if r.error:
            print(f"    BROKEN: {r.error}")
        else:
            print(
                f"    avg_session={r.avg_session_score:+.1f}  "
                f"win%={100*r.wins/max(r.games_played,1):.1f}  "
                f"({r.duration_s:.1f}s)"
            )

    print_report(results, promote_top_k=args.promote_top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
