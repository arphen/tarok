# League Play — Developer Guide

Multi-agent league play allows each training iteration to seat the learner against a dynamically-sampled mix of frozen NN checkpoints and heuristic bots. Opponents are tracked with Elo ratings. The learner is periodically snapshotted into the pool, producing AlphaStar-style historical self-play automatically.

---

## File Structure

```text
training-lab/training/
├── entities/
│   ├── league.py              ← NEW: all league domain types
│   └── training_config.py     ← EDIT: add `league` field
│
├── use_cases/
│   ├── sample_league_seats.py ← NEW: per-iteration seat sampling
│   ├── update_league_elo.py   ← NEW: Elo update after each iteration
│   ├── run_iteration.py       ← EDIT: seats_override param + mean_scores output
│   └── train_model.py         ← EDIT: wire all of the above together
│
├── entities/iteration_result.py ← EDIT: add seat_config_used + mean_scores
├── ports/presenter_port.py    ← EDIT: two new optional callbacks
└── adapters/presenter.py      ← EDIT: implement new callbacks


engine-rs/src/py_bindings.rs   ← EDIT: path-based seat type in run_self_play

training-lab/training/use_cases/resolve_config.py  ← EDIT: parse league: block
training-lab/configs/                              ← ADD: league YAML examples
```

---

## What Lives Where and Why

### entities/league.py — Pure domain types, no I/O

| Class | Frozen? | Purpose |
|---|---|---|
| LeagueOpponent | ✓ | Name, type (`nn_checkpoint`/`bot_v5`/`bot_v6`/`bot_m6`), optional path |
| LeagueConfig | ✓ | Config from YAML: `enabled`, `opponents`, `min_nn_per_game`, `sampling`, `pfsp_alpha`, `snapshot_interval`, `elo_outplace_unit_weight` |
| LeaguePoolEntry | ✗ | Live runtime entry per opponent: Elo, `games_played`, `learner_outplaces`, `outplace_rate` property |
| LeaguePool | ✗ | Aggregate root (mutable, like `TrainingRun`). Has `add_snapshot()` and `sampling_weights()` |

Rule: if it's a pure data shape with no logic beyond properties, it goes here. No external imports.

---

### use_cases/sample_league_seats.py — How seats are chosen each iteration

Pure logic, no I/O. Takes a LeaguePool, returns a seat_config string.

- Seat 0 is always "nn" — immutable rule
- Seats 1–3 drawn from pool.entries weighted by pool.sampling_weights()
- Enforces min_nn_per_game by replacing slots if needed
- nn_checkpoint opponents appear as their .path, bots as their type string

Why separate: seat sampling is non-trivial (PFSP weighting, min-NN enforcement) and changes independently of Elo logic.

---

### use_cases/update_league_elo.py — Elo accounting after each iteration

Pure logic. Mutates LeaguePool entries in place.

- Input metric: `seat_outcomes[seat_idx] = (learner_outplaces, opponent_outplaces, draws)`
- Per game, outplace means score comparison vs seat 0 learner:
  - learner_outplaces: learner_score > opponent_score
  - opponent_outplaces: learner_score < opponent_score
  - draws: learner_score == opponent_score
- For each seat 1–3 mapped to a named pool entry: pairwise Elo update vs seat 0 (K=32)
- Same opponent token in multiple seats: updated independently per seat

Elo translation for one seat uses:

```text
n_games = learner_outplaces + opponent_outplaces + draws
learner_outcome = (learner_outplaces + 0.5 * draws) / n_games
expected = 1 / (1 + 10 ^ ((opp_elo - learner_elo) / 400))
k_effective = 32 * max(1.0, elo_outplace_unit_weight)
learner_elo += k_effective * (learner_outcome - expected)
```

Important: opponent/snapshot Elo values remain fixed. Only learner Elo moves.

### Outplace Window — Is It "After 50 Games"?

No. Outplace is computed over every game in each training iteration, then accumulated.

- Per iteration:
  - If `games: 10000`, then the outplace counts for that iteration are based on 10,000 game-level comparisons.
- Across training:
  - `games_played` and `learner_outplaces` are cumulative for each pool entry.
  - `outplace_rate = learner_outplaces / games_played` (0.5 prior when games_played is 0).

So in warm-up config (`iterations: 50`, `games: 10000`), the number 50 is iterations, not a 50-game outplace window.
The effective outplace sample size per opponent depends on seat sampling and how often that opponent appears in seats 1..3.
By default, `elo_outplace_unit_weight` falls back to `outplace_session_size`; you can override it in `league:`.

PFSP weighting (sampling: pfsp): weight_i = win_rate_i ^ pfsp_alpha. Higher alpha concentrates sampling on the hardest opponents.

---

### entities/iteration_result.py — Two new fields

| Field | Type | Source |
|---|---|---|
| seat_config_used | str | effective_seats from RunIteration |
| mean_scores | tuple[float,float,float,float] | np.mean(raw["scores"], axis=0) |

---

### use_cases/run_iteration.py — Minimal surgery

One new param: seats_override: str | None = None. effective_seats = seats_override or config.seats. nn_seats/bot_seats derived locally — path-based tokens (.pt paths) fall into neither, so PPO ignores their experiences.

---

### use_cases/train_model.py — The wiring hub (~20 new lines)

Before loop: build LeaguePool from config.league.
Per-iteration (all guarded by if pool):
1. SampleLeagueSeats → seats_override
2. After iteration: UpdateLeagueElo → mutate pool → presenter callback
3. Every snapshot_interval iters: copy checkpoint → league_pool/iter_NNN.pt → pool.add_snapshot()

League concerns are cross-iteration so they live here, not in RunIteration.

---

### py_bindings.rs — Frozen NN opponents in Rust

New arm in seat parsing: label ending in .pt / containing / → load NeuralNetPlayer(label, ...), cached by path in a HashMap. After: cd engine-rs && maturin develop --release.

---

## YAML Config Shape

```yaml
league:
  enabled: true
  min_nn_per_game: 2        # learner occupies at least this many seats
  sampling: pfsp            # uniform | pfsp | hardest
  pfsp_alpha: 1.5
  snapshot_interval: 5      # auto-snapshot every N iterations

  opponents:
    - name: AnaVesel
      type: nn_checkpoint
      path: checkpoints/Ana_Vesel/_current.pt
    - name: StockSkisV5
      type: bot_v5
    - name: StockSkisM6
      type: bot_m6
```

---

## Data Flow

```text
TrainModel.execute()
│
├─ LeaguePool (built once, mutated each iteration)
│
└─ for each iteration i:
    ├─ SampleLeagueSeats(pool) ──► "nn,checkpoints/Ana...,bot_v5,nn"
    │
    ├─ RunIteration.execute(seats_override=...)
    │   ├─ RustSelfPlay → raw["scores"]: (n_games, 4)
    │   ├─ compute_run_stats → seat_outcomes per opponent seat
    │   ├─ PPOAdapter.update (seat 0 only)
    │   ├─ Benchmark
    │   └─► IterationResult(mean_scores, seat_config_used, seat_outcomes)
    │
    ├─ UpdateLeagueElo(result.seat_outcomes, ...) — mutates pool entries
    │
    └─ every snapshot_interval:
        copy ts_path → league_pool/iter_NNN.pt
        pool.add_snapshot(...)   ← pool grows; next iteration can sample it
```

---

## Adding a Checkpoint Manually

```yaml
league:
  opponents:
    - name: MyBestModel
      type: nn_checkpoint
      path: checkpoints/Petra_Golob/_current.pt
```

No code changes — just restart training.
