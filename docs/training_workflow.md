# Training Workflow

## Overview

Training a Tarok NN agent proceeds in two phases:

1. **Behavioral Cloning (BC)** — bootstrap from expert play
2. **Self-Play (PPO)** — improve through self-play with optional centaur endgame solver

Both phases use the same `TarokNetV4` architecture.

---

## Phase 1: Behavioral Cloning from bot_v5

A fresh model starts with random weights.  Pure self-play from scratch is
inefficient because the agent must re-discover basic Tarok heuristics.
Instead, we first clone the strongest heuristic bot (bot_v5/v6) via
supervised learning on its action distributions.

```yaml
# configs/behavioral-clone.yaml (example)
seats: "nn,bot_v5,bot_v5,bot_v5"
behavioral_clone_coef: 1.0
behavioral_clone_teacher: "bot_v5"
behavioral_clone_games_per_iteration: 10000
iterations: 20
```

This produces a model that plays at roughly bot_v5 strength — good enough
for self-play to take over.

**Q: Do I still behavioral-clone the v5 bot initially to train a new model?**

Yes.  BC provides the starting point for both the pure-NN and centaur
training paths.  Without it, the untrained NN's random play means the
centaur's PIMC endgame is the *only* source of reasonable actions, making
early-game learning extremely slow.

---

## Phase 2: Self-Play with PPO

Once the model reaches bot-level play, switch to PPO self-play:

```yaml
# configs/self-play.yaml
seats: "centaur,bot_v6,bot_v6,bot_v6"
centaur_handoff_trick: 8     # endgame solver from trick 8 onward
centaur_pimc_worlds: 100     # worlds sampled per decision
# centaur_endgame_solver: pimc   # pimc (default) or alpha_mu
# centaur_alpha_mu_depth: 2      # αμ depth (only for alpha_mu solver)
```

### Seat types

| Label      | Description |
|------------|-------------|
| `nn`       | Pure neural network — all decisions from the NN |
| `centaur`  | NN for early/mid-game + endgame solver for last N tricks |
| `bot_v5`   | Heuristic bot v5 (Stockskis) |
| `bot_v6`   | Heuristic bot v6 |
| `bot_m6`   | Heuristic bot m6 (with PIMC endgame on main branch) |
| `path.pt`  | Frozen NN checkpoint (plays greedily) |

Both `nn` and `centaur` are *learner* seats — their experiences feed into PPO.

### Centaur mode

The centaur player is a hybrid:
- **Early/mid-game** (tricks 0 through `handoff_trick - 1`): the NN decides
  all actions (bidding, king calling, talon exchange, card play).
- **Endgame** (tricks `handoff_trick` through 11): a search-based endgame
  solver takes over card play.  Bidding and other decisions are still NN.

Endgame decisions are tagged with `log_prob = NaN` so PPO skips their policy
loss while the terminal reward still flows back through GAE to improve the
NN's earlier decisions.

### Endgame solvers

| Solver     | Algorithm | Strengths |
|------------|-----------|-----------|
| `pimc`     | Perfect-Information Monte Carlo | Fast. Samples worlds, DD-solves each independently. |
| `alpha_mu` | αμ (Cazenave & Ventos, 2019) | Enforces strategy fusion via Pareto fronts. Stronger but slower. |

PIMC is the default.  AlphaMu at depth M=1 is equivalent to PIMC; depths
2-3 give measurable improvement at the cost of more computation per move.

---

## Architecture: Learner abstraction

Seat labels are classified into *learner* and *bot* categories:

```python
# training/entities/training_config.py
LEARNER_SEAT_LABELS = frozenset({"nn", "centaur"})
```

This constant is used consistently across:
- `TrainingConfig.nn_seat_indices` / `bot_seat_indices`
- `CollectExperiences` (which seats to emit PPO experiences for)
- `RustSelfPlay.compute_run_stats` (outplacement scoring)
- Rust `learner_seat_mask` in `py_bindings.rs`

To add a new learner seat type (e.g., a future "oracle" player), add it to
`LEARNER_SEAT_LABELS` and ensure the Rust `run_self_play` handles it.

---

## Typical training progression

```
1. Behavioral clone from bot_v5  (20 iterations, ~200K games)
   → model at ~bot_v5 strength

2. Self-play with centaur/PIMC   (100+ iterations, ~500K games)
   seats: "centaur,bot_v6,bot_v6,bot_v6"
   → NN learns from PIMC-optimal endgame rewards

3. Optional: switch to alpha_mu  (slower but stronger endgame)
   centaur_endgame_solver: alpha_mu
   centaur_alpha_mu_depth: 2

4. Optional: pure NN evaluation
   seats: "nn,bot_v6,bot_v6,bot_v6"
   → measure NN-only strength (no search at inference time)
```

The centaur mode acts as a training accelerator: the NN benefits from
near-perfect endgame play without needing to learn it from scratch.  At
deployment, you can use either `nn` (fast, no search) or `centaur`
(slightly slower, stronger endgame).
