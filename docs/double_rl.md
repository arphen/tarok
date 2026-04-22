# Duplicate Reinforcement Learning (DRL) for Tarok

> Status: **Design / Implementation Plan** — not yet built.
> Guiding principle: **strictly additive on top of the existing PPO + Fictitious Self-Play + Arena stack.** Nothing about current training, arena leaderboard, or ELO league changes unless the Duplicate feature flag is explicitly enabled.

---

## 1. Motivation and Background

### 1.1 The variance problem in PPO for Tarok

Tarok is a partnership-based imperfect-information game in the same family as Contract Bridge. Any given hand has a large luck component:

- Being dealt the Škis / Mond / Pagat (trula) is worth roughly +30 points before the agent makes a single decision.
- Partnership assignment via the king call routes card luck asymmetrically to players.
- Klop distributes card luck across all four seats.

In vanilla PPO the gradient signal is $A = R - V(s)$ where $V(s)$ is a learned critic. The critic has to guess the expected value of the hand through *all* of this variance. The result: extremely noisy gradients, slow convergence, and sensitivity to matchmaking drift.

### 1.2 Prior art

The techniques below are lifted directly from computer bridge and are the basis for this design.

| Paper | Contribution |
|---|---|
| *Simple is Better: Training an End-to-end Contract Bridge Bidding Agent without Human Knowledge* (2019) | Duplicate reward signal — each hand is played at two tables; reward is the difference. |
| *Joint Policy Search for Multi-agent Collaboration with Imperfect Information* (NeurIPS 2020) | Formal bot-vs-bot evaluation in IMPs/board on 1,000 duplicate boards. |
| *A new rating system for duplicate bridge* (Quint & Michael, 2007) | Isolating individual skill from partnership scores; justifies Elo over rotated duplicate seating. |
| *A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI* (2024) | Standardized open-source duplicate benchmarking methodology. |

### 1.3 The core idea

> Deal one hand. Play it N times from N different seat assignments. Subtract a **baseline twin** playthrough (same deck, same opponents, same seats, but learner replaced by a frozen snapshot of itself). The difference **is** the empirical advantage.

Because the deck and every non-learner opponent are byte-for-byte identical across the learner-table and the shadow-table, every source of variance that is not caused by the learner's new policy cancels exactly. What remains is a near-zero-variance gradient signal.

---

## 2. Design Decisions (locked)

These decisions are the conclusions of an extensive design discussion. They are captured here so the implementation doesn't relitigate them.

### 2.1 Do not use Elo as a reward signal
Elo is a trailing, population-level, non-stationary metric that discards magnitude. It would cause catastrophic non-stationarity in the PPO optimizer. Elo remains what it is today: a **league rating / matchmaking signal**, computed exactly as before from session outplacement outcomes.

### 2.2 The Duplicate Advantage (= empirical $A$)
For each duplicate pod:

$$
A_{\text{duplicate}} = R_{\text{learner}} - R_{\text{shadow}}
$$

where:
- $R_{\text{learner}}$: learner's per-game score in a game it participated in.
- $R_{\text{shadow}}$: score of the shadow (frozen previous-iteration snapshot) in the paired baseline game.

This replaces **only the per-trajectory reward fed into GAE** — everything else in PPO (ratio, clip, value loss, entropy, behavioral cloning, oracle distillation) is untouched.

### 2.3 The Shadow Baseline is the learner's *own previous snapshot*
Not `bot_v5`. Not a static fixed bot. The shadow is the learner frozen at the start of the current PPO iteration. This gives a self-referential advantage — "am I better than I was one iteration ago?" — which keeps the baseline co-evolving with the learner and eliminates overfitting to a specific bot's weaknesses.

### 2.4 `min_nn = 1` during duplicate rollouts
Having two exploring learners at one table entangles gradients: a good decision by Learner-North gets punished for a random exploration blunder by Learner-South, because both receive the same reward. **Inside a duplicate pod, at most one learner seat exists per game.** This is a hard invariant, not a knob.

This does **not** change the existing `min_nn_per_game` behavior of the standard (non-duplicate) self-play path — that pipeline continues to exist unmodified.

### 2.5 The Pod Structure (8 games per deck)
For a single RNG-seeded deal, a **Training Pod** consists of:

- **Active group (4 games, generate gradients):**
  - Game 1: `[Learner, opp_A, opp_B, opp_C]`
  - Game 2: `[opp_C, Learner, opp_A, opp_B]`
  - Game 3: `[opp_B, opp_C, Learner, opp_A]`
  - Game 4: `[opp_A, opp_B, opp_C, Learner]`
- **Shadow group (4 games, generate baseline):**
  - Games 5–8: identical seatings but with `Learner` replaced by `Shadow`.

All 8 games use the **same deck seed** and the **same `opp_A/B/C` identities** (sampled once per pod from the League Pool). This produces four (learner-trajectory, shadow-score) pairs per deal — enough to yield four PPO trajectories per pod with pure duplicate advantage.

Operators can also configure **4-game pods** (learner-vs-shadow in a single seat, no rotation) for cheaper variance reduction, or **2-game pods** (a single learner-seat + its shadow) for maximum throughput. The 8-game pod is the default because it also teaches the agent to handle all four seat positions on the same deck.

### 2.6 Coexistence, not replacement
The existing PPO path (`run_self_play` → `CollectExperiences` → `TorchPPO`) remains the default. Duplicate RL is opt-in via config flag (`duplicate.enabled: true`). Both paths must continue to work in CI.

---

## 3. Architecture — Ports, Adapters, Use Cases

The goal is to embody the duplicate-RL logic as **use cases** (Uncle Bob style) sitting on top of three new **ports**, with all infrastructure (Rust self-play, score bookkeeping, Torch PPO) hidden behind adapters.

### 3.1 Where the new code lives

```
training-lab/training/
├── entities/
│   ├── duplicate_pod.py           # NEW: DuplicatePod dataclass + PodSeating
│   └── duplicate_reward.py        # NEW: DuplicateRewardSpec
├── ports/
│   ├── duplicate_pairing_port.py  # NEW: how to build pods from a league pool
│   ├── duplicate_reward_port.py   # NEW: how to compute per-trajectory reward from paired scores
│   └── selfplay_port.py           # EXTENDED: optional seeded / per-game seats
├── use_cases/
│   └── collect_duplicate_experiences.py  # NEW: parallel to CollectExperiences
└── adapters/
    ├── duplicate/
    │   ├── rotation_pairing.py           # NEW: 8-game rotated pod builder
    │   ├── shadow_score_reward.py        # NEW: R_learner − R_shadow reward model
    │   └── seeded_self_play_adapter.py   # NEW: wraps run_self_play with seed + per-game seats
    └── ppo/
        └── ppo_batch_preparation.py      # EXTENDED: accept precomputed per-trajectory rewards
```

Nothing under `backend/src/tarok/use_cases/` changes. Nothing under `backend/src/tarok/adapters/` changes. This keeps the inference backend pristine.

### 3.2 Port 1 — `DuplicatePairingPort`

**Responsibility:** given the current league pool and a learner identity, produce the schedule of games that make up one training iteration's worth of duplicate pods.

```python
# training-lab/training/ports/duplicate_pairing_port.py
class DuplicatePairingPort(ABC):
    @abstractmethod
    def build_pods(
        self,
        pool: LeaguePool,
        learner_seat_token: str,
        shadow_seat_token: str,
        n_pods: int,
        rng_seed: int,
    ) -> list[DuplicatePod]: ...
```

Where `DuplicatePod` is:

```python
@dataclass(frozen=True)
class DuplicatePod:
    deck_seed: int
    opponents: tuple[str, str, str]          # seat tokens for the 3 non-learner seats
    active_seatings: tuple[tuple[str, str, str, str], ...]   # 4 rotated active games
    shadow_seatings: tuple[tuple[str, str, str, str], ...]   # 4 matched shadow games
    learner_positions: tuple[int, int, int, int]             # seat idx of learner per active game
```

The `RotationPairingAdapter` is the default adapter. Alternative future adapters (e.g., "2-game pod" for cheap experiments, "full 24! permutation" for research) can drop in without touching the use case.

**Why a port:** the pairing policy is a separable research axis. Swapping it should not require touching PPO batching, reward computation, or Rust.

### 3.3 Port 2 — `DuplicateRewardPort`

**Responsibility:** take the raw score tensors produced by the Rust self-play runs and return the per-trajectory reward array that will replace `scores_np[gids, players_np] / 100` in `ppo_batch_preparation.py`.

```python
# training-lab/training/ports/duplicate_reward_port.py
class DuplicateRewardPort(ABC):
    @abstractmethod
    def compute_rewards(
        self,
        active_raw: dict[str, Any],     # experience dict from active games only
        shadow_scores: np.ndarray,       # shape (n_pods, 4) — shadow's score per active game
        pods: list[DuplicatePod],
    ) -> np.ndarray:                     # shape (n_active_steps,) — per-step terminal reward
        ...
```

The default adapter `ShadowScoreRewardAdapter` implements:

```
reward[step] = (learner_score_in_game(g) − shadow_score_matched_to_g) / 100
```

Non-terminal steps still get zeroed in `ppo_batch_preparation` as today — the reward is only meaningful at trajectory end.

**Alternative adapters** this port enables without further refactor:
- `IMPsRewardAdapter` — map raw point deltas onto bridge-style IMPs for smoother gradients.
- `RankingRewardAdapter` — reward is +1 / 0 / −1 based on whether learner outplaces shadow on that pod.
- `ShapedDuplicateRewardAdapter` — composite of duplicate advantage + small trick-point shaping; for curriculum phases.

**Why a port:** the reward model is the *most* research-sensitive component. It is where duplicate RL meets bridge-research tradition. Whole papers are about choice of reward here (raw-diff vs IMPs vs matchpoints vs ranking).

### 3.4 Port 3 — extension to `SelfPlayPort`

Today `SelfPlayPort.run(...)` takes a single `seat_config` string and **no seed**. Rust internally `rand::rng()`'s, so we cannot currently reproduce a deal. Two additive extensions are required:

```python
class SelfPlayPort(ABC):
    # existing run(...) stays exactly as-is — no signature break.

    @abstractmethod
    def run_seeded_pods(
        self,
        learner_path: str,
        shadow_path: str,
        pods: list[DuplicatePod],
        explore_rate: float,
        concurrency: int,
        include_oracle_states: bool = False,
        # ... existing centaur knobs forwarded unchanged ...
    ) -> DuplicateRunResult: ...
```

Where `DuplicateRunResult`:

```python
@dataclass
class DuplicateRunResult:
    active: dict[str, Any]    # same schema as raw today, but only active-group steps
    shadow_scores: np.ndarray # shape (n_pods, 4_games_per_pod) aligned with pods
    pod_ids: np.ndarray       # shape (n_active_steps,)
    per_game_seats: np.ndarray # shape (n_active_games, 4) — seat tokens as ints
```

**Implementation surface:** a single new Rust binding (details in §5) that accepts a list of `(deck_seed, seat_config)` tuples and runs them deterministically.

### 3.5 Use case — `CollectDuplicateExperiences`

This is the new orchestrator. It parallels `CollectExperiences` one-for-one:

```python
class CollectDuplicateExperiences:
    def __init__(
        self,
        pairing: DuplicatePairingPort,
        selfplay: SelfPlayPort,
        reward: DuplicateRewardPort,
        ppo: PPOPort,
        presenter: PresenterPort,
    ) -> None: ...

    def execute(
        self,
        config: TrainingConfig,
        identity: ModelIdentity,
        learner_ts_path: str,
        shadow_ts_path: str,
        pool: LeaguePool,
    ) -> ExperienceBundle: ...
```

Steps:

1. Ask `pairing` for `n_pods = config.duplicate.pods_per_iteration` pods, seeded by `config.duplicate.rng_seed + iter`.
2. Ask `selfplay.run_seeded_pods(...)` to run the 8N games deterministically.
3. Ask `reward.compute_rewards(active_raw, shadow_scores, pods)` for the per-step reward tensor.
4. **Inject** the reward tensor into `active_raw` under a new key `precomputed_rewards`.
5. Delegate blending (human replays, expert BC data) to `ppo` exactly as `CollectExperiences` does — unchanged.
6. Return `ExperienceBundle` identical in shape to today's, so `TrainModelOrchestrator` is agnostic.

`TrainModelOrchestrator` picks between `CollectExperiences` and `CollectDuplicateExperiences` based on `config.duplicate.enabled`. Nothing deeper changes.

### 3.6 Clean-architecture guardrails

- No new imports of `numpy`, `torch`, `tarok_engine`, etc. inside the two new ports — ports stay pure.
- `CollectDuplicateExperiences` may import only entities and ports. No Rust, no Torch, no filesystem.
- Every new adapter lives under `training/adapters/duplicate/` and is wired up in `training-lab/training/adapters/factories.py` behind a feature flag.

This mirrors the existing Tarok architecture style (see [.github/copilot-instructions.md](.github/copilot-instructions.md) "Clean Architecture Guardrails").

---

## 4. Changes to PPO batch preparation

[training-lab/training/adapters/ppo/ppo_batch_preparation.py](training-lab/training/adapters/ppo/ppo_batch_preparation.py) currently computes:

```python
rewards_np = scores_np[gids, players_np].astype(np.float32) / 100.0
rewards_np = rewards_np + shaped_bonus_by_game[gids, players_np]
```

The change is **minimal and additive**:

```python
precomputed = raw.get("precomputed_rewards")
if precomputed is not None:
    rewards_np = np.asarray(precomputed, dtype=np.float32)
else:
    rewards_np = scores_np[gids, players_np].astype(np.float32) / 100.0
    rewards_np = rewards_np + shaped_bonus_by_game[gids, players_np]
```

Everything downstream — trajectory-key sort, terminal masking, `te.compute_gae`, global advantage normalisation, mini-batching — is untouched. This is the key architectural win: duplicate RL plugs into PPO as a **reward source**, not a PPO variant.

**Shaped bonuses under duplicate RL.** The existing `_compute_special_shaped_bonus_by_game` adds small trick-point shaping. Under pure duplicate RL these bonuses also cancel (both tables experience the same deck), so they are **disabled by default** when `duplicate.enabled`. A config flag `duplicate.apply_shaped_bonuses` can override.

---

## 5. Rust engine changes

### 5.1 New PyO3 binding

Add to [engine-rs/src/py_bindings.rs](engine-rs/src/py_bindings.rs):

```rust
#[pyfunction]
#[pyo3(signature = (pods, learner_model_path, shadow_model_path, explore_rate, concurrency, ...))]
pub fn run_seeded_pods(
    py: Python,
    pods: Vec<PodSpec>,           // each: deck_seed, active_seats[4][4], shadow_seats[4][4]
    learner_model_path: String,
    shadow_model_path: String,
    explore_rate: f32,
    concurrency: usize,
    include_oracle_states: bool,
    // ...existing centaur knobs...
) -> PyResult<PyObject>
```

Where `PodSpec` is a plain PyO3 struct carrying the 8-game schedule for one deck.

### 5.2 Seeded dealing in `SelfPlayRunner`

Today [engine-rs/src/self_play.rs](engine-rs/src/self_play.rs) calls `rand::rng()`. We add a second path that takes `deck_seed: u64` and constructs an `SmallRng::seed_from_u64(seed)` for the deal only (policy-side exploration stays stochastic — that's the whole point).

**Key invariant:** two games that share the same `deck_seed` receive the **byte-for-byte identical deal**, regardless of the seat configuration. Shuffling must not consume RNG entropy from players.

### 5.3 Per-game seat configs

`SelfPlayRunner::run` today builds one `[Arc<dyn BatchPlayer>; 4]` for the whole batch. The new entrypoint builds a fresh `[Arc<dyn BatchPlayer>; 4]` **per game** from the `PodSpec`. Player objects (NN / bot / centaur / snapshot) are constructed once and *referenced* by the per-game seating to avoid model reload cost.

### 5.4 Result shape

The binding returns:

```python
{
    # active group, stitched across all pods — same schema as today's run_self_play
    "states": ..., "actions": ..., "log_probs": ..., "values": ...,
    "decision_types": ..., "game_ids": ..., "players": ..., "scores": ...,
    "legal_masks": ..., "game_modes": ...,

    # duplicate-specific
    "pod_ids": np.ndarray,        # shape (n_active_steps,)
    "shadow_scores": np.ndarray,  # shape (n_pods, 4) — one per active game in the pod
    "learner_positions": np.ndarray,  # shape (n_pods, 4)
}
```

This keeps `ppo_batch_preparation.py` able to use the same GAE / sort path on the active-group slice.

### 5.5 Throughput

8 games per pod doubles CPU cost vs a traditional 4-game random batch of same-deck usefulness. In practice:

- Shadow games use a **frozen** network → no gradient pass, no optimizer, full fp16/jit compatible.
- Shadow games can be generated on the **same batch** as active games — Rust's `BatchPlayer` already groups by player type, so the NN inference for both is fused.
- Given the repo's ~50–66 games/s NN-bound training throughput, we expect duplicate RL to run at ~25–35 games/s effective learner-trajectory rate. That is acceptable in exchange for the drop in gradient variance, per §1.1.

---

## 6. Shadow snapshot management

The shadow is "the learner, frozen at iteration start". Two implementation paths:

### 6.1 In-memory shadow (preferred)

At the start of each iteration, `TrainModelOrchestrator` writes the current `state_dict` to a tempfile (`shadow_iter_{k}.pt`) and passes its path as `shadow_ts_path`. The file is deleted at iteration end. **Zero impact on the league pool and hall of fame.** This is the baseline design.

### 6.2 League-pool shadow (optional, later)

For a "k-step shadow" variant (e.g., shadow = learner from 5 iterations ago), we can mark one `LeaguePoolEntry` as `role = "shadow"` and rotate it every N iterations. This reuses existing pool plumbing and Elo maintenance verbatim — the shadow role just means "always seated opposite the learner in duplicate pods, with score subtracted for reward computation".

Phase 1 ships with in-memory shadow only. Phase 2 adds the pool-based shadow role behind a config flag.

---

## 7. Interaction with the existing League / Elo system

**Unchanged:**
- `LeaguePool`, `LeagueOpponent`, [sample_league_seats.py](training-lab/training/use_cases/sample_league_seats.py), [maintain_league_pool.py](training-lab/training/use_cases/maintain_league_pool.py) stay as today.
- PFSP / matchmaking / uniform sampling of opponents is still the source of `opp_A/B/C` inside a duplicate pod. The pairing adapter calls the same `sample_league_seats` use case per pod.
- Elo updates are still driven by outplacement outcomes in sessions. The duplicate pods also feed outplacement stats (both active and shadow games count, since they are real, scored games).
- Snapshot interval, `max_active_snapshots`, PFSP $\alpha$, hall-of-fame promotion rules — all unchanged.

**New:**
- A duplicate iteration produces $8N$ games. Half of these are shadow games and **must not** be attributed to the learner in the league-pool Elo updates (they are the baseline twin). A small change in `maintain_league_pool.py` partitions games by `role ∈ {learner, shadow, opponent}` before computing outplacements. The default `role` for the existing path is `learner` for nn seats — i.e., backward-compatible.
- A new metric in `TrainingMetrics`: `duplicate_advantage_mean`, `duplicate_advantage_std`. Exposed in the frontend TrainingDashboard under a new "Duplicate RL" collapsible panel (additive — no re-layout of existing charts).

---

## 8. Arena integration

The arena use case ([backend/src/tarok/use_cases/arena.py](backend/src/tarok/use_cases/arena.py)) gains a new **mode** that runs duplicate games between two nominated checkpoints on the same seeds. This is the "proof-of-strength" tool motivated by bridge-literature §1.2.

- CLI: `python -m tarok arena duplicate --challenger <ckpt_A> --defender <ckpt_B> --boards 1000 --seed 42`.
- Output: `boards_played`, `challenger_mean_score`, `defender_mean_score`, `mean_duplicate_advantage`, `imps_per_board`, 95% CI via bootstrap.
- Rendered as a new "Duplicate Match" card on the Arena page.

Critically, this arena mode shares the same `SelfPlayPort.run_seeded_pods` + `DuplicateRewardPort.compute_rewards` adapters used in training. **One implementation; two call sites.** This respects the project guardrail *"Single game engine path: always use `run_self_play` for arena games"* (from [.github/copilot-instructions.md](.github/copilot-instructions.md)) — we extend, not fork.

No change to leaderboard math, Avg Score semantics, session-based metrics, or zero-sum scoring. Those are locked.

---

## 9. Configuration

Extend [training-lab/configs/self-play.yaml](training-lab/configs/self-play.yaml) additively:

```yaml
duplicate:
  enabled: false                   # master switch
  pairing: rotation_8game          # "rotation_8game" | "rotation_4game" | "single_seat_2game"
  pods_per_iteration: 400          # produces 4 × 400 = 1600 learner trajectories/iter
  shadow_source: previous_iteration   # "previous_iteration" | "league_pool_role"
  apply_shaped_bonuses: false      # under duplicate, shaping is redundant
  reward_model: shadow_score_diff  # "shadow_score_diff" | "imps" | "ranking"
  rng_seed: 0                      # deterministic pod schedule
```

Config resolution lives in [training-lab/training/use_cases/resolve_config.py](training-lab/training/use_cases/resolve_config.py) (existing). A dataclass `DuplicateConfig` is added to `TrainingConfig`; absent/disabled means the existing path runs unchanged.

---

## 10. Testing strategy

| Layer | Test | Location |
|---|---|---|
| Rust | Same `deck_seed` → identical deal across two independent runner invocations. | `engine-rs/tests/duplicate_seeded_deal.rs` (new) |
| Rust | 8-game pod with learner-NN scores recoverable equally from any seat rotation. | same file |
| Port | `DuplicatePairingPort.build_pods` returns 8 seatings with invariants: each seat hosts learner exactly once in active, and opponent identity set is constant across pod. | `backend/tests/test_duplicate_pairing.py` |
| Port | `DuplicateRewardPort.compute_rewards` on synthetic `active_raw` + `shadow_scores` returns exactly `(learner_score − matched_shadow_score) / 100`. | `backend/tests/test_duplicate_reward.py` |
| Use case | `CollectDuplicateExperiences.execute` with fakes for all three ports returns a well-shaped `ExperienceBundle` and does not import `numpy`/`torch`/`tarok_engine` at module top. | `backend/tests/test_collect_duplicate_experiences.py` |
| Integration | End-to-end iteration with `duplicate.enabled=true` on 4 pods runs, updates weights, and writes a checkpoint whose Elo against `bot_v5` is at least as high as the starting checkpoint's. | `training-lab/tests/test_duplicate_iteration.py` |
| Regression | `duplicate.enabled=false` produces **bit-identical** gradients to `main` on a fixed seed. | `training-lab/tests/test_duplicate_disabled_is_noop.py` |
| Architecture | `make lint-architecture` passes with new ports/adapters/use-cases. The two new ports import nothing outside `training.entities`. | existing import-linter |

The `test_duplicate_disabled_is_noop.py` is a critical gate — it guarantees that shipping the feature cannot silently regress the existing pipeline.

---

## 11. Phased rollout

### Phase 1 — Plumbing (no training yet)
1. Rust: add `deck_seed` to the existing `SelfPlayRunner` internals. Add `run_seeded_pods` PyO3 binding returning active + shadow scores. Cargo + PyO3 tests.
2. Python ports: add `DuplicatePairingPort`, `DuplicateRewardPort`, `SelfPlayPort.run_seeded_pods`. Dataclasses in `entities/`.
3. Adapters: `RotationPairingAdapter`, `ShadowScoreRewardAdapter`, `SeededSelfPlayAdapter`.
4. Unit tests for the three port contracts.

**Exit criterion:** `SeededSelfPlayAdapter.run_seeded_pods(...)` runs 1 pod end-to-end and returns a `DuplicateRunResult` whose active and shadow games are confirmed to use the same deck.

### Phase 2 — PPO wiring
5. `CollectDuplicateExperiences` use case.
6. Minimal patch to `ppo_batch_preparation.py` (the `precomputed_rewards` fallback).
7. Config `DuplicateConfig` + YAML plumbing.
8. `TrainModelOrchestrator` branching on `config.duplicate.enabled`.
9. The disabled-is-noop regression test.
10. Full integration test: 4-pod iteration produces non-zero advantages, updated weights, checkpoint survives reload.

**Exit criterion:** 100-pod smoke training run completes without crash, `duplicate_advantage_std` decreases over 5 iterations.

### Phase 3 — Arena & UI
11. `arena duplicate` CLI subcommand.
12. Arena page: new "Duplicate Match" card.
13. TrainingDashboard: "Duplicate RL" collapsible panel with advantage histogram per session.
14. Frontend Playwright test covering the new UI affordances.

### Phase 4 — Research knobs
15. `IMPsRewardAdapter` + `RankingRewardAdapter` alternatives.
16. League-pool shadow role (shadow = learner from k iterations ago).
17. Hyperparameter sweep: pod size (2/4/8), `pods_per_iteration`, shadow staleness.

---

## 12. Mathematical summary

For a pod $p$ with deck $d_p$, opponent sample $O_p = (o_{A,p}, o_{B,p}, o_{C,p})$, and learner seat rotation $\pi \in \{0,1,2,3\}$:

$$
A_{p,\pi} = R_\text{learner}(d_p, O_p, \pi) - R_\text{shadow}(d_p, O_p, \pi)
$$

The variance of $A_{p,\pi}$ is:

$$
\text{Var}(A) = \text{Var}(R_\text{learner}) + \text{Var}(R_\text{shadow}) - 2\,\text{Cov}(R_\text{learner}, R_\text{shadow})
$$

Because shadow and learner share the deal, opponents, and seating, $\text{Cov}(R_\text{learner}, R_\text{shadow})$ approaches $\text{Var}(R_\text{shadow})$, and the expression collapses toward $\text{Var}(R_\text{learner} - R_\text{shadow}) \to \text{Var}(\Delta\text{policy})$ — only the variance caused by the learner's *policy difference from its past self* remains. That is exactly what PPO is supposed to be optimising.

---

## 13. Open questions (deliberately deferred)

1. **Bid/discard coverage.** Discards are still heuristic today. A duplicate advantage on a hand whose decisive moment was a discard has zero gradient to apply. Worth investigating whether the duplicate advantage should be attributed only to RL-driven decision steps (BID, KING, TALON, CARD, ANNOUNCE) — which the existing `decision_types` routing already supports.
2. **Klop equalisation.** Klop scores each player individually; the shadow subtraction must be done per-player, not team-level. The `ShadowScoreRewardAdapter` handles this naturally (score is indexed by player), but it's worth an explicit test.
3. **Centaur variance.** When a seat is `centaur`, its policy is partly deterministic (PIMC worlds + α-μ depth). The PIMC world sampling uses the Rust RNG — so its variance is **not** cancelled by duplicate seeding unless we also seed the PIMC sampler. This must be fixed as part of Phase 1 or documented as a known limitation.

---

## 14. Summary

Duplicate RL is implemented as a **reward source plugin** on top of the existing PPO + FSP + Arena stack:

- **Three new ports** cleanly delimit the pairing policy, the reward model, and the seeded self-play capability.
- **Four new adapters** provide the default rotation pairing, shadow-difference reward, seeded self-play wrapper, and (Rust-side) seeded pod runner.
- **One new use case** orchestrates a duplicate iteration and produces the exact `ExperienceBundle` shape the existing orchestrator already consumes.
- **Minimal patches** to `ppo_batch_preparation.py`, `maintain_league_pool.py`, and the config schema — all behind a disabled-by-default flag.

The existing training, arena, and ELO-league PPO loop continue to run unchanged when the flag is off, and the `test_duplicate_disabled_is_noop` regression test enforces that this remains true.
