# Duplicate Reinforcement Learning (DRL) for Tarok

> Status: **Phases 1, 2 + Phase 3 (PPO + network)** complete. Actor-only mode is now reachable from config and `model_arch: v5` physically drops the value side of the network. Phase 4 (arena UI) still pending.
> Guiding principle: **strictly additive on top of the existing PPO + Fictitious Self-Play + Arena stack.** Nothing about current training, arena leaderboard, or ELO league changes unless the Duplicate feature flag is explicitly enabled.

## Implementation status (as of last commit)

Completed:

- Entities, ports, adapters, Rust seeded dealing — see prior commits.
- `CollectDuplicateExperiences` use case + orchestrator branching on `config.duplicate.enabled`.
- Composition root auto-wires `RotationPairingAdapter` + `SeededSelfPlayAdapter` + `ShadowScoreRewardAdapter` when duplicate is enabled.
- End-to-end smoke test of the full pipeline (pairing → seeded self-play → reward → `prepare_batched`).
- **Phase 3 (PPO side):** `_broadcast_terminal_advantage` helper + `actor_only` branch in `ppo_batch_preparation.py`. When `raw["actor_only"]=True`, GAE is replaced by the discounted terminal-advantage broadcast (§4.2); the `vad` matrix's `old_values` column is forced to zero so the downstream PPO loss is critic-free in this regime. `CollectDuplicateExperiences` propagates `duplicate_config.actor_only` onto the raw dict automatically. `PPOAdapter._ppo_update_batched` accepts an `actor_only` kwarg threaded through `prepare_batched` and zeroes the `value_loss` + `il_loss` terms when set.
- **Config preset:** `training-lab/configs/duplicate.yaml` exposes every `duplicate:` knob with sane first-run defaults; drive a full iteration via `make train-new CONFIG=duplicate`.
- **Actor-only network (`TarokNetV5`):** `model_arch: v5` instantiates a subclass of `TarokNetV4` with the `critic` head and oracle backbone physically deleted; `state_dict` is a strict subset of v4's, so v4 checkpoints load via `strict=False`. `forward` / `forward_batch` return a zero-valued placeholder for the `values` tensor to keep TorchScript export and the PPO data path shape-compatible. Plumbed through `TorchModelAdapter.create_new` / `export_for_inference` / `save_checkpoint`.
- **Spawn runner duplicate support:** `ConfigurableIterationRunner._adapter_factory_for_spawn` now takes the `TrainingConfig` and constructs `SeededSelfPlayAdapter` + `RotationPairingAdapter` + `ShadowScoreRewardAdapter` inside the worker process when `duplicate.enabled=True`. The previous `NotImplementedError` guard is removed; `iteration_runner_mode: spawn` now works with duplicate configs.
- **Shadow-source port:** `DuplicateShadowSourcePort` has three adapters behind `duplicate.shadow_source`: `previous_iteration` (default; reads the learner's TorchScript path, which still holds prior weights at the start of each iteration), `league_pool` (Gaussian-weighted random sample over `nn_checkpoint` entries near the learner's Elo), and `best_snapshot` (highest-Elo snapshot — "best ghost ever"). Wired through `IterationRunnerPort.run_iteration(pool=...)`, across both in-process and spawn runners; constructed from config in `training.container` via `create_shadow_source(...)`.
- Tests: full training-lab suite passes; `make lint-architecture` green.

Still pending:

- Rust-side actor-only buffer suppression (optimisation; current v5 path is already correct, just wastes the values slot).
- Phase 4: Arena duplicate CLI + UI panel.

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

In the conservative (Phase 2) integration this replaces only the per-trajectory reward fed into GAE while keeping the critic and Oracle intact. Section §2.7 describes the more aggressive consequence: that once you have a mathematically pure empirical advantage, the Critic network, the Oracle, GAE itself, and oracle distillation all become redundant and can be pruned. The two are separate config modes.

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

### 2.7 Actor-Only mode: amputating the Value side of the network

This section records an important architectural consequence of §2.2 that the initial plan underestimated.

#### Why the Critic existed in the first place

In standard PPO, the critic's sole purpose is to produce a low-variance baseline for $A$:

$$
A_{\text{vanilla}} = R_{\text{game}} - V_\theta(s)
$$

Without a good $V_\theta$, luck dominates the gradient. Being dealt the Tarok 21 inflates $R$ by +30; the network incorrectly learns that whatever it did was brilliant. The Oracle Critic was introduced specifically to give $V_\theta$ access to perfect-information opponent hands so it could distinguish skill-driven outcomes from card-luck outcomes.

#### Why Duplicate RL makes the Critic mathematically redundant

In Duplicate RL, the Shadow *is* the baseline:

$$
A_{\text{duplicate}} = R_{\text{learner}} - R_{\text{shadow}}
$$

Both learner and shadow receive the exact same deck and opponents. The Tarok-21 luck cancels:

$$
(\underbrace{30}_{\text{luck}} + \delta_\text{learner}) - (\underbrace{30}_{\text{luck}} + \delta_\text{shadow}) = \delta_\text{learner} - \delta_\text{shadow}
$$

The residual is purely the policy delta — the quantity PPO was always trying to isolate. This cancellation is **exact** (not statistical) because the deals are byte-identical. No neural network can improve on an exact cancellation.

#### What can be deleted

| Component | Role | Status under Duplicate RL |
|---|---|---|
| `critic` value head | Estimates $V(s)$ for baseline | **Redundant** — Shadow is the baseline |
| Oracle critic backbone | Estimates $V(s)$ from perfect info | **Redundant** — variance already cancelled |
| `ORACLE_STATE_SIZE` (+162 dims) | Input to oracle critic | **Deletable** — no oracle critic in actor-only |
| Oracle distillation loss | Aligns actor latent ↔ oracle critic | **Deletable** |
| GAE (`te.compute_gae`) | Bootstraps $V$ across steps | **Replaceable** — see below |

#### What replaces GAE

GAE with $\lambda < 1$ does temporal credit assignment within a trajectory by exponentially discounting the value-bootstrap across steps. Under duplicate RL with terminal-only rewards, the per-step advantage becomes:

$$
A_t = \gamma^{T-t} \cdot A_{\text{duplicate}}
$$

where $T$ is the terminal step and $\gamma = 0.99$. This is REINFORCE with a duplicate baseline: every step in the trajectory gets the discounted duplicate advantage, and no critic bootstrap is needed. The `te.compute_gae` call in `ppo_batch_preparation.py` is replaced by a simple broadcast:

```python
# actor-only duplicate path
duplicate_adv = precomputed_rewards   # terminal reward = A_duplicate
returns_np = broadcast_terminal_reward_to_trajectory(
    duplicate_adv, game_ids_np, players_np, gamma=config.gamma
)
advantages_np = returns_np          # no critic subtraction needed
```

The critic value loss term and the `vad` matrix column for `old_values` are dropped from the PPO update.

#### Network size impact

Current TarokNet v2 (oracle critic enabled):
- Input: 450-dim actor state + 612-dim oracle state
- Critic backbone: `Linear(612) → LN → ReLU → Linear(256) → LN → ReLU → ResidualBlock(256)`
- Oracle distillation loss adds a forward pass on the oracle backbone each step

After actor-only pruning:
- Input: 450-dim actor state only
- Critic backbone: **deleted**
- Oracle backbone: **deleted**
- Critic head `Linear(256) → ReLU → Linear(1)`: **deleted**
- Net effect: ~30–40% fewer parameters, significantly faster per-step forward pass on Apple Silicon MPS, no oracle-state buffer in Rust (saves memory during rollout).

#### The trade-off: within-game credit assignment

The critic, even a noisy one, provides *within-game* credit assignment — it can signal that the state after a brilliant Trick 4 discard is intrinsically more valuable than before. The duplicate baseline only operates at *game* granularity. Removing the critic entirely means all ~50 decisions in a game get the same advantage signal, and the network has to learn credit assignment implicitly from policy gradient alone.

This is acceptable because:
1. Tarok rewards are already **terminal** — there is no intermediate reward signal the critic was bootstrapping, only its own value estimate.
2. The within-game variance the critic was reducing is small compared to the between-game card-luck variance that duplicate RL eliminates.
3. A larger `pods_per_iteration` can compensate for the coarser within-game signal by providing more trajectory samples.

#### Configuration

```yaml
duplicate:
  enabled: true
  actor_only: true    # drops critic, oracle, GAE; broadcasts terminal duplicate advantage
```

When `actor_only: false` (default for Phase 2), the critic is retained and duplicate RL only changes the reward source — a conservative starting point. When `actor_only: true` (Phase 3 opt-in), the network is pruned and GAE replaced. The two config flags are independent so research can compare the two regimes on equal footing.

**Important:** `actor_only: true` produces a checkpoint without a critic head. The existing checkpoint loader (`load_state_dict` with `strict=False`) already tolerates missing modules; no migration code is needed.

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

[training-lab/training/adapters/ppo/ppo_batch_preparation.py](training-lab/training/adapters/ppo/ppo_batch_preparation.py) is touched in two distinct ways depending on which mode is active.

### 4.1 Conservative mode (`actor_only: false`) — reward source only

The current reward computation:

```python
rewards_np = scores_np[gids, players_np].astype(np.float32) / 100.0
rewards_np = rewards_np + shaped_bonus_by_game[gids, players_np]
```

becomes:

```python
precomputed = raw.get("precomputed_rewards")
if precomputed is not None:
    rewards_np = np.asarray(precomputed, dtype=np.float32)
else:
    rewards_np = scores_np[gids, players_np].astype(np.float32) / 100.0
    rewards_np = rewards_np + shaped_bonus_by_game[gids, players_np]
```

Everything downstream — trajectory-key sort, terminal masking, `te.compute_gae`, global advantage normalisation, critic value loss, oracle distillation, mini-batching — is untouched. This is the minimal viable change: duplicate RL plugs in purely as a **reward source**.

**Shaped bonuses under duplicate RL.** The existing `_compute_special_shaped_bonus_by_game` adds small trick-point shaping. Under pure duplicate RL these bonuses also cancel (both tables experience the same deck), so they are **disabled by default** when `duplicate.enabled`. A config flag `duplicate.apply_shaped_bonuses` can override.

### 4.2 Actor-only mode (`actor_only: true`) — full Value side removed

When `actor_only: true`, `ppo_batch_preparation.py` skips GAE entirely and replaces it with a simple terminal-reward broadcast:

```python
# actor-only path — no critic, no GAE
assert raw.get("precomputed_rewards") is not None, "actor_only requires precomputed_rewards"
terminal_adv_np = np.asarray(raw["precomputed_rewards"], dtype=np.float32)

# Broadcast: every step in a (game, player) trajectory gets the discounted terminal advantage
advantages_np = _broadcast_terminal_advantage(
    terminal_adv_np, game_ids_np, players_np, gamma=gamma
)

# vad matrix shrinks: no old_values column (critic gone)
vad_np = np.stack([advantages_np, advantages_np], axis=1)  # col 0=adv, col 1=returns (same)
```

The PPO loss in `torch_ppo.py` correspondingly drops the `value_loss` term and the oracle distillation forward pass. The `values` tensor returned by the Rust NN forward pass is also suppressed (the actor-only TarokNet simply doesn't produce it).

The `_broadcast_terminal_advantage` helper is a new function in `ppo_batch_preparation.py`:

```python
def _broadcast_terminal_advantage(
    terminal_adv: np.ndarray,   # shape (n_steps,) — nonzero only at terminal steps
    game_ids: np.ndarray,
    players: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Assign discounted terminal advantage to every step in the trajectory.

    For a trajectory of T steps ending with advantage A, step t gets γ^(T−t) × A.
    This is REINFORCE with a duplicate baseline — no critic bootstrap required.
    """
```

This function is pure numpy, fully testable without Rust or PyTorch.

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
    "states": ..., "actions": ..., "log_probs": ...,
    "values": ...,                # zeros / absent when actor_only=True
    "decision_types": ..., "game_ids": ..., "players": ..., "scores": ...,
    "legal_masks": ..., "game_modes": ...,
    "oracle_states": ...,         # absent when actor_only=True (no oracle backbone)

    # duplicate-specific
    "pod_ids": np.ndarray,        # shape (n_active_steps,)
    "shadow_scores": np.ndarray,  # shape (n_pods, 4) — one per active game in the pod
    "learner_positions": np.ndarray,  # shape (n_pods, 4)
}
```

When `actor_only=True`, the Rust actor-only TarokNet does not run a critic forward pass and does not allocate oracle-state buffers, so `values` is a zero tensor and `oracle_states` is absent. `ppo_batch_preparation.py` detects this and routes to the `_broadcast_terminal_advantage` path (§4.2).

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
  actor_only: false                # drop critic/oracle/GAE (see §2.7); requires enabled: true
  pairing: rotation_8game          # "rotation_8game" | "rotation_4game" | "single_seat_2game"
  pods_per_iteration: 400          # produces 4 × 400 = 1600 learner trajectories/iter
  shadow_source: previous_iteration   # "previous_iteration" | "league_pool_role"
  apply_shaped_bonuses: false      # under duplicate, shaping is redundant
  reward_model: shadow_score_diff  # "shadow_score_diff" | "imps" | "ranking"
  rng_seed: 0                      # deterministic pod schedule
```

Config resolution lives in [training-lab/training/use_cases/resolve_config.py](training-lab/training/use_cases/resolve_config.py) (existing). A dataclass `DuplicateConfig` is added to `TrainingConfig`; absent/disabled means the existing path runs unchanged.

**Constraint:** `actor_only: true` requires `enabled: true`. `resolve_config.py` raises a `ConfigError` if `actor_only` is set without `enabled`.

---

## 10. Testing strategy

| Layer | Test | Location |
|---|---|---|
| Rust | Same `deck_seed` → identical deal across two independent runner invocations. | `engine-rs/tests/duplicate_seeded_deal.rs` (new) |
| Rust | 8-game pod with learner-NN scores recoverable equally from any seat rotation. | same file |
| Port | `DuplicatePairingPort.build_pods` returns 8 seatings with invariants: each seat hosts learner exactly once in active, and opponent identity set is constant across pod. | `backend/tests/test_duplicate_pairing.py` |
| Port | `DuplicateRewardPort.compute_rewards` on synthetic `active_raw` + `shadow_scores` returns exactly `(learner_score − matched_shadow_score) / 100`. | `backend/tests/test_duplicate_reward.py` |
| Batch prep | `_broadcast_terminal_advantage` with a 2-game trajectory of length 10 returns exactly `γ^(T-t) × A` at each step. | `training-lab/tests/test_ppo_batch_preparation.py` (extend existing) |
| Batch prep | `actor_only=true` path produces no `old_values` column in `vad`, no `value_loss` in metrics. | same file |
| Use case | `CollectDuplicateExperiences.execute` with fakes for all three ports returns a well-shaped `ExperienceBundle` and does not import `numpy`/`torch`/`tarok_engine` at module top. | `backend/tests/test_collect_duplicate_experiences.py` |
| Integration | End-to-end iteration with `duplicate.enabled=true, actor_only=false` on 4 pods runs, updates weights, checkpoint survives reload. | `training-lab/tests/test_duplicate_iteration.py` |
| Integration | Same with `actor_only=true` — checkpoint has no `critic` or `oracle_critic_backbone` keys. | same file, second parametrized case |
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

### Phase 2 — Conservative PPO wiring (`actor_only: false`)
5. `CollectDuplicateExperiences` use case.
6. Reward-source patch to `ppo_batch_preparation.py` (the `precomputed_rewards` fallback, §4.1).
7. Config `DuplicateConfig` + YAML plumbing including `actor_only` flag.
8. `TrainModelOrchestrator` branching on `config.duplicate.enabled`.
9. The disabled-is-noop regression test.
10. Full integration test: 4-pod iteration produces non-zero advantages, updated weights, checkpoint survives reload.

**Exit criterion:** 100-pod smoke training run completes without crash, `duplicate_advantage_std` decreases over 5 iterations.

### Phase 3 — Actor-Only pruning (`actor_only: true`)
11. `_broadcast_terminal_advantage` helper + actor-only branch in `ppo_batch_preparation.py` (§4.2).
12. `TarokNet` actor-only variant: `critic`, `oracle_critic_backbone`, oracle distillation loss deleted when `actor_only=True` is passed to constructor. Existing `load_state_dict(strict=False)` already handles missing keys.
13. Rust: suppress `values` output and oracle-state buffer allocation when actor-only model is loaded.
14. Integration test: `actor_only=true` checkpoint has no critic keys; loads cleanly; outplaces `bot_v5` after 200-pod training.
15. Ablation test: compare `actor_only=false` vs `actor_only=true` Elo after 1,000 pods on identical seeds.

**Exit criterion:** actor-only forward pass is measurably faster on MPS (target: ≥20% throughput gain).

### Phase 4 — Arena & UI
16. `arena duplicate` CLI subcommand.
17. Arena page: new "Duplicate Match" card.
18. TrainingDashboard: "Duplicate RL" collapsible panel with advantage histogram and `actor_only` indicator.
19. Frontend Playwright test covering the new UI affordances.

### Phase 5 — Research knobs
20. `IMPsRewardAdapter` + `RankingRewardAdapter` alternatives.
21. League-pool shadow role (shadow = learner from k iterations ago).
22. Hyperparameter sweep: pod size (2/4/8), `pods_per_iteration`, shadow staleness, `actor_only` on/off.

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
4. **Within-game credit assignment in actor-only mode.** Broadcasting the terminal duplicate advantage to all ~50 steps with $\gamma^{T-t}$ decay is a rough proxy for credit assignment. A learned lightweight step-value head (much smaller than today's oracle critic — e.g., 2-layer MLP on 64 dims) could provide finer within-game credit without reintroducing the full oracle machinery. Worth experimenting with after Phase 3 establishes the actor-only baseline.
5. **Entropy scheduling.** Today entropy coefficient is fixed. In actor-only mode with a pure REINFORCE gradient, entropy regularisation becomes more important to prevent premature convergence. An entropy schedule (high early, decay over iterations) may be needed and should be part of the Phase 5 sweep.

---

## 14. Summary

Duplicate RL is implemented in two tiers, both strictly additive:

**Tier 1 — Conservative (`enabled: true, actor_only: false`):**
- **Three new ports** cleanly delimit the pairing policy, the reward model, and the seeded self-play capability.
- **Four new adapters** provide the default rotation pairing, shadow-difference reward, seeded self-play wrapper, and (Rust-side) seeded pod runner.
- **One new use case** orchestrates a duplicate iteration and produces the exact `ExperienceBundle` shape the existing orchestrator already consumes.
- Critic, Oracle, and GAE are retained — duplicate RL only changes the *reward source* fed into GAE.

**Tier 2 — Actor-Only (`actor_only: true`):**
- The duplicate Shadow baseline provides an exact, mathematically perfect variance cancellation that no neural network baseline can improve on.
- The Critic head, Oracle critic backbone, Oracle state inputs (+162 dims), oracle distillation loss, and GAE are all **deleted**.
- GAE is replaced by `_broadcast_terminal_advantage`: a pure-numpy REINFORCE broadcast of the terminal duplicate advantage across the trajectory.
- Network shrinks by ~30–40% of parameters; forward passes are faster; Rust rollout buffers shrink (no oracle states).
- Existing `load_state_dict(strict=False)` already handles missing critic keys — no migration code needed.

**In both tiers:** minimal patches to `ppo_batch_preparation.py`, `maintain_league_pool.py`, and the config schema — all behind a disabled-by-default flag. The existing training, arena, and ELO-league PPO loop continue to run unchanged when `duplicate.enabled=false`, enforced by `test_duplicate_disabled_is_noop`.
