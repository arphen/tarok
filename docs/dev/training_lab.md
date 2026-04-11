# Training Lab — Architecture & Migration Plan

## 1. Vision

Extract all training, RL, and neural-network code from `backend/src/tarok/adapters/ai/`
into a standalone Python package called **`training-lab`** that lives at the repo root
alongside `backend/`, `frontend/`, and `engine-rs/`.

The backend installs `training-lab` as a local dependency and exposes its API
through FastAPI endpoints.  The training lab has **zero knowledge** of FastAPI,
WebSockets, or any web framework.  The backend is just a thin adapter.

Nothing in the current `backend/` code is touched during the build-out.
The old `adapters/ai/` keeps running until the new package is validated, at
which point the backend switches its imports and the old code is deleted.

---

## 2. Repo Layout (After Migration)

```
tarok/
├── backend/              # FastAPI service — thin adapter over game + training-lab
│   ├── pyproject.toml    # depends on training-lab (path dep)
│   └── src/tarok/
│       ├── entities/     # Card, GameState, Scoring (UNCHANGED)
│       ├── engine/       # Trick eval, legal moves (UNCHANGED)
│       ├── ports/        # PlayerPort, ObserverPort (UNCHANGED)
│       ├── use_cases/    # GameLoop, Deal, Bid, … (UNCHANGED)
│       └── adapters/
│           ├── api/      # FastAPI routes, WebSocket, schemas (UNCHANGED)
│           └── ai/       # OLD code — deleted after migration verified
│
├── training-lab/         # NEW — standalone training package
│   ├── pyproject.toml
│   └── src/training_lab/
│       ├── entities/     # Domain objects (Network, Experience, Checkpoint, …)
│       ├── ports/        # Abstract interfaces (GameSimulator, ComputeBackend, …)
│       ├── use_cases/    # Training orchestration (PPO, Imitation, Breeding, …)
│       ├── adapters/     # Concrete implementations
│       │   ├── compute/  # GPU/CPU/MPS backends
│       │   ├── engine/   # Rust engine adapter (tarok_engine FFI)
│       │   └── storage/  # Checkpoint I/O, HoF persistence
│       └── infra/        # Shared utilities (logging, config, metrics)
│
├── engine-rs/            # Rust engine + PyO3 bindings (UNCHANGED)
├── frontend/             # React + Vite (UNCHANGED)
├── docs/
│   └── dev/
│       └── training_lab.md   # ← this file
└── Makefile
```

---

## 3. Clean Architecture — Domain Model

### 3.1 Entities (innermost ring — no dependencies)

These are plain data objects with domain logic but zero I/O, no PyTorch,
no filesystem.  They may use `dataclasses`, `enum`, and pure functions.

| Entity | Responsibility | Source file(s) |
|--------|---------------|----------------|
| `Network` | Describes a neural network architecture (layer sizes, head configs). Wraps `torch.nn.Module` but the *entity* is the config, not the module. | `network.py` (config portion) |
| `Experience` | A single (s, a, r, log_prob, value, decision_type, legal_mask, game_id, step) tuple. Immutable. | `agent.py` → `Experience` dataclass |
| `ExperienceBatch` | A materialized tensor batch of N experiences, grouped by `DecisionType`. Ready for GPU. | NEW |
| `Checkpoint` | Model weights + metadata (hash, win_rate, persona, train_step, timestamp). | `hof_manager.py` metadata |
| `DecisionType` | Enum: BID, KING, TALON, CARD, ANNOUNCE. | `encoding.py` |
| `TrainingConfig` | All hyperparameters (lr, gamma, clip_ε, batch_size, …). Frozen dataclass. | `trainer.py` defaults |
| `SessionMetrics` | Per-session stats (reward, win_rate, loss, games/sec). | `trainer.py` `TrainingMetrics` |
| `BehavioralProfile` | Trait vector (bid_aggression, solo_propensity, …). | `behavioral_profile.py` |
| `OpponentSpec` | Describes an opponent: type (self-play / FSP / heuristic / HoF) + config. | `opponent_pool.py` |

### 3.2 Ports (interfaces — depend only on entities)

Abstract protocols that the use cases program against.

| Port | Methods | Purpose |
|------|---------|---------|
| `GameSimulatorPort` | `play_batch(network, n_games, opponents, explore_rate) → list[GameResult]` | Runs N games and returns experiences.  Adapter = Rust batch runner or Python game loop. |
| `ComputeBackendPort` | `forward_batch(network, states, dtypes, oracle?) → (logits, values)`, `to_device(tensor)`, `device` | GPU/CPU/MPS inference abstraction. |
| `CheckpointStorePort` | `save(checkpoint)`, `load(id) → Checkpoint`, `list() → list[Checkpoint]`, `delete(id)` | Filesystem or cloud persistence. |
| `HoFPort` | `save(checkpoint, pinned?)`, `list()`, `load(hash) → Checkpoint`, `pin/unpin/remove(hash)` | Hall of Fame management. |
| `MetricsSinkPort` | `record(session_metrics)`, `flush()` | Push metrics to TensorBoard, a queue, or the API layer. |
| `ProgressPort` | `report(phase, progress_pct, metrics)` | Callback so the web layer can poll training status. |

### 3.3 Use Cases (application layer — depend on entities + ports)

Each use case is a single class with a `run()` or `__call__()` method.
Use cases do not know about FastAPI, Rust FFI details, or filesystem paths.

| Use Case | Responsibility |
|----------|---------------|
| `RunPPOTraining` | The main self-play + PPO loop.  **NEW async producer-consumer architecture** (see §4). |
| `RunImitationPretraining` | Phase 1: Generate expert data via `GameSimulatorPort`, train supervised. |
| `RunWarmup` | Value-network bootstrap from random play data. |
| `RunPipeline` | Orchestrates Phase 1 → Phase 2 → Phase 3 in sequence. |
| `RunBreeding` | DEAP-based behavioral trait evolution. |
| `RunEvolution` | DEAP-based hyperparameter search. |
| `EvaluateModel` | Run N games vs reference opponents, compute win_rate/placement. |
| `ExportModel` | Export checkpoint to ONNX (for Rust-native inference, Phase 3 optimization). |
| `ManageHoF` | Save / pin / unpin / prune Hall of Fame entries. |

### 3.4 Adapters (outermost ring — implement ports)

| Adapter | Implements | Notes |
|---------|-----------|-------|
| `RustBatchGameRunner` | `GameSimulatorPort` | Wraps `BatchGameRunner` logic: M concurrent Rust games + batched NN inference. |
| `PythonGameRunner` | `GameSimulatorPort` | Fallback: sequential `GameLoop` from `backend/use_cases`. |
| `GpuBackend` | `ComputeBackendPort` | AMP, torch.compile, pinned memory, non-blocking transfers. |
| `CpuBackend` | `ComputeBackendPort` | Plain CPU inference. |
| `MpsBackend` | `ComputeBackendPort` | Apple Silicon GPU. |
| `FileCheckpointStore` | `CheckpointStorePort` | Save/load `.pt` files on disk. |
| `FileHoFManager` | `HoFPort` | JSON metadata + `.pt` weights in `hall_of_fame/` directory. |
| `QueueMetricsSink` | `MetricsSinkPort` | Pushes metrics into `asyncio.Queue` readable by the API layer. |
| `CallbackProgress` | `ProgressPort` | Invokes a callback / sets shared state for API polling. |

---

## 4. Async Producer-Consumer PPO (Key Performance Improvement)

The biggest architectural change.  **Decouple self-play from gradient updates.**

### 4.1 Current Problem

```
Session N:  [--- play 20 games (GPU: inference only) ---][--- PPO update (GPU: training) ---]
Session N+1:  [--- play 20 games ---][--- PPO update ---]
             ↑                       ↑
             GPU mostly idle         CPU idle
```

- GPU alternates between light inference load and heavy training load.
- CPU/Rust engine sits idle during PPO updates.
- Only ~1,200 experiences per PPO update (20 games × ~60 decisions).

### 4.2 New Architecture

```
Producer thread (Rust games + inference):
  ┌──────────────────────────────────────────────────────────────┐
  │  BatchGameRunner runs continuously (128 games in-flight)     │
  │  Uses FROZEN network copy for inference                      │
  │  Pushes completed GameResults into ExperienceBuffer          │
  │  Refreshes network copy every K consumer updates             │
  └──────────────────────────────────────────────────────────┬───┘
                                                             │
                                                             ▼
                                                 ┌──────────────────┐
                                                 │ ExperienceBuffer  │
                                                 │ (thread-safe)     │
                                                 │ capacity: ~50K    │
                                                 │ staleness tagging │
                                                 └────────┬─────────┘
                                                          │
  ┌───────────────────────────────────────────────────────┘
  ▼
Consumer thread (PPO training on GPU):
  ┌──────────────────────────────────────────────────────────────┐
  │  Waits until buffer has ≥ min_experiences (e.g. 5,000)      │
  │  Pulls batch, computes GAE per-game                          │
  │  Groups by DecisionType                                      │
  │  PPO update: 6-8 epochs × 256-batch mini-batches             │
  │  Updates canonical network weights                           │
  │  Signals producer to refresh its frozen copy                 │
  └──────────────────────────────────────────────────────────────┘
```

### 4.3 Why This Is Safe for PPO

PPO already uses importance sampling via the policy ratio `π_new(a|s) / π_old(a|s)`
and clips it to `[1-ε, 1+ε]`.  This means:

- Experiences from a **slightly stale** policy (1–3 updates old) are still valid.
- The clip ratio prevents catastrophic divergence.
- We add a **staleness tag** (policy version at collection time) and discard
  experiences older than `max_staleness` updates (default: 3).
- If needed later, V-trace importance weighting (IMPALA-style) can be added,
  but simple staleness cutoff is sufficient to start.

### 4.4 ExperienceBuffer Entity

```python
@dataclass
class TaggedExperience:
    experience: Experience
    policy_version: int          # which network version generated this
    collection_time: float       # wall-clock timestamp

class ExperienceBuffer:
    """Thread-safe ring buffer for completed game experiences."""
    capacity: int = 50_000
    max_staleness: int = 3       # discard if policy_version < current - max_staleness

    def push_game(self, experiences: list[Experience], policy_version: int) -> None
    def pull_batch(self, min_size: int) -> list[TaggedExperience] | None
    def discard_stale(self, current_version: int) -> int
    def __len__(self) -> int
```

### 4.5 Expected Gains

| Metric | Before | After (est.) |
|--------|--------|-------------|
| GPU utilization during self-play | ~40% (inference only) | ~40% inference + training overlapped |
| GPU utilization during PPO | ~80% | ~80% (same, but runs concurrently) |
| **Effective GPU utilization** | **~50%** (alternating) | **~75-85%** (overlapped) |
| Experiences per PPO update | ~1,200 | ~5,000–10,000 |
| Gradient steps per update | ~60 | ~500–1,000 |
| Wall-clock time per 1K sessions | Baseline | ~40-60% reduction (est.) |

---

## 5. New Package Structure (Detail)

```
training-lab/
├── pyproject.toml
├── README.md
└── src/
    └── training_lab/
        ├── __init__.py               # Public API surface
        │
        ├── entities/
        │   ├── __init__.py
        │   ├── network.py            # TarokNet (nn.Module) + NetworkConfig
        │   ├── experience.py         # Experience, ExperienceBatch, TaggedExperience
        │   ├── experience_buffer.py  # ExperienceBuffer (thread-safe ring buffer)
        │   ├── checkpoint.py         # Checkpoint metadata dataclass
        │   ├── config.py             # TrainingConfig, PipelineConfig
        │   ├── metrics.py            # SessionMetrics, TrainingProgress
        │   ├── encoding.py           # DecisionType, state encoding, action maps
        │   ├── decision.py           # BID/KING/TALON/CARD/ANNOUNCE action sizes & maps
        │   ├── behavioral_profile.py # BehavioralProfile + gene bounds
        │   └── opponent.py           # OpponentSpec, OpponentStats
        │
        ├── ports/
        │   ├── __init__.py
        │   ├── game_simulator.py     # GameSimulatorPort protocol
        │   ├── compute_backend.py    # ComputeBackendPort protocol
        │   ├── checkpoint_store.py   # CheckpointStorePort protocol
        │   ├── hof.py                # HoFPort protocol
        │   ├── metrics_sink.py       # MetricsSinkPort protocol
        │   └── progress.py           # ProgressPort protocol
        │
        ├── use_cases/
        │   ├── __init__.py
        │   ├── ppo_training.py       # RunPPOTraining (async producer-consumer)
        │   ├── imitation.py          # RunImitationPretraining
        │   ├── warmup.py             # RunWarmup
        │   ├── pipeline.py           # RunPipeline (Phase 1→2→3)
        │   ├── breeding.py           # RunBreeding (DEAP behavioral)
        │   ├── evolution.py          # RunEvolution (DEAP hyperparams)
        │   ├── evaluate.py           # EvaluateModel
        │   ├── export_model.py       # ExportModel (ONNX)
        │   └── manage_hof.py         # ManageHoF
        │
        ├── adapters/
        │   ├── __init__.py
        │   ├── compute/
        │   │   ├── __init__.py
        │   │   ├── gpu_backend.py    # GpuBackend (AMP, torch.compile)
        │   │   ├── cpu_backend.py    # CpuBackend
        │   │   ├── mps_backend.py    # MpsBackend (Apple Silicon)
        │   │   └── factory.py        # create_backend() auto-detect
        │   ├── engine/
        │   │   ├── __init__.py
        │   │   ├── rust_batch_runner.py   # RustBatchGameRunner
        │   │   ├── python_game_runner.py  # PythonGameRunner (fallback)
        │   │   └── rust_game_loop.py      # Single-game Rust loop
        │   └── storage/
        │       ├── __init__.py
        │       ├── file_checkpoint_store.py
        │       └── file_hof_manager.py
        │
        └── infra/
            ├── __init__.py
            ├── logging.py            # Structured logging setup
            └── device.py             # Device detection helpers
```

---

## 6. Dependency Graph

```
training-lab
├── torch               # Neural network, tensors
├── numpy               # Array interchange with Rust
├── deap                # Evolutionary algorithms (breeding, evo)
└── tarok_engine        # Rust engine (optional — graceful fallback)

backend
├── training-lab        # Local path dependency
├── fastapi, uvicorn    # Web framework
├── tarok entities      # Card, GameState, Scoring (game domain)
└── tarok_engine        # Rust engine (for game play)
```

**Critical boundary:** `training-lab` does NOT import from `tarok.entities`,
`tarok.use_cases`, `tarok.ports`, or `tarok.adapters`.  It receives game
simulation results through its `GameSimulatorPort`.  The backend's adapter
layer bridges the two packages.

However, `encoding.py` currently depends on `tarok.entities.card` and
`tarok.entities.game_state` for `Card`, `Contract`, `Phase`, etc.
This dependency is resolved by:

1. The encoding module operates on **raw tensors and indices** internally.
2. A thin **bridge adapter** in the backend maps `tarok.entities` objects
   to the raw indices that `training_lab.entities.encoding` expects.
3. The Rust engine already provides encoded states as numpy arrays,
   bypassing the Python entity layer entirely for the batch path.

For the Python game runner (fallback), the backend provides a
`PythonGameSimulator` adapter that wraps `GameLoop` and calls encoding
functions that know about `tarok.entities`.

---

## 7. Backend Integration

### 7.1 pyproject.toml Change

```toml
[project]
dependencies = [
    # ... existing ...
    "training-lab",       # local path dependency
]

[tool.uv.sources]
training-lab = { path = "../training-lab", editable = true }
```

### 7.2 API Adapter (backend side)

The backend's `server.py` training endpoints delegate to `training_lab`:

```python
# backend/src/tarok/adapters/api/training_adapter.py  (NEW — thin bridge)

from training_lab.use_cases.ppo_training import RunPPOTraining
from training_lab.use_cases.pipeline import RunPipeline
from training_lab.adapters.compute.factory import create_backend
from training_lab.adapters.engine.rust_batch_runner import RustBatchGameRunner
from training_lab.adapters.storage.file_checkpoint_store import FileCheckpointStore
from training_lab.entities.config import TrainingConfig

class TrainingAdapter:
    """Bridges FastAPI ↔ training-lab use cases."""

    def __init__(self, checkpoint_dir: Path):
        self.compute = create_backend()
        self.simulator = RustBatchGameRunner(self.compute)
        self.store = FileCheckpointStore(checkpoint_dir)
        self._task: asyncio.Task | None = None

    async def start_training(self, request: TrainingRequest) -> str:
        config = TrainingConfig.from_api_request(request)
        use_case = RunPPOTraining(
            simulator=self.simulator,
            compute=self.compute,
            store=self.store,
            config=config,
        )
        self._task = asyncio.create_task(use_case.run())
        return use_case.run_id

    def get_metrics(self) -> dict: ...
    async def stop(self) -> None: ...
```

---

## 8. Migration Strategy (Zero Disruption)

### Phase A: Scaffold & Copy (no behavior change)

1. Create `training-lab/` directory with `pyproject.toml` and package structure.
2. Copy existing code from `backend/src/tarok/adapters/ai/` into the correct
   clean-architecture locations in `training-lab/src/training_lab/`.
3. Refactor imports within the new package to use `training_lab.*` paths.
4. Add basic tests that the new package imports cleanly.
5. Add `training-lab` as a path dependency in `backend/pyproject.toml`.

### Phase B: Build Producer-Consumer PPO

6. Implement `ExperienceBuffer` entity (thread-safe ring buffer).
7. Implement `RunPPOTraining` use case with async producer-consumer loop.
8. Wire up `RustBatchGameRunner` adapter with continuous mode.
9. Add integration tests: verify training converges on a small run.
10. Benchmark: compare games/sec and GPU utilization vs old trainer.

### Phase C: Backend Integration

11. Create `TrainingAdapter` in the backend that delegates to `training_lab`.
12. Add new API endpoints (or modify existing ones) to use the adapter.
13. Run side-by-side: old endpoints still work, new endpoints available at
    `/api/v2/training/*` for testing.
14. Verify frontend works with new endpoints (same schema, different import path).

### Phase D: Validate & Cut Over

15. Run full training pipeline (imitation → PPO → FSP) with new package.
16. Compare win-rate learning curves between old and new implementations.
17. Remove old `backend/src/tarok/adapters/ai/` code.
18. Update `Makefile`, CLI commands, and documentation.

### Phase E: ONNX Export (Optional, Future)

19. Add `ExportModel` use case with `torch.onnx.export`.
20. Add `ort` crate to `engine-rs`, implement Rust-native batch self-play.
21. Expose `generate_self_play_data(onnx_path, num_games)` via PyO3.
22. Update `RustBatchGameRunner` to use ONNX path when available.

---

## 9. Files to Copy & Their New Locations

| Old Location (`backend/src/tarok/adapters/ai/`) | New Location (`training-lab/src/training_lab/`) | Notes |
|---|---|---|
| `network.py` | `entities/network.py` | Core NN module |
| `encoding.py` | `entities/encoding.py` | Decouple from `tarok.entities` |
| `agent.py` → `Experience` | `entities/experience.py` | Extract dataclass only |
| `behavioral_profile.py` | `entities/behavioral_profile.py` | Pure data |
| `opponent_pool.py` | `entities/opponent.py` + `adapters/engine/opponent_pool.py` | Split data vs logic |
| `compute/backend.py` | `ports/compute_backend.py` | Abstract interface |
| `compute/gpu_backend.py` | `adapters/compute/gpu_backend.py` | GPU adapter |
| `compute/cpu_backend.py` | `adapters/compute/cpu_backend.py` | CPU adapter |
| `compute/factory.py` | `adapters/compute/factory.py` | Auto-detection |
| `batch_game_runner.py` | `adapters/engine/rust_batch_runner.py` | Rust FFI adapter |
| `rust_game_loop.py` | `adapters/engine/rust_game_loop.py` | Single-game Rust loop |
| `trainer.py` | `use_cases/ppo_training.py` | Rewrite with producer-consumer |
| `imitation.py` | `use_cases/imitation.py` | Supervised learning |
| `warmup.py` | `use_cases/warmup.py` | Value pre-training |
| `breeding.py` | `use_cases/breeding.py` | DEAP behavioral |
| `evo_optimizer.py` | `use_cases/evolution.py` | DEAP hyperparams |
| `training_lab.py` | `use_cases/pipeline.py` + `infra/lab_state.py` | Split orchestration vs state |
| `hof_manager.py` | `adapters/storage/file_hof_manager.py` | Filesystem HoF |
| `network_bank.py` | `adapters/storage/network_bank.py` | FSP snapshot storage |
| `island_worker.py` | `adapters/engine/island_worker.py` | PBT worker |
| `tournament_results.py` | `adapters/storage/tournament_results.py` | Results I/O |
| `stockskis_v5.py` | `adapters/engine/stockskis_adapter.py` | Heuristic bot bridge |
| `bot_registry.py` | `adapters/engine/bot_registry.py` | Bot catalog |
| `lookahead_agent.py` | `adapters/engine/lookahead_agent.py` | MCTS/lookahead |
| `random_agent.py` | stays in backend | Used by game play, not training |
| `agent.py` (RLAgent class) | stays in backend (uses `training_lab.entities.network`) | Implements `PlayerPort` for game play |

---

## 10. pyproject.toml for training-lab

```toml
[project]
name = "training-lab"
version = "0.1.0"
description = "RL training lab for Tarok neural network agents"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.4",
    "numpy>=2.0",
    "deap>=1.4",
    "shortuuid>=1.0.13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/training_lab"]

[tool.ruff]
line-length = 100
```

---

## 11. Key Design Decisions

1. **`training-lab` does NOT depend on `tarok` (the backend package).**
   The game domain (Card, GameState, etc.) stays in `backend`.
   The training lab receives raw tensors / numpy arrays through its ports.

2. **`RLAgent` stays in the backend.**
   `RLAgent` implements `PlayerPort` (a game-domain interface) and is needed
   for human-vs-AI gameplay.  It imports `TarokNet` from `training_lab`.

3. **Encoding bridge.**
   `training_lab.entities.encoding` works with raw indices and tensor dims.
   A bridge function in the backend converts `tarok.entities.GameState` →
   raw feature vector.  The Rust engine bypasses this entirely (returns
   pre-encoded numpy arrays).

4. **The ExperienceBuffer is an entity, not an adapter.**
   It's a thread-safe in-memory data structure with domain semantics
   (staleness, policy versioning).  No I/O.

5. **Each use case gets its own file.**
   No god-class trainers.  `RunPPOTraining` is ~200 lines.
   `RunImitationPretraining` is ~100 lines.  Complexity stays manageable.

6. **`agent.py` (the full RLAgent with behavioral biases, action sampling,
   experience recording) stays in the backend** since it implements the
   game-domain `PlayerPort` interface.  But it imports `TarokNet` and
   `Experience` from `training_lab`.

---

## 12. Open Questions

- **Shared `tarok_engine` type definitions?**  Both `backend` and `training-lab`
  depend on `tarok_engine` (Rust).  The Rust engine returns raw numpy arrays,
  so there's no type clash — both packages just use numpy/torch tensors.

- **CLI entry point?**  Should `training-lab` have a `__main__.py` for
  standalone training runs (no FastAPI), or does that stay in `backend`?
  Recommendation: add a minimal CLI (`python -m training_lab train ...`)
  for headless training without the web server.

- **Test data fixtures?**  Training tests need sample game data.
  Use `tarok_engine.generate_expert_data()` in test fixtures, or ship
  small `.npz` fixture files in `training-lab/tests/fixtures/`.
