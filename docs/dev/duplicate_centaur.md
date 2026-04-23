# Duplicate-Centaur Training

Status: Phase A (Python plumbing + YAML config) landed. Phase B (engine-side
deterministic PIMC seeding) remains a follow-up.

## Motivation

Current duplicate RL (`configs/duplicate.yaml`) pits a pure-NN learner
against a pure-NN shadow. But we already have a Centaur player (NN +
PIMC/alpha-mu endgame solver) that plays the last ~4 tricks optimally. If
we make the learner play as Centaur in each pod, two things happen:

1. Early-NN decisions (bid / king / talon / tricks 0–7) get gradient signal
   backed by a solver-optimal endgame, not a noisy self-played endgame.
   The "what would a perfect player do with these cards?" credit
   assignment is much cleaner.
2. The learner converges against a more realistic *deployed* policy —
   Centaur is what we'd actually ship, so training it directly is
   preferable to training pure NN and then bolting a solver on.

Duplicate invariance is preserved: both active and shadow tables use
Centaur with the same solver knobs, so the PIMC noise in the endgame
appears on both sides and (approximately) cancels in `R_active - R_shadow`.

## Phase A — wired ✅

- `DuplicateConfig.learner_seat_token: "nn" | "centaur"` (default `"nn"`).
- `ResolveConfig` parses both `duplicate.learner_seat_token` and the
  previously-dropped top-level `centaur_*` knobs
  (`centaur_handoff_trick`, `centaur_pimc_worlds`, `centaur_endgame_solver`,
  `centaur_alpha_mu_depth`).
- `CollectDuplicateExperiences.execute(...)` now takes the four
  `centaur_*` kwargs and forwards them into
  `SelfPlayPort.run_seeded_pods(...)`.
- `RunIteration` forwards `config.centaur_*` into
  `collect_duplicate_experiences.execute(...)` on the duplicate branch.
- `_render_seat_config` in the seeded self-play adapter no longer rewrites
  the learner seat to `"nn"`; it preserves whatever learner token the
  pairing put there (which comes from `duplicate.learner_seat_token`),
  so the Rust engine builds a `CentaurBot` from `model_path` when
  `learner_seat_token: centaur`.
- New config: `configs/duplicate-centaur.yaml`.

### How to train

```bash
make train-new CONFIG=duplicate-centaur
```

Expect per-iteration wall time ~2–4× slower than `duplicate.yaml` because
PIMC runs in the last 4 tricks of every pod game. `pods_per_iteration` in
the new config is therefore halved (100 vs 200) to keep each iteration in
the same wall-time ballpark as the pure-NN version.

## Phase B — deterministic PIMC seeding (follow-up)

**Problem.** Even though `deck_seeds` align the deals between active and
shadow, the endgame solver uses `rand::rng().random()` to seed its
per-world PRNGs (see `engine-rs/src/pimc.rs::pimc_choose_card` and
`engine-rs/src/alpha_mu.rs::alpha_mu_choose_card`). That introduces a
source of variance that is independent of the deal — it doesn't cancel
under duplicate. `docs/double_rl.md` §6.3 calls this out as a known
limitation of the current implementation.

**Fix.**

1. Add seeded entry points in the Rust engine:
   ```rust
   pub fn pimc_choose_card_seeded(
       gs: &GameState, viewer: u8, num_worlds: u32, base_seed: u64,
   ) -> Card;

   pub fn alpha_mu_choose_card_seeded(
       gs: &GameState, viewer: u8, num_worlds: u32, max_depth: usize, base_seed: u64,
   ) -> Card;
   ```
   The existing `pimc_choose_card` / `alpha_mu_choose_card` become thin
   wrappers that generate `rand::rng().random()` and delegate.

2. Add `EndgameSeedingMode::Deterministic` (or a thread-local toggle
   reached via `pimc::set_deterministic(true)` similar to
   `lapajne::set_mc_sims`) and a stateless hash that derives `base_seed`
   from a canonical fingerprint of the decision context:
   ```
   base_seed = hash(
       deck_seed,                // shared by active+shadow
       viewer,
       gs.tricks_played(),
       current_trick canonical bytes,
       completed_tricks canonical bytes,
       contract, declarer, partner
   )
   ```
   When active and shadow reach *the same* game state, they sample the
   same worlds → the PIMC noise cancels exactly. When they reach
   different states (because learner and shadow NN diverged earlier),
   the seeds differ, which is the correct "don't synthesize spurious
   correlation" behavior.

3. Plumb a new `centaur_deterministic_seeding: Option<bool>` kwarg
   through `py_bindings.run_self_play` → `SelfPlayPort.run_seeded_pods`
   → `SeededSelfPlayAdapter` → `CollectDuplicateExperiences`, defaulting
   to `None` (current behavior). The `duplicate-centaur.yaml` config
   should then opt in once the engine side lands.

4. Rebuild the engine:
   ```bash
   cd engine-rs && uv run maturin develop --release
   ```

### Scope Phase B does **not** cover

- The Lustrek and `bot_m6` bots also call into `pimc::pimc_choose_card`.
  If either of them sits as an opponent in duplicate pods, their
  randomness is likewise uncorrelated active↔shadow. The seeded entry
  points above give us what we need to fix them later; doing so is
  mechanical (swap the call site and thread the flag), but out of scope
  for the initial Phase B landing.
- No scoring change. As per `copilot-instructions.md`: scoring is final.

## Testing notes

Phase A tests added:
- `test_duplicate_config.py::test_invalid_learner_seat_token_rejected`
  and sibling acceptance tests.
- `test_duplicate_centaur_wiring.py`: full round-trip through the
  use case + adapter proving `centaur` lands in `seat_config` and
  the `centaur_*` kwargs reach the fake `run_self_play` unchanged.
- `test_resolve_config.py`: top-level `centaur_*` and
  `duplicate.learner_seat_token` YAML parsing.

When Phase B lands, add:
- A Rust unit test that `pimc_choose_card_seeded(..., seed=X) ==
  pimc_choose_card_seeded(..., seed=X)` and differs for different seeds.
- A Python test that with `deterministic_seeding=True`, two
  `run_self_play` invocations on the same `deck_seeds` + seating +
  `model_path` produce byte-identical `scores`.
