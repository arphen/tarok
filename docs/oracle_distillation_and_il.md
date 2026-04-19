# Oracle Distillation and IL in Tarok

This document explains, in plain language, how our Oracle Distillation and Imitation Learning (IL) setup works in the current training pipeline.

## Why This Exists

Tarok is an imperfect-information game. During real play, the model cannot see opponents' hidden cards or the unrevealed talon. That makes learning slow and noisy.

Our solution is to train with extra guidance (oracles) while keeping inference-time behavior legal and realistic.

## The Two Types of Oracles

| Type | What it is | How it is "Smarter" |
|---|---|---|
| The Expert Bot | A hand-coded heuristic bot or a stronger baseline policy (for example seats like `bot_v1`, `bot_v3`, `bot_m6`, `bot_v5`, `bot_v6`). | It already encodes practical strategy and game priors. It acts as a source of good trajectories and stable opponents. |
| The Privileged Agent | A neural sub-network used only in training with privileged inputs (oracle state). | It sees hidden information (opponents' card locations and talon context), so it learns a cleaner target representation than a blind actor can build alone. |

## Mental Model

Think of training as a classroom:

- The expert bots are the experienced players you scrim against.
- The privileged oracle is the coach who can see the full board.
- The student actor is what we deploy in production.

At deployment, only the student actor remains. No cheat codes are available at inference time.

## What "IL" Means Here

In this codebase, IL refers to oracle-guided representation distillation during PPO updates.

- The knob is `imitation_coef` (plus schedule settings).
- The actor is nudged to match privileged critic features.
- The matching objective is cosine-based latent distillation.

Important naming rule:

- `imitation_*` means **oracle latent distillation** only.
- `behavioral_clone_*` means **action-level behavioral cloning** only.

Mathematically:

$$
\mathcal{L}_{\text{IL}} = 1 - \cos\left(f_{\text{actor}}(s), f_{\text{oracle}}(s_{\text{oracle}})\right)
$$

This is then weighted by `imitation_coef` and added to PPO losses.

## Behavioral Cloning Warm-up (Action-Level)

Warm-up now uses a separate action-cloning path.

- Expert actions are generated from a V5 teacher.
- PPO adds a cross-entropy style term `-log pi(a_expert | s)` on those samples.
- This term is weighted by `behavioral_clone_coef`.
- Oracle latent distillation can be kept at `imitation_coef: 0` during warm-up.

Mathematically:

$$
\mathcal{L}_{\text{BC}} = -\log \pi_\theta(a_{\text{expert}} \mid s)
$$

## Behavioral Cloning Deep Dive (Implementation-Level)

This section describes exactly what happens in one iteration when BC is enabled.

### 1. Activation Gates

BC data is loaded only if all of these are true:

- `behavioral_clone_coef > 0`
- `behavioral_clone_games_per_iteration > 0`
- `behavioral_clone_teacher is not None`

If any condition fails, the iteration is standard PPO (plus optional IL).

### 2. Expert Data Source and Schema

Expert trajectories come from Rust:

- `tarok_engine.generate_expert_data(num_games, include_oracle=False)`

This provides, per decision:

- state
- expert action
- decision type
- legal-action mask (flattened stream; variable width by decision type)

Current teacher support is intentionally strict:

- `behavioral_clone_teacher: bot_v5` only

### 3. Legal Mask Reconstruction (Important)

The expert generator emits masks as one flat array where each row width depends on decision type:

- bid: 9
- king call: 4
- talon pick: 6
- card play: 54

The loader walks that flat stream using `decision_types`, then pads each reconstructed mask to width 54 (`CARD_ACTION_SIZE`).

Why this matters:

- PPO code expects a dense `(N, 54)` mask tensor, same shape as self-play.
- Without reconstruction/padding, reshaping by `(N, 54)` can fail or silently misalign masks.

### 4. How Expert Samples Are Tagged

Expert samples are converted into the same raw payload shape used by self-play, with neutral PPO terms and an explicit BC flag:

- `log_probs = 0`
- `values = 0`
- `scores = 0`
- `behavioral_clone_mask = True`

Self-play samples carry `behavioral_clone_mask = False`.

After merge, one mixed batch exists with both sources.

### 5. What PPO Preprocessing Does to Mixed Data

During `prepare_batched`:

- GAE and returns are still computed for all samples in the merged batch.
- advantages are globally normalized once.
- the BC mask is passed through unchanged as a boolean tensor.

In effect, BC examples flow through the same batching path as PPO examples, but are distinguishable at loss time.

### 6. Loss Computation on a Mixed Minibatch

For each minibatch and decision-head group, training computes:

- PPO policy loss (clipped surrogate)
- PPO value loss (clipped, return-normalized target space)
- optional IL latent loss
- optional BC action loss on BC-flagged rows only
- entropy bonus

The BC term is:

$$
\mathcal{L}_{\text{BC}} = -\frac{1}{|B_{\text{bc}}|}\sum_{i \in B_{\text{bc}}}\log\pi_\theta(a_i^{\text{expert}}\mid s_i)
$$

where $B_{\text{bc}}$ is the subset with `behavioral_clone_mask=True`.

Total loss is:

$$
\mathcal{L}=\lambda_p\mathcal{L}_{\text{PPO-policy}} +
\lambda_v\mathcal{L}_{\text{PPO-value}} +
\lambda_{il}\mathcal{L}_{\text{IL}} +
\lambda_{bc}\mathcal{L}_{\text{BC}} -
\lambda_H H(\pi)
$$

with:

- `\lambda_p = policy_coef`
- `\lambda_v = value_coef`
- `\lambda_{il} = imitation_coef`
- `\lambda_{bc} = behavioral_clone_coef`
- `\lambda_H = entropy_coef`

### 7. Why BC and Entropy Must Be Tuned Together

BC pushes policy confidence up (lower entropy), while entropy bonus pushes it down (more exploration).

During warm-up cloning, keep entropy low (for example `0.001`) so BC can imprint teacher behavior. High entropy in this phase often shows up as "almost-V5" play with avoidable random slips.

### 8. What Is and Is Not Being Cloned

Cloned:

- action choices from V5 at each decision point
- legal-action conditioning via masks

Not cloned:

- V5 value function
- V5 internal latent features
- privileged-oracle representation (that is IL, controlled separately by `imitation_*`)

### 9. Practical Read of Metrics

When BC is active, track at least:

- `bc_loss`: should trend down in warm-up
- `entropy`: should typically decline as policy sharpens
- `policy_loss`, `value_loss`: should remain stable, not explode

If `bc_loss` is flat and entropy stays high, BC is underweighted (or entropy is too high).

If value loss dominates and destabilizes actor updates, lower value pressure and/or verify return normalization and clipping are active.

### 10. Design Intent

BC is implemented as a first-class supervised signal that co-trains with PPO, not as a separate pretraining stage.

That gives:

- one optimizer path
- one replay schema
- one training loop
- explicit coefficient control for smooth handoff from cloning-heavy warm-up to RL-heavy refinement

## Current Training Flow (High Level)

1. Collect self-play experiences.
2. If oracle mode is active and IL weight is positive, also include oracle states.
3. Optionally mix in expert V5 action samples when behavioral cloning is enabled.
4. Run PPO updates.
5. Add IL/distillation loss that aligns actor features to privileged critic features.
6. Add BC action loss for expert-labeled warm-up samples.
7. Continue with scheduled IL coefficient (constant/linear/cosine, or Gaussian Elo policy when configured).

## When Oracle States Are Included

Oracle states are only passed through when all conditions are true:

- model identity has `oracle_critic=True`
- current iteration IL coefficient is greater than zero

If either is false, training proceeds without oracle-state distillation.

## Where the Pieces Live

- `training-lab/configs/warm-up.yaml`: bootstrap profile (high BC weight, easy classroom opponents).
- `training-lab/training/use_cases/collect_experiences.py`: gate for `include_oracle_states`.
- `training-lab/training/adapters/ppo/torch_ppo.py`: PPO + oracle distillation + BC action loss.
- `training-lab/training/adapters/ppo/expert_replay.py`: V5 expert sample loader.
- `training-lab/training/entities/training_config.py`: IL/distillation and BC coefficient definitions.
- `training-lab/training/use_cases/train_model/policies.py`: schedule policies (including Elo-aware variants).
- `training-lab/training/use_cases/resolve_model.py`: new models default to `oracle_critic=True`.

## Practical Interpretation

- Expert Bot oracle improves data quality and opponent diversity.
- Privileged oracle improves training signal quality.
- The student actor learns to infer hidden structure from public signals.
- Inference stays clean: no hidden-state access required.

This is the core reason the system can learn faster while still producing a valid imperfect-information player at runtime.