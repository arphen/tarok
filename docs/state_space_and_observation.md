# Tarok Agent State Space And Observation Encoding

This document explains what data the learning agent receives during training.
It is based on the active Rust self-play path and encoder implementation.

## Short Answer

- The policy acts on a 585-dimensional state vector per decision (v8).
- Optional oracle critic training uses a 747-dimensional vector.
- The agent sees previous trick information as public-card planes plus
  explicit per-opponent memory features (tarok-void, suit-void, per-opp
  played cards, live kings, live trula).  It does not receive an ordered
  per-trick transcript tensor.
- During self-play training, only learner seats ("nn") are emitted into the
  PPO batch.

## Source Of Truth In Code

- State encoder: engine-rs/src/encoding.rs
- Experience generation: engine-rs/src/self_play.rs
- Python bindings and tensor payload: engine-rs/src/py_bindings.rs
- PPO preprocessing and reward/GAE mapping: training-lab/training/adapters/ppo/ppo_batch_preparation.py

## Two Different "State Spaces"

## 1) True game state space (environment)

The underlying Tarok environment is a very large partially observed state space:

- Card allocation combinations (4x12 cards + 6 talon cards), roughly:

  C(54,12) * C(42,12) * C(30,12) * C(18,12) * C(6,6)

- Multiplied by phase, bidding history, contract, roles, trick progress, announcements, kontra levels, etc.

So the real environment state space is combinatorially huge.

## 2) Agent observation space (what model actually gets)

The policy/value network receives a fixed-size encoded vector:

- Imperfect info actor state size: 585
- Oracle critic state size: 747 (= 585 + 162)

## Decision Points During Self-Play

At each decision point, self-play encodes state from the acting seat's perspective:

- BID (decision type 0)
- KING_CALL (decision type 1)
- TALON_PICK (decision type 2)
- CARD_PLAY (decision type 3)

Note: the network architecture has an announce head, but run_self_play currently emits these 4 decision types.

## 585-Dim Observation Layout (v8)

All offsets are zero-based and match engine-rs/src/encoding.rs.

v8 groups all card planes contiguously at the start of the vector in a
card-attention-friendly layout (every plane starts on a 54-aligned
offset).  The scalar tail then follows from offset 486.

Slovenian Tarok rule: the entire talon is revealed to all players before
the declarer picks one group.  The declarer's discard (put_down) remains
private.  Non-picked groups are publicly retired — they can never appear
in tricks.  These unpicked-talon cards are folded into the declarer's
per-opponent played plane rather than a separate global played plane.
Of the picked group, taroks and kings cannot legally be discarded, so
they are publicly known to still be in the declarer's hand; this
information is carried by the belief block pinning those cards onto the
declarer's column (at probability 1.0) at offset 54..215.

| Slice | Size | Meaning |
|---|---:|---|
| 0..53 | 54 | Own hand (binary card indicators) |
| 54..107 | 54 | Opponent +1 belief probabilities |
| 108..161 | 54 | Opponent +2 belief probabilities |
| 162..215 | 54 | Opponent +3 belief probabilities |
| 216..269 | 54 | Own played cards (v8 new — fixes "disappearing pagat" problem) |
| 270..323 | 54 | Opponent +1 played cards |
| 324..377 | 54 | Opponent +2 played cards |
| 378..431 | 54 | Opponent +3 played cards (declarer plane ∪ unpicked talon) |
| 432..485 | 54 | Active trick plane (cards currently on the table) |
| 486..489 | 4 | Seat position relative to dealer (one-hot) |
| 490..499 | 10 | Contract one-hot |
| 500..502 | 3 | Phase one-hot: bidding / trick_play / other |
| 503 | 1 | Tricks played (normalized by 12) |
| 504..508 | 5 | Decision type one-hot |
| 509..517 | 9 | Highest bid so far one-hot (no_bid + contracts) |
| 518..521 | 4 | Passed players bit flags (dealer-relative order) |
| 522..525 | 4 | Own-team announcements: trula, kings, pagat, valat |
| 526..529 | 4 | Opponent-team announcements: trula, kings, pagat, valat |
| 530..534 | 5 | Kontra levels (normalized) |
| 535..537 | 3 | Role one-hot: declarer / partner / opposition (all-zero ⇒ bidding) |
| 538..541 | 4 | Partner relative seat one-hot (0 = self is partner; all-zero ⇒ unknown) |
| 542..544 | 3 | Centaur team points: mine/70, opp/70, current_trick/20 |
| 545..548 | 4 | Trick leader relative seat one-hot |
| 549..552 | 4 | Trick currently-winning seat relative one-hot |
| 553..558 | 6 | Trick context: trick position + lead type/suit |
| 559..561 | 3 | Per-opponent tarok-void flag |
| 562..573 | 12 | Per-opponent suit-void flags (3 opponents × 4 suits) |
| 574..577 | 4 | Live kings one-hot (per suit) — 1 if that king has not been played |
| 578..580 | 3 | Live trula one-hot (pagat, mond, skis) — 1 if not yet played |
| 581..584 | 4 | Called-king suit one-hot (public once king is called) |

Total: 585.

### v8 vs v7 changes (blank slate — no checkpoint migration)

- Added: own-played-cards plane at offset 216..269 (+54 dims).  In v7,
  the per-opponent played planes covered only opponents — the network
  had no direct signal for which cards *it* had personally played.
  Later in the game, this could make the network forget its own played
  specials (e.g. "where did my pagat go?"), which matters for pagat
  ultimo tracking and for not repeating known-impossible plays.  The
  new plane adds that information as a clean binary mask.
- Reorder: all nine card planes are now contiguous at the start of the
  vector (own hand → 3 belief → self-played → 3 opp-played → active
  trick), each starting on a 54-aligned offset.  This lets the card
  attention module in the network slice all per-card channels with
  uniform strides and without ad-hoc zero-padding.  The scalar tail
  follows unchanged from v7 (same 99 features), just shifted to offset
  486.

## Oracle 747-Dim Layout

Oracle state = 585 base + 162 extra dims:

- Extra 162 dims are exact opponent hands: 3 opponents × 54 cards.

This is for training-time critic/oracle use only (perfect training, imperfect execution).

## What The Agent Sees About History

## Yes, it sees previous trick information

Via:

- Per-opponent played identity planes (3×54): records who played what
  (declarer's plane also includes unpicked-talon cards).
- Team tricks won count, tricks played count, and running team
  card-point totals.
- Opponent summary stats (taroks played, suit cards played, kings played, total played).
- Void inference from trick history (used inside the belief block).

## What it does not directly see as an ordered sequence

- No explicit time-ordered tensor of prior tricks (for example, "trick 1 cards, trick 2 cards, ...").
- No explicit per-trick winner sequence vector.

So history is present mostly as public-card sets and derived summaries/inference, not a raw chronological replay tensor.

## Imperfect Information And Belief Encoding

The belief block (54..215) is built per acting player:

- Known cards are removed: own hand, completed-trick cards, current-trick
  cards, and unpicked-talon cards (which are public and publicly
  retired).
- Forced retention: every tarok and king in the picked talon group is
  public knowledge that it sits in the declarer's hand.  In the belief
  block this pins probability 1.0 onto the declarer's column (and 0 on
  the other two opponent columns) for each such card.  This pinning is
  the only surface for forced-retention — there is no dedicated plane.
- For the remaining unknown cards, each opponent initially gets uniform
  1/3 probability.
- If trick history implies an opponent is void in a suit (failed to
  follow suit, or trumped with a tarok), cards in that suit get zero
  probability for that opponent.
- If tarok was led and an opponent did not play a tarok, that opponent
  is marked tarok-void: all remaining tarok cards get zero probability
  for that opponent.

Role and partner identity are carried by two adjacent blocks:

- Role one-hot (535..537): declarer / partner / opposition.  The acting
  player always knows their own role once bidding ends (declarer is
  public; holding the called king ⇒ partner; otherwise opposition).
  All-zero means bidding is still in progress.
- Partner relative seat (538..541): one-hot identifying *which seat*
  holds the called king.  Populated only for players who actually know
  — the partner themselves, every seat once the called king has been
  publicly played, and self-called-king declarers.  Declarer cannot
  otherwise distinguish which opponent is the partner, so this stays
  all-zero for them.

The public-memory tail (559..584) surfaces high-signal features
explicitly so the network doesn't have to re-derive them:

- 3 dims: per-opponent tarok-void flags.
- 12 dims: per-opponent suit-void flags (3 opponents x 4 suits).
- 4 dims: live kings — which of the four kings have not yet been played.
- 3 dims: live trula — whether pagat / mond / skis are still in play.
- 4 dims: called-king suit one-hot (public once the king has been called).

Opponent order in the 3x54 belief block, the 3x54 per-opponent played
planes, and all per-opponent features is relative to the current
player:

- Opponent +1 seat, then +2, then +3 (modulo 4).

## Action Spaces And Legal Masks

The batch always carries a padded legal mask width of 54.
For non-card decisions, only the head-specific prefix is meaningful.

- BID: first 9 entries used
- KING_CALL: first 4 entries used
- TALON_PICK: first 6 entries used
- CARD_PLAY: all 54 entries used

Bid action index mapping:

- 0 pass
- 1 three
- 2 two
- 3 one
- 4 solo_three
- 5 solo_two
- 6 solo_one
- 7 solo
- 8 berac

King-call action mapping is by suit index:

- 0 hearts
- 1 diamonds
- 2 clubs
- 3 spades

Card-play action mapping is direct card index 0..53 in canonical deck order:

- 0..21 taroks I..XXII
- 22..53 suit cards in suit-major order (hearts, diamonds, clubs, spades), each with 8 ranks

## Training Payload Produced By run_self_play

Main tensors in the dictionary returned by run_self_play:

- states: (N, 585) float32
- actions: (N,) uint16
- log_probs: (N,) float32
- values: (N,) float32
- decision_types: (N,) uint8
- game_modes: (N,) uint8
- legal_masks: (N, 54) float32
- players: (N,) uint8
- game_ids: (N,) uint32
- scores: (num_games, 4) int32
- oracle_states: (N, 747) float32 (optional)

## How Rewards Are Assigned For PPO

In PPO preprocessing:

- Per-sample reward is taken from final game score for that sample's (game_id, player), scaled by /100.
- Rewards are zeroed on non-terminal steps of each (game, player) trajectory.
- GAE is then computed over the sorted (game, player) trajectories.

So intermediate steps learn via bootstrapped value/advantage targets, while terminal outcomes come from final game score.

## Practical Interpretation

If you are asking "does the agent remember what happened", the answer is:

- Yes, through public card accumulation, trick progress features, and inferred opponent constraints.
- No, not as a full explicit sequence model input.

If you want explicit chronological memory in the input, that would require extending the encoder layout and model architecture.