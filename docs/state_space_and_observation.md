# Tarok Agent State Space And Observation Encoding

This document explains what data the learning agent receives during training.
It is based on the active Rust self-play path and encoder implementation.

## Short Answer

- The policy acts on a 578-dimensional state vector per decision (v6).
- Optional oracle critic training uses a 740-dimensional vector.
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

- Imperfect info actor state size: 578
- Oracle critic state size: 740 (= 578 + 162)

## Decision Points During Self-Play

At each decision point, self-play encodes state from the acting seat's perspective:

- BID (decision type 0)
- KING_CALL (decision type 1)
- TALON_PICK (decision type 2)
- CARD_PLAY (decision type 3)

Note: the network architecture has an announce head, but run_self_play currently emits these 4 decision types.

## 578-Dim Observation Layout (v6)

All offsets are zero-based and match engine-rs/src/encoding.rs.

Slovenian Tarok rule: the entire talon is revealed to all players before
the declarer picks one group.  The declarer's discard (put_down) remains
private.  Non-picked groups are publicly retired — they can never appear
in tricks.  In v6 these unpicked-talon cards are folded into the
declarer's per-opponent played plane (390..551) rather than a separate
global played plane.  Of the picked group, taroks and kings cannot
legally be discarded, so they are publicly known to still be in the
declarer's hand and are surfaced via the forced-retention plane
(108..161).

| Slice | Size | Meaning |
|---|---:|---|
| 0..53 | 54 | Own hand (binary card indicators) |
| 54..107 | 54 | Active trick plane (cards currently on the table) |
| 108..161 | 54 | Declarer forced-retention plane: taroks & kings from picked talon group (public) |
| 162..165 | 4 | Seat position relative to dealer (one-hot) |
| 166..175 | 10 | Contract one-hot |
| 176..178 | 3 | Phase one-hot: bidding / trick_play / other |
| 179 | 1 | Tricks won by my team (normalized by 12) |
| 180 | 1 | Tricks played (normalized by 12) |
| 181..185 | 5 | Decision type one-hot |
| 186..194 | 9 | Highest bid so far one-hot (no_bid + contracts) |
| 195..198 | 4 | Passed players bit flags (dealer-relative order) |
| 199..202 | 4 | Any team announced: trula, kings, pagat, valat |
| 203..207 | 5 | Kontra levels (normalized) |
| 208..210 | 3 | Role one-hot: declarer / partner / opposition (all-zero ⇒ role unknown) |
| 211..213 | 3 | Centaur team points: mine/70, opp/70, current_trick/20 |
| 214..217 | 4 | Trick leader relative seat one-hot |
| 218..221 | 4 | Trick currently-winning seat relative one-hot |
| 222..383 | 162 | Opponent belief block: 3 opponents × 54 cards (with forced-retention + void constraints) |
| 384..389 | 6 | Trick context: trick position + lead type/suit |
| 390..551 | 162 | Per-opponent played identity planes: 3 opponents × 54 (declarer plane ∪ unpicked talon) |
| 552..554 | 3 | Per-opponent tarok-void flag |
| 555..566 | 12 | Per-opponent suit-void flags (3 opponents × 4 suits) |
| 567..570 | 4 | Live kings one-hot (per suit) — 1 if that king has not been played |
| 571..573 | 3 | Live trula one-hot (pagat, mond, skis) — 1 if not yet played |
| 574..577 | 4 | Called-king suit one-hot (public once king is called) |

Total: 578.

### v6 vs v5 changes (blank slate — no checkpoint migration)

- Removed: global "publicly played" plane (subsumed by per-opp planes
  with declarer ∪ unpicked talon), `partner_known` flag (subsumed by the
  role one-hot: all-zeros means unknown), hand-strength summary (low
  mid-play signal).
- Added: running team card-point totals + current-trick value, trick
  leader relative seat, trick currently-winning seat.  These three
  features target the early/mid trick regime where the NN drives play
  (the centaur hands off to PIMC from trick 9 onward).
- Header plane reorder for cleaner card-attention channels:
  hand → active trick → declarer forced-retention.

## Oracle 740-Dim Layout

Oracle state = 578 base + 162 extra dims:

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

The belief block (222..383) is built per acting player:

- Known cards are removed: own hand, completed-trick cards, current-trick
  cards, and unpicked-talon cards (which are public and publicly
  retired).
- Forced retention: every tarok and king in the picked talon group is
  public knowledge that it sits in the declarer's hand.  In the belief
  block this pins probability 1.0 onto the declarer's column (and 0 on
  the other two opponent columns) for each such card.
- For the remaining unknown cards, each opponent initially gets uniform
  1/3 probability.
- If trick history implies an opponent is void in a suit (failed to
  follow suit, or trumped with a tarok), cards in that suit get zero
  probability for that opponent.
- If tarok was led and an opponent did not play a tarok, that opponent
  is marked tarok-void: all remaining tarok cards get zero probability
  for that opponent.

Partner-known flag (233) is true when the partner is publicly revealed
(called king has fallen) OR when the acting player already knows their
own pairing — the declarer always knows, and the hidden partner knows
as soon as the bidding is over because they hold the called king.

The public-memory tail (600..625) surfaces high-signal features
explicitly so the network doesn't have to re-derive them:

- 3 dims: per-opponent tarok-void flags.
- 12 dims: per-opponent suit-void flags (3 opponents x 4 suits).
- 4 dims: live kings — which of the four kings have not yet been played.
- 3 dims: live trula — whether pagat / mond / skis are still in play.
- 4 dims: called-king suit one-hot (public once the king has been called).

Opponent order in the 3x54 belief block, the 3x54 per-opponent played
planes, and all per-opponent v5 features is relative to the current
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

- states: (N, 578) float32
- actions: (N,) uint16
- log_probs: (N,) float32
- values: (N,) float32
- decision_types: (N,) uint8
- game_modes: (N,) uint8
- legal_masks: (N, 54) float32
- players: (N,) uint8
- game_ids: (N,) uint32
- scores: (num_games, 4) int32
- oracle_states: (N, 740) float32 (optional)

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