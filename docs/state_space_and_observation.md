# Tarok Agent State Space And Observation Encoding

This document is the design spec for the observation vector fed to the
policy / value networks. It is the contract between the Rust encoder
([engine-rs/src/encoding.rs](../engine-rs/src/encoding.rs)) and the
Python encoder ([model/src/tarok_model/encoding.py](../model/src/tarok_model/encoding.py)).

Hard rule for this spec: every feature must be derivable from public
game state (or from the acting player's private knowledge) via a rule
stated in writing here. We do **not** feed the network heuristics,
"good practice" tells, or bluffing priors — only deterministic
inferences from the rules of Slovenian Tarok.

- Current runtime implementation: v9 (639 dims actor / 801 dims oracle)
- Next layout under design in this file: **v10** (585 dims actor / 747 dims oracle)

## Table of contents

1. [Two different "state spaces"](#two-different-state-spaces)
2. [Decision points during self-play](#decision-points-during-self-play)
3. [v10 observation layout](#v10-observation-layout)
4. [Card planes — rules and math](#card-planes--rules-and-math)
5. [Belief plane construction (cardinality-aware)](#belief-plane-construction-cardinality-aware)
6. [Scalar tail — rules and math](#scalar-tail--rules-and-math)
7. [Provable inferences we add in v10](#provable-inferences-we-add-in-v10)
8. [What we deliberately do NOT encode](#what-we-deliberately-do-not-encode)
9. [v10 vs v9 changes](#v10-vs-v9-changes)
10. [Oracle 766-dim layout](#oracle-766-dim-layout)
11. [Action spaces and legal masks](#action-spaces-and-legal-masks)
12. [Training payload produced by run_self_play](#training-payload-produced-by-run_self_play)
13. [How rewards are assigned for PPO](#how-rewards-are-assigned-for-ppo)

---

## Two different "state spaces"

### True game state space (environment)

The underlying partially observed state space has size roughly

$$
\binom{54}{12} \binom{42}{12} \binom{30}{12} \binom{18}{12} \binom{6}{6} \times (\text{phase} \times \text{bid history} \times \text{contract} \times \text{roles} \times \text{announcements} \times \text{kontra} \times \text{trick progress})
$$

i.e. combinatorial — we never materialise this.

### Agent observation space (what the model receives)

Fixed-size per-decision vector:

- **v10 actor state: 585 dims**
- **v10 oracle critic state: 747 dims** (= 585 + 3 × 54 exact opponent hands)

## Decision points during self-play

Self-play encodes state from the *acting* seat's perspective at each
decision point. Each uses the same 585-dim layout; the decision-type
one-hot at offset 504..508 disambiguates.

- `BID` (dt = 0)
- `KING_CALL` (dt = 1)
- `TALON_PICK` (dt = 2)
- `CARD_PLAY` (dt = 3)
- `ANNOUNCE` (dt = 4) — architecturally supported; not yet emitted by `run_self_play`.

## v10 observation layout

All offsets are zero-based. Card planes occupy the first 9 × 54 = 486
dims (54-aligned so the card-attention module can slice uniformly).
The scalar tail follows from offset 486.

| Slice      | Size | Field                                                    |
|-----------:|-----:|----------------------------------------------------------|
| 0..53      |  54  | Own hand (binary)                                         |
| 54..107    |  54  | Opponent +1 belief marginals (cardinality-aware)          |
| 108..161   |  54  | Opponent +2 belief marginals                              |
| 162..215   |  54  | Opponent +3 belief marginals                              |
| 216..269   |  54  | Own played cards                                          |
| 270..323   |  54  | Opponent +1 played cards                                  |
| 324..377   |  54  | Opponent +2 played cards                                  |
| 378..431   |  54  | Opponent +3 played cards (declarer plane ∪ unpicked talon)|
| 432..485   |  54  | Active trick (cards currently on the table)              |
| 486..489   |   4  | Seat position relative to dealer (one-hot)                |
| 490..499   |  10  | Contract one-hot                                          |
| 500..502   |   3  | Phase one-hot: bidding / trick_play / other               |
| 503        |   1  | Tricks played / 12                                        |
| 504..508   |   5  | Decision type one-hot                                     |
| 509..517   |   9  | Highest bid so far (no_bid + 8 contracts)                |
| 518..521   |   4  | Passed players (bit flags, dealer-relative order)         |
| 522..526   |   5  | Own-team announcements: trula, kings, pagat, king, valat  |
| 527..531   |   5  | Opponent-team announcements: same 5                       |
| 532..536   |   5  | Kontra levels (normalised)                                |
| 537..539   |   3  | Role one-hot: declarer / partner / opposition             |
| 540..543   |   4  | Partner relative seat one-hot (all-zero ⇒ unknown)        |
| 544..546   |   3  | Centaur team points: mine/70, opp/70, current_trick/20    |
| 547..550   |   4  | Trick leader relative seat one-hot                        |
| 551..554   |   4  | Trick currently-winning seat relative one-hot             |
| 555..560   |   6  | Trick context: position + lead type/suit                  |
| 561..563   |   3  | Per-opponent tarok-void flag                              |
| 564..575   |  12  | Per-opponent suit-void flags (3 × 4)                      |
| 576..579   |   4  | Called-king suit one-hot (public once king is called)     |
| 580..584   |   5  | Remaining-in-play counts: taroks/22, H/8, D/8, C/8, S/8   |

**Total: 585.**

---

## Card planes — rules and math

All nine card planes are binary or probability masks indexed by the
canonical 54-card deck:

- Indices 0..21 → Taroks I..XXII (Tarok I = Pagat, XX = Mond, XXI = Skis
  form the *trula*; the Rust canonical index of Skis is 21).
- Indices 22..53 → Suit cards (8 per suit × 4 suits), suit-major order
  hearts, diamonds, clubs, spades; rank-minor within each suit
  (4–10, valet, cavall, dama, kralj — exact rank order follows
  `engine-rs/src/card.rs`).

### Plane 0 — Own hand (0..53)

Binary. `1` iff card is currently in the acting player's hand.
Post-talon-pick this includes the picked group; it excludes `put_down`.

### Planes 1..3 — Belief marginals (54..215)

Opponents are ordered relative to the acting player: +1, +2, +3 mod 4.

Each opponent's column is the **marginal** $P[\text{card } c \in \text{opp}_k]$
under a uniform prior over all card placements that are consistent with
every public constraint listed below. Construction algorithm is in the
dedicated section below.

### Plane 4 — Own played cards (216..269)

Binary. `1` iff the acting player has played this card in a completed
trick or in the currently-active trick.

Rationale: fixes the "disappearing pagat" problem — the network needs
a first-class signal for what *it* has already played so it can track
pagat ultimo and never re-plan impossible plays.

### Planes 5..7 — Per-opponent played cards (270..431)

Binary. `1` iff the referenced opponent has played that card in a
completed trick or in the active trick.

Special case for the declarer's plane: unpicked talon groups are
publicly retired the moment the declarer picks their group, so they
cannot appear in any future trick. We attribute those cards to the
declarer's played plane (not to an eliminated "ghost" plane) because
that is the plane representing cards that have *left play through the
declarer side of the table*.

### Plane 8 — Active trick (432..485)

Binary. `1` iff the card is currently on the table in the trick that
has not yet been resolved.

---

## Belief plane construction (cardinality-aware)

This is the core piece the v10 spec tightens vs. v9. v9 filled every
cell with either `0` (impossible) or a flat `1/3` (feasible across
three opponents). v10 replaces the flat `1/3` with the **exact
marginal** under uniform prior subject to all public constraints.

### Notation

- `hand_i` = size of opponent *i*'s remaining hand (3 values).
- `V(i)` = set of suits opponent *i* is publicly void in.
- `T(i)` = true iff opponent *i* is publicly tarok-void.
- `U` = set of cards whose location is unknown to the acting player.

### Step 1 — Build the "known-out" set

```
known =  own_hand
       ∪ cards played in any completed trick
       ∪ cards on the current trick
       ∪ unpicked_talon
       ∪ forced_retention       (see step 2)
       ∪ (put_down  if  acting_player == declarer)   ← v10 addition
```

The `put_down` line replaces v9's separate own-discarded plane.
Rationale: the only player who *knows* `put_down` is the declarer.
Adding it to `known` makes every such card impossible for every
opponent (belief 0 in all three columns), which is strictly stronger
information than a flag plane — and it is only applied when the
acting player is the declarer, so opponents still see `put_down` as
part of the unknown set. Non-declarer networks are completely
unaffected: their `known` set does not include `put_down`.

### Step 2 — Forced retention (picked talon → declarer pin)

In Slovenian Tarok the entire talon is revealed to all players before
the declarer picks one group. Once picked, every **tarok** or **king**
in the picked group is publicly known to sit in the declarer's hand
(they cannot legally be discarded; only low suit cards can). We
enforce this by *pinning* those cards to the declarer's belief column:

```
for c in picked_group:
    if c is tarok or c is king:
        belief[+k, c] = 1.0  where +k is the declarer's relative offset
        belief[other opps, c] = 0
```

There is no standalone "forced retention" plane — the pin is the only
surface needed.

### Step 3 — Void constraints

From trick history, extract:

- **Suit-void.** Suit $s$ is led and opponent *i* played a card not of
  suit $s$ → opp *i* is void in $s$. This includes trumping with a
  tarok *and* discarding a non-$s$ suit card (rule: follow suit if you
  can, else play tarok if you can, else anything).
- **Tarok-void (direct).** Tarok is led and opponent *i* played a
  non-tarok → opp *i* is tarok-void.
- **Tarok-void (inferred from second-step suit failure).** Suit $s$ is
  led, opp *i* played a **non-tarok non-$s$ suit card** → opp *i* is
  simultaneously void in $s$ (already captured above) *and* tarok-void
  (this is new in v10 — v9 missed this tarok-void case).

Formal justification for the new case: the forced-follow rule requires
tarok when suit-void. If opp *i* was suit-void in $s$ and still did
not play a tarok, they must have been tarok-void at that moment; the
tarok-void property is monotone in time (you can never acquire taroks
mid-game), so it holds now too.

Zero out belief cells that violate voids:

```
for cidx in U:
    c = DECK[cidx]
    if c is tarok and T(i):         belief[i, cidx] = 0
    if c is suit s and s ∈ V(i):    belief[i, cidx] = 0
```

### Step 4 — Cardinality-aware renormalisation (v10 addition)

After steps 1–3 every belief cell is either pinned (∈ {0, 1}) or free.
For free cells, we want the marginal under a uniform prior over
consistent hand assignments, i.e. each card in $U$ is placed in
opponent *i* with the probability that a uniform random draw from all
assignments satisfying

1. each opponent *i* receives exactly `hand_i` cards,
2. no opponent *i* receives any card forbidden by void / pin,

places that card in opponent *i*.

#### Why a flat 1/3 is wrong

Example at the end of trick 9: three opponents have `(2, 2, 2)` cards
left (six unknown). Assume only 3 hearts remain unseen and opponents 1
and 2 are heart-void. Then opponent 3 must hold all 3 hearts — yet
they only have 2 slots. Contradiction ⇒ the scenario is impossible
under the flat prior, but the flat prior cannot detect this; v10's
normalisation forces the 3 hearts into opp 3 and propagates the
resulting slack to the remaining unknowns.

#### Exact algorithm (iterative proportional fitting, aka Sinkhorn)

Let $M \in [0, 1]^{3 \times |U|}$ be the belief matrix (rows =
opponents, cols = unknown cards), pre-seeded as:

- 0 for pinned-absent and void-forbidden cells,
- 1 for pinned-present cells (the picked-talon force-retention pins),
- 1 otherwise (feasible cells — the "uniform" starting weight).

Let $r_i = \text{hand}_i - (\text{cards already pinned to opp } i)$
be the **free slots** row target, and $c_j = 1 - (\text{pinned column mass})$
be the column target (almost always $c_j = 1$; it is $0$ when the card
is pinned so no rescaling of that column is needed).

Iterate until convergence (typically < 10 iterations, numerically
robust):

```
repeat:
    # Row scaling: each opponent fills exactly their remaining slots.
    for i in {0,1,2}:
        s = sum_j M[i,j]
        if s > 0: M[i, :] *= r_i / s
    # Column scaling: each unknown card is held by exactly one opponent.
    for j in U:
        s = sum_i M[:,j]
        if s > 0: M[:, j] *= c_j / s
until max change < 1e-4
```

Complexity: $O(|U| \cdot 3 \cdot \text{iters})$ ≤ $O(54 \cdot 3 \cdot 10)$
per encode — cheap.

#### Edge cases

- If $r_i = 0$ (opponent has no free slots; e.g. all their cards
  pinned): row *i* collapses to zero, making every unpinned card
  impossible for them. Correct.
- If an unknown card has no feasible opponent (all three zeroed by
  voids/pins): the scenario is infeasible given our constraint
  set, which can only happen if upstream constraints are
  inconsistent. We fall back to the pre-IPF matrix for that column
  and log a debug-only warning rather than crash the encoder.

#### Determinism

The encoder runs the same fixed iteration count with fixed
initialisation, so two identical game states always produce the same
belief marginals. No randomness, no sampling.

---

## Scalar tail — rules and math

Each field below has a precise rule. Where the rule is trivial the
entry is short; where it hides game-rule logic the derivation is
included.

### Seat position relative to dealer (486..489)

One-hot of $(\text{player} - \text{dealer}) \bmod 4$. Used so the
policy can model positional effects (forehand leads first trick, etc.)
without baking in specific seats.

### Contract one-hot (490..499)

Index 0..9 over `Klop, Three, Two, One, SoloThree, SoloTwo, SoloOne,
Solo, Berac, BarvniValat`. All zero during bidding (contract = None).

### Phase one-hot (500..502)

Bidding / TrickPlay / other (scoring, announcement, talon-pick etc.).

### Tricks played / 12 (503)

Just $n_\text{tricks\_completed}/12$; a simple time feature.

### Decision type one-hot (504..508)

Identifies which head should act on this observation.

### Highest bid so far (509..517)

One-hot over `no_bid + 8 contracts` (Three, Two, One, SoloThree,
SoloTwo, SoloOne, Solo, Berac). Computed via
`max_by_key(strength)` over accepted bids.

### Passed players (518..521)

Bit flags in dealer-relative order. Active only in the bidding phase.

### Announcement flags (522..531) — **v10 expanded to 5 each**

Both own-team and opp-team blocks now cover five announcements:

```
index 0: Trula
index 1: Kings
index 2: Pagat Ultimo   (Pagat wins the last trick)
index 3: King Ultimo    (the called king wins the last trick)   ← v10
index 4: Valat
```

The encoder splits the public `announcements` set by team (via
`state.get_team(seat) == my_team`). King Ultimo is mathematically
distinct from Pagat Ultimo (different card, different winning
condition), so it earns its own flag rather than being conflated with
Pagat Ultimo. See "v10 vs v9 changes" for the requirement to add
`Announcement.KING_ULTIMO` to the entity enum.

### Kontra levels (532..536)

Five levels in `KontraTarget` order (game + four bonus contracts),
encoded as `(multiplier - 1) / 7` so `None → 0`, `Kontra → 1/7`,
`Re → 3/7`, `Sub → 7/7`.

### Role one-hot (537..539)

Populated only after bidding resolves:

- Acting player == declarer ⇒ index 0.
- Acting player == public partner (holds called king) ⇒ index 1.
- Acting player is on `Team::DeclarerTeam` for any other reason
  (self-call) ⇒ index 1.
- Otherwise ⇒ index 2 (opposition).

All-zero ⇒ still bidding.

### Partner relative seat (540..543)

One-hot over the 4 dealer-independent relative offsets `(partner -
player) mod 4`. Populated only when the acting player *knows* who the
partner is:

- The partner seat always knows itself.
- Once the called king is publicly played, every seat knows.
- If the declarer self-called the king, the declarer knows.

For every other seat this block stays all-zero.

### Centaur team points (544..546)

- `my_team_points / 70`
- `opp_team_points / 70`
- `current_trick_points / 20`

Computed by actually running `trick_eval::evaluate_trick` on every
completed trick (respecting last-trick bonus rules), then summing
card points by the winner's team.

### Trick leader / winner relative seat (547..554)

Two 4-wide one-hots keyed by `(seat - player) mod 4`. The
currently-winning seat is computed by replaying `beats()` across the
cards already on the table with the lead suit fixed by the lead card.

### Trick context (555..560)

```
555: trick.count / 4            # how many seats have played so far
556: lead_is_tarok flag
557..560: lead suit one-hot     (hearts, diamonds, clubs, spades)
```

### Per-opponent tarok-void flag (561..563)

Result of the void inference rules above: one bit per opponent (+1,
+2, +3). Includes v10's new suit-lead ∧ non-tarok-non-suit case.

### Per-opponent suit-void flags (564..575)

`(3 opponents) × (4 suits)`, row-major per opponent.

### Called-king suit (576..579)

One-hot over the 4 suits. Populated once the king is called
(`state.called_king` is set), all-zero otherwise. Note that
*solo-calling* is a legal move (declarer calls their own king) — the
suit is still public.

---

## Provable inferences we add in v10

Every feature in this section is a deterministic function of public
state under the rules of Slovenian Tarok. No heuristics.

### Remaining-in-play counts per bucket (580..584)

```
580: (22 - taroks_played_globally) / 22
581: ( 8 - hearts_played_globally ) /  8
582: ( 8 - diamonds_played_globally ) / 8
583: ( 8 - clubs_played_globally    ) / 8
584: ( 8 - spades_played_globally   ) / 8
```

Global counters over every completed trick and the current trick.
These are cheap summaries that avoid the network having to sum 54
plane bits per decision.

---

## What we deliberately do NOT encode

Explicit exclusion list to keep us disciplined. Everything here is
either redundant with other fields or non-provable heuristic:

- **Live kings one-hot** (was 628..631 in v9): derivable in one
  element-wise op as `1 - OR(played planes)[king_indices]`. Removed.
- **Live trula one-hot** (was 632..634 in v9): same argument for
  Pagat / Mond / Skis.
- **Per-opponent remaining hand size**: derivable by summing each
  opponent's played-cards plane. The belief IPF already consumes hand
  sizes internally during construction; no need to re-surface the
  scalar to the policy head.
- **Per-opponent tarok rank ceiling, per-opponent × suit rank ceiling,
  overplay-contract indicator**: these are Klop/Berač-specific overplay
  inferences. Klop and Berač have their own dedicated network head;
  the PIMC planner already exploits these bounds directly via
  `PlayerConstraints` in `engine-rs/src/pimc.rs`. Feeding them to the
  actor observation would duplicate work that belongs to the Klop/Berač
  head and its PIMC layer, not to the shared belief encoder.
- **Own-discarded plane** (was 270..323 in v9): removed in favour of
  `put_down ∈ known` in the declarer's belief construction. For the
  declarer: equivalent — actually strictly stronger, because it also
  zeroes the cells for the opponents. For non-declarers: `put_down`
  is private, so we explicitly do **not** apply the zeroing; they
  keep the v9 behaviour and lose nothing.
- **Ordered per-trick transcript tensor**: no chronological history
  plane. History is surfaced as public-card sets, scalar summaries,
  and derived constraints. Adding an explicit time tensor would
  require a recurrent/attention block we have not budgeted for.
- **Opponent "likely" card distributions from human play patterns**:
  no heuristic priors. The only prior is uniform over feasible
  worlds.
- **Per-trick winner sequence**: derivable from the completed-trick
  encoding if you need it; not a first-class feature.
- **Announcement-based positional priors** (e.g. "pagat ultimo ⇒
  announcer will save the Pagat for last"): that is opponent
  strategy, not a rule-derived fact, and has no place here.

## v10 vs v9 changes

- **Removed** (−61 dims): own-discarded plane (−54), live kings (−4),
  live trula (−3).
- **Added** (+7 dims): King Ultimo announcement slots (+2), remaining
  deck counts per bucket (+5).
- **Restructured**: scalar tail shifted down by 54 because the
  own-discarded plane is gone.
- **Tightened**: belief planes now use cardinality-aware IPF
  normalisation instead of a flat `1/3` prior, and include
  `put_down ∈ known` for the declarer perspective.
- **Fixed**: tarok-void inference now also fires when a suit-void
  opponent discards a non-tarok non-suit card (see rules block).
- **Not added** (vs earlier draft): per-opp remaining hand size,
  tarok/suit rank ceilings, overplay-contract indicator — see
  "What we deliberately do NOT encode" for reasoning.

Net delta: **639 → 585 dims** (−54). Oracle state: **801 → 747 dims**.

Migration note: v10 requires a blank-slate retrain.
`Announcement` enum in
[`backend/src/tarok/entities/game_types.py`](../backend/src/tarok/entities/game_types.py)
must gain a new member `KING_ULTIMO = "king_ultimo"`; the
announce-action mapping in the encoder (`ANNOUNCE_IDX_TO_ANN`) and
the scorer must be updated in the same change.

## Oracle 766-dim layout

Oracle state = 585 base + 162 extra dims:

- 162 extra dims are exact opponent hands (3 opponents × 54 cards,
  binary) at offsets 585..746.

Used for training-time critic / oracle distillation only (perfect
training, imperfect execution).

## Action spaces and legal masks

Unchanged from v9. The batch carries a padded legal mask of width 54.
Only the head-specific prefix is meaningful for non-card decisions.

- `BID`: first 9 entries
- `KING_CALL`: first 4 entries
- `TALON_PICK`: first 6 entries
- `CARD_PLAY`: all 54 entries

### Bid action index

0 pass · 1 three · 2 two · 3 one · 4 solo_three · 5 solo_two ·
6 solo_one · 7 solo · 8 berac.

### King-call action index

0 hearts · 1 diamonds · 2 clubs · 3 spades.

### Card-play action index

Direct canonical deck index 0..53:

- 0..21 Taroks I..XXII
- 22..53 Suit cards (hearts, diamonds, clubs, spades × 8 ranks)

## Training payload produced by run_self_play

Dictionary produced by
[`run_self_play`](../engine-rs/src/py_bindings.rs):

- `states`: (N, 585) float32
- `actions`: (N,) uint16
- `log_probs`: (N,) float32
- `values`: (N,) float32
- `decision_types`: (N,) uint8
- `game_modes`: (N,) uint8
- `legal_masks`: (N, 54) float32
- `players`: (N,) uint8
- `game_ids`: (N,) uint32
- `scores`: (num_games, 4) int32
- `oracle_states`: (N, 747) float32 (optional)
- `initial_hands`, `initial_talon`, `traces`: optional replay payload

Only learner seats (those labelled `"nn"` or `"centaur"`) contribute
experiences to the batch.

## How rewards are assigned for PPO

See [`training-lab/training/adapters/ppo/ppo_batch_preparation.py`](../training-lab/training/adapters/ppo/ppo_batch_preparation.py).

- Per-sample reward = final game score for `(game_id, player)`, scaled
  by `/100`.
- Non-terminal steps in each `(game, player)` trajectory get reward
  zero.
- GAE is computed over the sorted `(game, player)` trajectories.

Intermediate steps learn via bootstrapped value/advantage targets;
terminal outcomes come straight from the final scoring rule.
