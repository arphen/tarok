# Outplace Metric — How It Works and How It Calibrates

This note explains what the **outplace** metric actually measures in training
reporting and Elo updates, how it behaves when a table has multiple learner
seats, and what the *expected* outplace rate is when every player is of equal
strength. The goal is to make the metric legible enough that we can tune Elo
K-factors without being surprised by its variance.

Code references:
- [training-lab/training/adapters/self_play/rust_self_play_adapter.py](training-lab/training/adapters/self_play/rust_self_play_adapter.py) — `compute_run_stats`
- [training-lab/training/use_cases/update_league_elo.py](training-lab/training/use_cases/update_league_elo.py) — `UpdateLeagueElo.execute`
- [training-lab/training/entities/league.py](training-lab/training/entities/league.py) — `LeaguePoolEntry`, `LeagueConfig`
- [docs/dev/league_play.md](docs/dev/league_play.md) — existing league/Elo overview

---

## 1. Definition

An **outplace** is a pairwise score comparison between one *learner* (NN) seat
and one *opponent* seat over a comparison unit (either a single game or a
cumulative session).

For a single comparison unit:

| Relation | Counts as |
|---|---|
| `learner_score > opponent_score` | `learner_outplaces` |
| `learner_score < opponent_score` | `opponent_outplaces` |
| `learner_score == opponent_score` | `draws` |

The reported per-entry rate is:

```text
decisive_games = learner_outplaces + opponent_outplaces
outplace_rate  = learner_outplaces / decisive_games
```

**Draws are excluded from the denominator.** A tie on the comparison unit
neither helps nor hurts the reported rate — because a "same" outcome is
strictly better for the network than a "worse" outcome, and weighting draws
equal to losses (the old behaviour, `learner_outplaces / games_played`)
unfairly penalised networks that played to a draw against equal-strength
opponents. This makes the dashboard number intuitive: **0.5 = equal
strength on decisive comparisons, >0.5 = beating the opponent, <0.5 =
losing to it.**

For the Elo update, draws still count as 0.5 — so ratings stay calibrated
even though the reported rate ignores draws:

```text
n_games         = learner_outplaces + opponent_outplaces + draws
learner_outcome = (learner_outplaces + 0.5 * draws) / n_games
```

---

## 2. Comparison units: single games vs sessions

`outplace_session_size` (default `50`) controls the unit. Logic in
`compute_run_stats`:

- If `session_size <= 1` or fewer than one session’s worth of games are
  available, comparisons are **per-game**.
- Otherwise, scores are summed over non-overlapping chunks of `session_size`
  games, and comparisons are done on those **cumulative session totals**.
  Leftover games that don’t fill a full session are ignored for outplace
  counts (they still show up in `mean_scores`).

Session-based comparison dramatically reduces draw frequency: over 50 games of
zero-sum-ish tarok scoring, cumulative totals almost never tie.

---

## 3. Multiple learner seats at the same table

When `min_nn_per_game > 1`, several seats can be `nn`. For example, the
constellation

```
seat 0: nn
seat 1: bot1
seat 2: bot2
seat 3: nn
```

generates the following comparisons **per comparison unit**:

| Opponent seat | NN seats compared | Outcomes contributed |
|---|---|---|
| seat 1 (`bot1`) | seat 0, seat 3 | 2 pairwise outcomes → `bot1` bucket |
| seat 2 (`bot2`) | seat 0, seat 3 | 2 pairwise outcomes → `bot2` bucket |

So each unit yields **4 pairwise outcomes** total (2 per bot token). Both
learner seats contribute; nothing is discarded. The two learners share
weights, so this is not two independent samples — seat 0 and seat 3
correlate through identical policy choices but play different hands.

If the same bot token fills two seats (e.g. `nn,bot_v5,bot_v5,nn`), each seat
index still triggers its own Elo update on the matching pool entry. That
entry therefore gets updated twice per iteration.

---

## 4. Expected outplace rate when everyone is equal strength

This is the calibration question. We analyse the `nn,bot1,bot2,nn` case
assuming all four models are equally strong.

### 4a. Per-game rate (session_size = 1)

In a normal (non-Klop) tarok game with this repo’s scoring rule (declarer
±X, everyone else 0), and assuming:

- declarer seat is uniformly distributed over the 4 seats,
- declarer wins with probability 0.5 given equal strength,

then for any fixed pair of seats (A, B) and a single game:

```
P(A is declarer, wins)  = 1/4 * 1/2 = 0.125   → A > B
P(A is declarer, loses) = 1/4 * 1/2 = 0.125   → A < B  (A gets -X, B gets 0)
P(B is declarer, wins)  = 1/4 * 1/2 = 0.125   → B > A
P(B is declarer, loses) = 1/4 * 1/2 = 0.125   → A > B  (A gets 0, B gets -X)
P(neither declarer)     = 2/4       = 0.500   → tie (both 0)
```

Totals per game:

| Event | Probability |
|---|---|
| A outplaces B | 0.25 |
| B outplaces A | 0.25 |
| draw | 0.50 |

Because draws are excluded from the reported denominator
(`decisive_games = outplaces_{A} + outplaces_{B}`), the **expected
per-game `outplace_rate` at equal strength is 0.5**, not 0.25 as under the
old "draws-count-as-losses" definition. Ties still dominate the raw game
count (0.5 of games are draws in this equal-strength model), but they no
longer deflate the reported number — the rate reflects only decisive
comparisons.

Klop games shift the *share* of draws (Klop scores individually, so ties
are rare), but the reported rate stays near 0.5 at equal strength because
it already conditions on decisive outcomes.

For the Elo *outcome* value (`learner_outcome`), draws are 0.5, so:

```
E[learner_outcome] = 0.25 + 0.5 * 0.5 = 0.5
```

Elo therefore has zero expected drift at equilibrium, even though the
display number is 0.25.

### 4b. Per-session rate (session_size = 50)

Summing 50 game scores per seat produces near-continuous totals. Ties
virtually disappear. With equal strength, the cumulative-total comparison
becomes a fair coin flip:

```
E[learner_outplaces / n_sessions] ≈ 0.5
E[draws] ≈ 0
```

So reported `outplace_rate` at equilibrium with `session_size = 50` is
**~0.5**, and Elo drift is again zero in expectation.

### 4c. Summary table

| Unit | Expected `outplace_rate` at equal strength | Expected Elo drift |
|---|---|---|
| per game (session_size=1) | ~0.5 (draws excluded from denominator) | 0 |
| session_size=50 | ~0.5 | 0 |

---

## 5. Variance — why Elo swings so hard per iteration

Two separate problems stack here:

### 5a. K-factor scaling

`UpdateLeagueElo` uses

```text
k_effective = 32 * max(1.0, elo_outplace_unit_weight)
```

The shipped configs (`self-play.yaml`, `warm-up.yaml`, `handoff.yaml`) set
`elo_outplace_unit_weight: 5`, so `k_effective = 160`. The default in
`LeagueConfig` is `1.0` (K = 32). The comments in `update_league_elo.py`
describe a stronger default (`= outplace_session_size`, giving K = 1600),
which is what the "brutal" intuition refers to.

### 5b. Aggregate-outcome variance

The Elo update is applied **once per iteration per seat-entry** using the
aggregate `learner_outcome` fraction across *all* comparison units that
iteration. For N Bernoulli(0.5) sessions the outcome-proportion standard
deviation is `0.5 / sqrt(N)`, so the per-iteration Elo swing has

```
σ(ΔElo) ≈ k_effective * 0.5 / sqrt(N)
```

Example: iteration with `games: 10000`, `session_size: 50`, two learner
seats against one bot token sitting in seat 1:
- N = (10000/50) * 2 = 400 units
- σ(outcome) ≈ 0.5 / √400 = 0.025
- σ(ΔElo) ≈ 160 * 0.025 = **±4 points / iteration / entry** at K=160
- At K=1600 (unit weight = 50): **±40 points / iteration / entry**

Over 50 iterations this is a random walk with stdev ≈ √50 × σ:
- K=160  → ±28 rating points pure drift
- K=1600 → ±280 rating points pure drift

The drift scales inversely with `sqrt(games_per_iteration)` — doubling
`games` halves the per-iteration σ, but only helps by √2.

---

## 6. Practical calibration implications

1. **Read per-game `outplace_rate` directly as a win rate on decisive
   games.** The equal-strength equilibrium is ~0.5 at any session size.
   A sustained value below 0.5 means the learner loses more decisive
   comparisons than it wins; above 0.5 means the opposite.

2. **Prefer session-based outplacing (`session_size=50`) for human-readable
   dashboards.** The number then matches intuition: 0.5 = equal, >0.5 =
   beating the bot.

3. **Treat `elo_outplace_unit_weight` as the variance dial, not a correctness
   dial.** Setting it equal to `outplace_session_size` (as the code comments
   suggest) treats "one session ≈ one decisive comparison" — fast to
   converge but very noisy. The shipped value `5` is a reasonable middle
   ground; `1` is the least noisy.

4. **Two learner seats don't double the statistical power.** Their outcomes
   are correlated through shared weights and through the fact that only one
   seat at a time can be declarer. Treat 4 comparisons per game as closer to
   ~2 effective independent samples when sizing iterations.

5. **If you want Elo to be a reliable ranking signal rather than a
   scheduling signal**, either:
   - set `elo_outplace_unit_weight: 1` and live with slow convergence, or
   - keep the higher weight but require more games per iteration (the
     variance formula above scales as `1/√N`), or
   - lean on the greedy-eval batch (`elo_use_greedy_eval_only: true`,
     `elo_eval_games: 2000`) which already trades iteration count for
     sample stability.

6. **Calibration check.** If you see steady drift of learner Elo against a
   frozen equal-strength bot over many iterations, the metric is working
   correctly only when the *cumulative* outplace rate converges to the
   expected equilibrium (~0.5 at any session size, since draws are
   excluded). A short-run rate far from that baseline is noise, not
   signal.

---

## 7. Open questions / future work

- Per-entry K tapering (`k_factor / sqrt(entry.games_played)`) would damp
  variance as a pairing matures. Noted but not implemented in
  `update_league_elo.py`.
- The equal-strength derivation above assumes uniform declarer rotation and
  a 50% declarer win rate. Real tarok deviates: declarer win rate depends
  on bid type and hand distribution, and Klop hands invalidate the "three
  zeros per game" assumption. A measurement of the empirical
  equilibrium outplace rate against a known-equal bot would refine these
  numbers.
