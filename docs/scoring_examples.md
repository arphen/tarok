# Scoring Examples

This document walks through concrete numerical examples for every major contract
type. For each game two scores are shown side-by-side:

| Column | Meaning |
|---|---|
| **Leaderboard** | `score_game()` — the locked display/arena score; opponents always receive 0. |
| **Reward** | `score_game_reward()` — the RL training signal; defenders receive a non-zero signal so the agent learns to defend. |

Seats are labelled **D** (declarer), **P** (partner, 2v2 only), **O1/O2**
(opponents / defenders).

---

## Contract base values (reference)

| Contract | Base |
|---|---|
| Three | 10 |
| Two | 20 |
| One | 30 |
| Solo Three | 40 |
| Solo Two | 50 |
| Solo One | 60 |
| Solo | 80 |
| Berač | 70 |
| Barvni Valat | 125 |

Silent bonuses: Trula 10, Kings 10, Pagat Ultimo 25, Valat 250.
Announced bonuses: Trula 20, Kings 20, Pagat Ultimo 50, Valat 500.
Kontra doubles the affected component.

---

## 1. Three (2 v 2) — declarer wins, no bonuses

**Setup:** contract = Three (base 10). Declarer team collects 46 card-points
(needs > 35). No bonuses. No kontra.

```
point_diff = |46 − 35| = 11
sign        = +1 (won)
total_declarer = sign × (base + point_diff) = +(10 + 11) = 21
partner_total  = total_declarer − contract_base = 21 − 10 = 11
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | +21 | +21 |
| P (partner) | +11 | +11 |
| O1 (opponent) | 0 | −11 *(= −partner_total)* |
| O2 (opponent) | 0 | −11 |

> Defenders receive `−partner_total` when the declarer wins.

---

## 2. Three (2 v 2) — declarer loses, no bonuses

**Setup:** same contract. Declarer team collects only 28 card-points.

```
point_diff = |28 − 35| = 7
sign        = −1 (lost)
total_declarer = −(10 + 7) = −17
partner_total  = −17 − (−10) = −7
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | −17 | −17 |
| P (partner) | −7 | −7 |
| O1 (opponent) | 0 | +17 *(= −total_declarer)* |
| O2 (opponent) | 0 | +17 |

> Defenders receive `−total_declarer` when the declarer loses.

---

## 3. Solo — declarer wins, silent Trula bonus

**Setup:** contract = Solo (base 80). Declarer is the only member of the
declarer team (solo contract). Declarer collects 44 card-points. Declarer team
captures all three taroks for the Trula (Mond, Skis, Pagat) → silent Trula +10.

```
point_diff = |44 − 35| = 9
sign        = +1
total_declarer = +(80 + 9) + 10 = 99
partner_total  = 99 − 80 = 19    (no partner on team; field unused in leaderboard)
```

In a Solo contract there is no partner seat — only the declarer and three
opponents. The `partner_total` field is internal; leaderboard and reward both
use `total_declarer` for the single declarer seat.

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | +99 | +99 |
| O1 | 0 | −19 *(= −partner_total)* |
| O2 | 0 | −19 |
| O3 | 0 | −19 |

---

## 4. Solo — declarer loses, announced Pagat Ultimo fails

**Setup:** contract = Solo (base 80). Declarer announces Pagat Ultimo (+50 if
successful, −50 if not). The declarer collects 30 card-points (loses) and also
fails the Pagat Ultimo.

```
point_diff         = |30 − 35| = 5
sign               = −1 (lost)
base_score         = −(80 + 5) = −85
pagat_ultimo_bonus = −50 (announced, failed)
total_declarer     = −85 + (−50) = −135
partner_total      = −135 − (−80) = −55
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | −135 | −135 |
| O1 | 0 | +135 *(= −total_declarer)* |
| O2 | 0 | +135 |
| O3 | 0 | +135 |

---

## 5. Two (2 v 2) — declarer wins with kontra on the game

**Setup:** contract = Two (base 20). Opponents call kontra on the game
(multiplier = 2). Declarer team collects 40 card-points. No bonuses.

```
point_diff     = |40 − 35| = 5
sign           = +1
km_game        = 2
contract_base  = +20 × 2 = 40
point_diff_sc  = +5 × 2 = 10
total_declarer = 40 + 10 = 50
partner_total  = 50 − 40 = 10
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | +50 | +50 |
| P (partner) | +10 | +10 |
| O1 | 0 | −10 |
| O2 | 0 | −10 |

---

## 6. One (2 v 2) — declarer wins, silent Kings bonus, opponents have Trula

**Setup:** contract = One (base 30). Declarer team collects 38 card-points.
Declarer team captures all four kings (silent Kings +10). Opponent team captures
the Trula (silent Trula −10 from declarer's perspective). No kontra.

```
point_diff     = |38 − 35| = 3
sign           = +1
base_score     = +(30 + 3) = 33
kings_bonus    = +10 (declarer team has all kings)
trula_bonus    = −10 (opponent team has trula)
total_declarer = 33 + 10 − 10 = 33
partner_total  = 33 − 30 = 3
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | +33 | +33 |
| P (partner) | +3 | +3 |
| O1 | 0 | −3 |
| O2 | 0 | −3 |

---

## 7. Berač — declarer wins (takes no tricks)

**Setup:** contract = Berač (base 70). Declarer takes zero tricks.

```
total_declarer = +70
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | +70 | +70 |
| O1 | 0 | −70 |
| O2 | 0 | −70 |
| O3 | 0 | −70 |

---

## 8. Berač — declarer fails (takes at least one trick)

**Setup:** same contract; declarer accidentally wins a trick.

```
total_declarer = −70
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | −70 | −70 |
| O1 | 0 | +70 |
| O2 | 0 | +70 |
| O3 | 0 | +70 |

---

## 9. Valat achieved silently (2 v 2)

**Setup:** contract = Three. Declarer team wins *every* trick → silent Valat
(250). When valat is achieved it **replaces all other scoring** — the
contract base and point-diff are discarded; only the valat magnitude counts.
Both declarer and partner receive the full 250.

```
total_declarer = +250
partner_total  = +250   (valat: contract_base zeroed, partner gets full value)
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | +250 | +250 |
| P (partner) | +250 | +250 |
| O1 | 0 | −250 |
| O2 | 0 | −250 |

---

## 10. Valat announced but failed (2 v 2)

**Setup:** contract = Two. Declarer team announces Valat (+500 if successful).
They fail to win every trick.

```
valat_failed_bonus = −500 (announced valat, failed)
point_diff = some value, e.g. 44 card-points → diff = 9, sign = +1
base_score = +(20 + 9) = 29
total_declarer = 29 + (−500) = −471
partner_total  = −471 − 20 = −491
```

| Seat | Leaderboard | Reward |
|---|---|---|
| D (declarer) | −471 | −471 |
| P (partner) | −491 | −491 |
| O1 | 0 | +471 |
| O2 | 0 | +471 |

---

## 11. Klop — per-player scoring

**Setup:** Every player scores individually. No declarer team.
Card-points by seat after the hand: P0=32, P1=8, P2=18, P3=12.

Rules:
- **> 35 points** → score = −70 (all game points, as penalty)
- **0 tricks taken** → score = +70
- **Otherwise** → score = −(own card-points)

```
P0: 32 points, ≤ 35 and took tricks → score = −32
P1:  8 points, ≤ 35 and took tricks → score = −8
P2: 18 points, ≤ 35 and took tricks → score = −18
P3: 12 points, ≤ 35 and took tricks → score = −12
```

| Seat | Leaderboard | Reward |
|---|---|---|
| P0 | −32 | −32 |
| P1 | −8 | −8 |
| P2 | −18 | −18 |
| P3 | −12 | −12 |

> Klop reward = leaderboard score (each player already has a meaningful
> individual signal; no override needed).

**Klop: P1 sweeps (takes no tricks):**

```
P0: some tricks → −points
P1: 0 tricks    → +70
P2: some tricks → −points
P3: some tricks → −points
```

---

## Summary: reward-signal formula

For all normal contracts (non-Klop):

| Seat | Reward |
|---|---|
| Declarer | `sign × (base + point_diff + bonuses) × km` |
| Partner (2v2) | `sign × (point_diff + bonuses) × km` (no contract base) |
| Each defender — declarer won | `−partner_total` |
| Each defender — declarer lost | `−total_declarer` |

When **valat** is achieved, `contract_base` is zeroed and both declarer and
partner receive the full valat magnitude; defenders receive `−total_declarer`
in both the won and lost branches (since `partner_total = total_declarer` when
`contract_base = 0`).
