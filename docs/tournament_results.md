# Tournament Results — RL Model Comparison

Two round-robin tournaments were run to evaluate RL model performance.
Each tournament consists of 20 rounds with 8 players. Models are ranked by
average placement (lower is better).

**Column legend:**

| Column | Meaning |
|--------|---------|
| Avg | Average placement across all rounds |
| 1st | Number of rounds the model finished 1st |
| Top 2 | Number of rounds the model finished 1st or 2nd |
| Top Half | Number of rounds the model finished in top 4 (out of 8) |

---

## Tournament 1 (partial — ranks 6-8 incomplete)

| Rank | Model           | Type   | Avg  | 1st | Top 2 | Top Half | Rounds |
|------|-----------------|--------|------|-----|-------|----------|--------|
| 1    | ula bizjak 31   | rl     | 2.60 |   4 |    12 |       19 |     20 |
| 2    | ula bizjak 35   | rl     | 3.50 |   4 |     8 |       12 |     20 |
| 3    | ule bizjak 38   | rl     | 3.65 |   6 |     8 |       13 |     20 |
| 4    | anja korosec 7  | rl     | 3.70 |   4 |     6 |       12 |     20 |
| 5    | ula bizjak 13   | rl     | 4.15 |   2 |     4 |       10 |     20 |

> Ranks 6-8 were not recorded for this tournament.

## Tournament 2

| Rank | Model           | Type   | Avg  | 1st | Top 2 | Top Half | Rounds |
|------|-----------------|--------|------|-----|-------|----------|--------|
| 1    | ule bizjak 38   | rl     | 2.75 |   5 |     9 |       18 |     20 |
| 2    | ula bizjak 13   | rl     | 3.15 |   5 |     8 |       15 |     20 |
| 3    | ula bizjak 31   | rl     | 3.25 |   4 |     9 |       13 |     20 |
| 4    | ula bizjak 35   | rl     | 3.55 |   3 |     6 |       15 |     20 |
| 5    | anja korosec 7  | rl     | 4.40 |   2 |     4 |       10 |     20 |
| 6    | ema mlakar 316  | rl     | 5.00 |   0 |     3 |        7 |     20 |
| 7    | petra vidmar 2  | rl     | 6.00 |   1 |     1 |        2 |     20 |
| 8    | Random-7        | random | 7.90 |   0 |     0 |        0 |     20 |

---

## Summary

### Combined Average Placement (across both tournaments)

| Model           | T1 Avg | T2 Avg | Combined Avg |
|-----------------|--------|--------|--------------|
| ule bizjak 38   |   3.65 |   2.75 |         3.20 |
| ula bizjak 31   |   2.60 |   3.25 |         2.93 |
| ula bizjak 35   |   3.50 |   3.55 |         3.53 |
| ula bizjak 13   |   4.15 |   3.15 |         3.65 |
| anja korosec 7  |   3.70 |   4.40 |         4.05 |

### Key Observations

1. **ula bizjak 31** has the best combined average (2.93) — strongest overall
   model, winning Tournament 1 and placing 3rd in Tournament 2.
2. **ule bizjak 38** was the most improved between tournaments, winning T2
   with the best single-tournament average (2.75). Highest 1st-place count
   across both tournaments (6 + 5 = 11).
3. **ula bizjak 13** showed large variance — 5th in T1 but 2nd in T2.
4. **anja korosec 7** is consistently mid-tier (4th in T1, 5th in T2).
5. **ula bizjak 35** is consistently solid (2nd in T1, 4th in T2).
6. **Random-7** baseline finishes last in T2 with avg 7.90 (near worst
   possible 8.0), confirming all RL models learned meaningful play.
7. **ema mlakar 316** and **petra vidmar 2** underperform — candidates for
   retraining or removal from the model pool.

### Top 3 Models (recommended for further training / deployment)

1. **ula bizjak 31** — best combined average
2. **ule bizjak 38** — best single-tournament performance, most 1st places
3. **ula bizjak 35** — consistent top performer
