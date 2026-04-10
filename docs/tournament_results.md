# Tournament Results — RL Model Comparison

Three round-robin tournaments were run to evaluate RL model performance.
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

## Tournament 3

| Rank | Model             | Type   | Avg  | 1st | Top 2 | Top Half | Rounds |
|------|-------------------|--------|------|-----|-------|----------|--------|
| 1    | katja vidmar 52   | rl     | 2.50 |   7 |    10 |       20 |     20 |
| 2    | katja vidmar 39   | rl     | 2.90 |   3 |     8 |       17 |     20 |
| 3    | ula bizjak 38     | rl     | 3.05 |   3 |     7 |       16 |     20 |
| 4    | anja korosec      | rl     | 3.40 |   2 |     7 |       13 |     20 |
| 5    | ula vodmar 35     | rl     | 3.40 |   5 |     8 |       12 |     20 |
| 6    | Random-7          | random | 6.80 |   0 |     0 |        0 |     20 |
| 7    | Random-6          | random | 6.95 |   0 |     0 |        1 |     20 |
| 8    | Random-5          | random | 7.00 |   0 |     0 |        1 |     20 |

---

## Summary

### Combined Average Placement (across all tournaments)

| Model             | T1 Avg | T2 Avg | T3 Avg | Combined Avg |
|-------------------|--------|--------|--------|--------------|
| katja vidmar 52   |      — |      — |   2.50 |         2.50 |
| katja vidmar 39   |      — |      — |   2.90 |         2.90 |
| ula bizjak 31     |   2.60 |   3.25 |      — |         2.93 |
| ula bizjak 38     |      — |      — |   3.05 |         3.05 |
| ule bizjak 38     |   3.65 |   2.75 |      — |         3.20 |
| anja korosec      |      — |      — |   3.40 |         3.40 |
| ula bizjak 35     |   3.50 |   3.55 |      — |         3.53 |
| ula bizjak 13     |   4.15 |   3.15 |      — |         3.65 |
| anja korosec 7    |   3.70 |   4.40 |      — |         4.05 |

### Key Observations

1. **katja vidmar 52** dominates Tournament 3 with avg 2.50 — best single-
   tournament score across all 3 tournaments. Finished in top 4 every round
   (20/20), with 7 first-place finishes.
2. **katja vidmar 39** places 2nd in T3 (avg 2.90), confirming the Katja Vidmar
   lineage as the strongest model family.
3. **ula bizjak 31** remains the best performer from the original tournaments
   (combined avg 2.93 across T1+T2).
4. **ula bizjak 38** (T3 rank 3, avg 3.05) and **ule bizjak 38** (T1+T2 avg 3.20)
   appear to be related models with consistent top performance.
5. All RL models decisively beat the Random baselines (avg 6.80-7.00).

### Top 3 Models (recommended for further training / deployment)

1. **katja vidmar 52** — best single-tournament average (2.50), perfect top-half rate
2. **katja vidmar 39** — strong 2nd place, consistent top-4 finisher (17/20)
3. **ula bizjak 31** — best combined average across T1+T2 (2.93)
