# StockŠkis v3: Strongest Heuristic Player

StockŠkis v3 builds directly upon the improvements established in v2, acting as the strongest pure-heuristic agent in the training lab. This version transitions from observable-only tracking to probabilistic reasoning and endgame optimization.

## Key Enhancements over v2

1. **Inference About Hidden Cards:**
   v3 introduces a Bayesian-style reasoning layer. It does not just track what has been played, but attempts to infer what cards opponents *likely* hold. It deduces these probabilities based on their bidding behavior and the specific cards they have chosen to play in previous tricks.

2. **Trick Counting and Dynamic Adjustment:**
   It actively tracks how many tricks each team has won. Using this running score, it dynamically adjusts its strategy—playing more conservatively to secure a narrow win, or taking risks if it needs to catch up.

3. **Endgame Solver:**
   When the remaining cards per player drop below a certain threshold, the heuristic evaluation is entirely bypassed in favor of an exhaustive endgame solver. This allows v3 to play perfectly in the final few tricks by evaluating all possible remaining play sequences.

4. **Lead Selection Matrix:**
   Rather than using uniform heuristics for leading a trick, v3 utilizes a phase-dependent matrix. It selects its lead based on the current stage of the game (early, mid, or late) and its position within its team structure (declarer vs. defender).

5. **Defensive Signaling:**
   When stuck on defense against the declarer's team, v3 coordinates its plays to silently communicate suit lengths and preferences to its partner, mimicking high-level human defender play.

6. **Counter-Play in Negative Contracts:**
   The logic for *Klop* and *Berač* (Beggar) is completely overhauled. v3 is capable of ruthless counter-play, actively forcing the player attempting to avoid tricks to take them by leading their weak suits or stripping their exits.
