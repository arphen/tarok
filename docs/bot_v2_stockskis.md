# StockŠkis v2: Improved Heuristic Player

StockŠkis v2 represents a significant evolution over the foundational v1 heuristic bot. It introduces several advanced features to make the agent a substantially more capable player, particularly focusing on memory, positioning, and team coordination. 

## Key Enhancements over v1

1. **Card Counting:**
   Unlike the base version, v2 actively tracks played cards throughout the hand. This allows it to reason accurately about remaining threats, such as outstanding high taroks or missing kings.

2. **Positional Awareness:**
   The bot tailors its play based on its seat position in the current trick. For example, it plays distinctly differently when acting in 2nd seat (where it must be cautious of trailing players) versus 4th seat (where it knows the exact outcome of the trick).

3. **Void-Building & Talon Selection:**
   During the talon exchange and discarding phase, v2 makes smarter choices explicitly to create "voids" (empty suits) in its hand. This increases its ability to ruff (trump) suit leads later in the game.

4. **Partner Signaling & Protection:**
   The bot incorporates "šmiranje" (point-feeding) logic. When it identifies its partner, it will actively try to feed them high-value cards (like Kings or high Taroks) if the partner is guaranteed to win the trick. It also plays defensively to protect its partner's vulnerable cards.

5. **Endgame Play:**
   As the hand approaches the final tricks, v2 counts the exact remaining taroks and suits to optimize its strategy for winning the last trick (especially relevant for the Pagat ultimo).

6. **Improved Bidding:**
   The bidding heuristic now actively considers suit distribution, correctly valuing hands with singletons and voids higher for assertive contracts.

7. **Better Klop Play:**
   In the negative contract *Klop* (where the goal is not to take points), v2 ducks tricks far more aggressively. It actively tracks which opponents are dangerous and forces them to take tricks.
