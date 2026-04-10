# StockŠkis v4: Specialized Refinements

StockŠkis v4 acts as a specialized refinement built on top of the powerful v3 architecture. Instead of introducing massive new architectural systems, v4 applies strict rules and gating to address specific edge cases where v3's probabilistic logic occasionally faltered, particularly surrounding the *Berač* (Beggar) contract and opening leads.

## Key Enhancements over v3

1. **Tighter Berač Gating:**
   The heuristic for bidding *Berač* (where the declarer must take zero tricks) has been rigidly clamped down. 
   - v4 will **never** bid Berač if it holds more than two taroks.
   - v4 will **never** bid Berač if it has any singleton suit (a suit with exactly one card), as this represents an unacceptable risk of being forced to play a high card or being thrown into the lead.

2. **Clearer Lead Roles and Team Coordination:**
   The logic for the opening lead of a trick has been highly specialized based on the player's allegiance:
   - **As the Declarer's Partner:** If the bot recognizes itself as the declarer's partner, it will strongly prefer to open the trick with its *highest tarok*. This "clears the board" of enemy taroks, effectively pulling them out and protecting the declarer's power cards.
   - **As the Opposition:** If it is defending against the declarer, it plays to preserve its team's taroks. It will prioritize leading low-value suit cards to force the declarer's team to spend their taroks to win the trick, actively draining the declarer's trump control.
