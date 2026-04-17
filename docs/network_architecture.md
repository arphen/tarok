# Tarok AI: Neural Network Architecture (v2)

The Artificial Intelligence powering our reinforcement learning agent utilizes a custom, multi-headed **Actor-Critic** neural network (`TarokNet`). It is specifically engineered to handle the multi-phase discrete decision spaces of Slovenian Tarok and supports the **Perfect Training, Imperfect Execution (PTIE)** paradigm.

## 1. Core Architectural Paradigms

### Perfect Training, Imperfect Execution (PTIE) / Oracle Critic
Unlike games of perfect information (like Chess), Tarok hides the vast majority of its state (opponents' hands and the talon). 
In our architecture, the **Actor** (the part of the network making the decisions) is fed *strictly restricted, imperfect information*—exactly what a human player would see at the table. 
However, when the network trains using Proximal Policy Optimization (PPO), we utilize an **Oracle Critic**. The Critic network evaluates the current state strictly to estimate its "Value"—providing the Actor with a baseline advantage rating. By feeding the Critic *perfect information* (the hands of all other players and the contents of the talon), we massively reduce the value estimation variance, stabilizing policy gradients and accelerating training.

*(Note: During deployment, the Oracle Critic subgraph is fully discarded and only the Actor subgraphs execute.)*

### Oracle Guiding (v2)
Beyond standard PTIE, we implement **Oracle Guiding** — an auxiliary distillation loss that aligns the actor's latent representation with the oracle critic's latent space during training. By minimizing cosine distance between the actor and critic hidden states, the actor learns to implicitly approximate the hidden information (opponent hands) purely from publicly observable features. This provides an artificial "theory of mind" that dramatically improves card play decisions.

### Multi-Decision Heads
A standard RL environment has a single homogeneous action space. In Tarok, a player makes completely different *types* of decisions depending on the game phase (bidding, exchanging talon cards, playing tricks). Rather than forcing these mutually exclusive actions into one massive 83-dimensional flat vector that the network struggles to untangle, our Actor utilizes a **Shared Latent Representation (Backbone)** that branches out into **five specialized decision heads**. 

---

## 2. Network Topology & Dimensions

### Inputs (State Encoding v2)
- **Actor State (`STATE_SIZE=450`):** Represents imperfect information. Includes the player's current hand, historical plays, bid history, trick counts, announcements, **opponent belief probability vectors** (3×54 = 162 dims), **opponent card-play statistics** (3×4 = 12 dims), and **trick context features** (6 dims).
- **Oracle State (`ORACLE_STATE_SIZE=612`):** During training, the perfect-information Critic receives the base 450 dimensions *plus* 162 additional dimensions (3 opponents $\times$ 54 possible cards) that reveal exactly where every unplayed card is located.

### v2 Belief Tracking (Public Belief State)
The v2 encoding adds **Bayesian belief vectors** for each opponent:
- Each of the 3 opponents gets a 54-dimensional probability vector estimating the likelihood they hold each card
- Uniform prior ($\frac{1}{3}$ for unknown cards) with **void-suit inference**: when an opponent fails to follow suit during trick play, their belief probability for all cards of that suit drops to 0
- This gives the network an explicit "theory of mind" channel without requiring recurrent memory

### The Actor Backbone (v2)
All observations first pass through a shared sequence of layers, followed by **residual blocks** and **card-level multi-head self-attention**:

**Input Projection:**
1. `Linear(450, 256)` → `LayerNorm(256)` → `ReLU`
2. `Linear(256, 256)` → `LayerNorm(256)` → `ReLU`

**Residual Blocks (×2):**
Each block: `LayerNorm(256)` → `Linear(256, 256)` → `ReLU` → `Linear(256, 256)` → `Add(residual)`

**Card-Level Self-Attention:**
- Extracts per-card features (hand, played, trick, 3× belief = 6 channels per card)
- 54 card tokens → `Linear(6, 64)` → `MultiheadAttention(64, 4 heads)` → `LayerNorm` → `MeanPool` → `Linear(64, 64)`
- Captures card-card relationships (suit correlations, void inference patterns)

**Fusion:**
`Concat(backbone_256, attention_64)` → `Linear(320, 256)` → `LayerNorm(256)` → `ReLU`

### The Critic Backbone (Oracle Enabled)
A secondary, entirely distinct backbone is instantiated for the Critic:
1. `Linear(612, 256)` → `LayerNorm(256)` → `ReLU`
2. `Linear(256, 256)` → `LayerNorm(256)` → `ReLU`
3. `ResidualBlock(256)` (×1)

### Specialized Output Heads
The fused 256-dimensional output is routed to the currently active decision head based on the game's phase. Every head passes the shared state through an independent intermediate projection layer (`Linear(256, 128) -> ReLU`) before mapping to its specific discrete output size:

- **Bidding Head (`BID_ACTION_SIZE=9`):** 
  Outputs probabilities for *Passing* or declaring one of the 8 legal contracts (e.g., Three, Two, One, Solo, Berač).
- **King Call Head (`KING_ACTION_SIZE=4`):**
  When playing a contract that allows calling a partner, outputs probabilities for which suit's King to call (Spades, Clubs, Hearts, Diamonds).
- **Talon Pick Head (`TALON_ACTION_SIZE=6`):**
  If exchanging cards with the talon, decides which group to take (up to 6 groups of 1 card for "Contract One").
- **Announcement Head (`ANNOUNCE_ACTION_SIZE=10`):**
  Decides whether to pass, make one of the 4 standard point/bonus announcements (e.g., Trula, Kings, Pagat Ultimo), or declare one of 5 counter-announcements (Kontras).
- **Card Play Head (`CARD_ACTION_SIZE=54`):**
  The actual trick-playing output. Produces a distribution across the entire 54-card deck. *(Note: Action masking is heavily applied post-network output to zero-out illegal plays like suit-revokes or playing cards not in hand).*

### The Critic Head
The Critic backbone's 256-dimensional output is routed to a shared Value Head (`Linear(256, 128) -> ReLU -> Linear(128, 1)`), which outputs a single scalar value estimating the final session score from the current state.

---

## 3. v2 Training Enhancements

### Oracle Guiding Loss
During PPO updates, an auxiliary distillation loss aligns the actor's 256-dim latent vector with the oracle critic's 256-dim latent vector:

$$\mathcal{L}_{\text{guide}} = 1 - \text{CosineSimilarity}(f_{\text{actor}}(s), f_{\text{critic}}(s_{\text{oracle}}))$$

This is weighted by `oracle_guiding_coef` (default 0.1) and added to the standard PPO loss.

### Parametric Monte Carlo Policy Adaptation (pMCPA)
Before each hand, the agent can optionally run rapid self-play rollouts with its actual dealt cards to fine-tune a temporary copy of the network. The adapted network is used for that single hand, then discarded. This extracts maximum expected value from the specific starting hand.
