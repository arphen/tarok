# Tarok AI Project Codebase

## Context Information for LLM
This is a codebase for a Tarok card game AI. The system comprises a Python backend featuring a Reinforcement Learning (PPO) agent that learns through self-play. It evaluates the agent's performance dynamically, splitting stats into Declarer vs. Defender win rates. The backend uses a single-core asynchronous Python event loop combining the CPU-heavy PPO simulation steps with a FastAPI web server (yielding control via `asyncio.sleep` to stream real-time metrics). 
The frontend is a React/Vite application that provides a real-time training dashboard with live charts to monitor the AI's learning progress.

## Research Questions
Please do a deep research pass over this codebase and answer the following questions:
1. **Rule correctness**: Are the Tarok game rules implemented correctly in this codebase (e.g., card ranking, bidding phases, legal move constraints, and trick scoring)?
2. **Modern academic improvements**: What are the most recent academic improvements for self-play learning in trick-taking or imperfect-information card games that could be relevant to our learning here?
3. **Applying recent methods**: How exactly can we improve the agent's learning performance, training stability, and policy convergence using these recent methods?
4. **State & Reward shaping**: How can we optimize our state representation (observations) and reward shaping to better capture Tarok's hidden information dynamics?
5. **Architectural improvements**: Are there any critical flaws or bottlenecks in how the PPO algorithm is currently integrated with the Tarok game loop, and how can we safely scale this architecture?

---
## Codebase

### File: `/Users/swozny/work/tarok/backend/src/tarok/__init__.py`
```python

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/__main__.py`
```python
"""CLI entry point — run the server or training."""

import asyncio
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        num_sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        games_per_session = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        asyncio.run(run_training(num_sessions, games_per_session))
    else:
        run_server()


def run_server():
    import uvicorn
    uvicorn.run(
        "tarok.adapters.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


async def run_training(num_sessions: int, games_per_session: int):
    import os
    import torch
    from tarok.adapters.ai.agent import RLAgent
    from tarok.adapters.ai.trainer import PPOTrainer

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    total_games = num_sessions * games_per_session
    print(f"🧠 Starting session-based training: {num_sessions} sessions × {games_per_session} games = {total_games} games")
    print(f"   Device: {device}")
    print(f"   Agent learns bidding, king calling, talon selection, announcements, and card play via PPO")
    print()

    agents = [RLAgent(name=f"Agent-{i}", device=device) for i in range(4)]
    trainer = PPOTrainer(agents, device=device, games_per_session=games_per_session)

    async def print_metrics(metrics):
        bar_len = 30
        filled = int(bar_len * metrics.episode / metrics.total_episodes)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r[{bar}] S{metrics.session}/{metrics.total_sessions} G{metrics.episode}/{metrics.total_episodes} | "
            f"Win: {metrics.win_rate:.1%} | "
            f"Reward: {metrics.avg_reward:+.2f} | "
            f"Bid: {metrics.bid_rate:.0%} | "
            f"Klop: {metrics.klop_rate:.0%} | "
            f"Solo: {metrics.solo_rate:.0%} | "
            f"{metrics.games_per_second:.1f} g/s",
            end="",
            flush=True,
        )

    trainer.add_metrics_callback(print_metrics)
    result = await trainer.train(num_sessions)

    print()
    print()
    print("✅ Training complete!")
    print(f"   Final win rate: {result.win_rate:.1%}")
    print(f"   Final avg reward: {result.avg_reward:+.2f}")
    print(f"   Throughput: {result.games_per_second:.1f} games/sec")
    print()

    # Per-contract breakdown
    d = result.to_dict()
    print("   Contract breakdown (as declarer → as defender):")
    print(f"   {'Contract':<14} {'Decl':>5} {'D.Win%':>7} {'D.Avg':>7} │ {'Def':>5} {'Def.Win%':>8} {'Def.Avg':>8}")
    print(f"   {'─'*14} {'─'*5} {'─'*7} {'─'*7} │ {'─'*5} {'─'*8} {'─'*8}")
    for cname, cs in d["contract_stats"].items():
        if cs["played"] > 0:
            dp = cs["decl_played"]
            dw = f"{cs['decl_win_rate']:.0%}" if dp > 0 else "—"
            da = f"{cs['decl_avg_score']:+.1f}" if dp > 0 else "—"
            fp = cs["def_played"]
            fw = f"{cs['def_win_rate']:.0%}" if fp > 0 else "—"
            fa = f"{cs['def_avg_score']:+.1f}" if fp > 0 else "—"
            print(f"   {cname:<14} {dp:>5} {dw:>7} {da:>7} │ {fp:>5} {fw:>8} {fa:>8}")
    print()
    # Per-session avg score trend
    scores_hist = result.session_avg_score_history
    if scores_hist:
        first5 = sum(scores_hist[:5]) / min(5, len(scores_hist))
        last5 = sum(scores_hist[-5:]) / min(5, len(scores_hist))
        print(f"   Session avg score: first 5 sessions={first5:+.1f} → last 5 sessions={last5:+.1f}")
        print()
    print(f"   Snapshots saved: {len(result.snapshots)}")
    print(f"   Checkpoint: checkpoints/tarok_agent_latest.pt")


if __name__ == "__main__":
    main()

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/__init__.py`
```python

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/ai/__init__.py`
```python

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/ai/agent.py`
```python
"""RL Agent adapter — implements PlayerPort using a neural network with PPO.

All decisions (bid, king call, talon pick, card play) go through the network.
Discard remains heuristic (combinatorial action space too large for a single head).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from tarok.entities.card import Card, CardType, Suit
from tarok.entities.game_state import Announcement, Contract, GameState, KontraLevel
from tarok.adapters.ai.network import TarokNet
from tarok.adapters.ai.encoding import (
    DecisionType,
    encode_state,
    encode_legal_mask,
    encode_bid_mask,
    encode_king_mask,
    encode_talon_mask,
    encode_announce_mask,
    card_idx_to_card,
    CARD_TO_IDX,
    BID_ACTIONS,
    BID_TO_IDX,
    KING_ACTIONS,
    SUIT_TO_IDX,
    ANNOUNCE_PASS,
    ANNOUNCE_IDX_TO_ANN,
    KONTRA_IDX_TO_KEY,
)


@dataclass
class Experience:
    """A single step of experience for PPO training."""
    state: torch.Tensor
    action: int
    log_prob: torch.Tensor
    value: torch.Tensor
    decision_type: DecisionType = DecisionType.CARD_PLAY
    reward: float = 0.0
    done: bool = False


class RLAgent:
    """Deep RL agent that learns to play Tarok via self-play.

    All game decisions (bidding, king calling, talon selection, card play)
    are learned through PPO.  The agent maximises cumulative score across
    a session of many games, which naturally teaches it when to bid high,
    when to pass, and when to call berac.
    """

    def __init__(
        self,
        name: str = "RL-Agent",
        hidden_size: int = 256,
        device: str = "cpu",
        explore_rate: float = 0.1,
    ):
        self._name = name
        self.device = torch.device(device)
        self.network = TarokNet(hidden_size).to(self.device)
        self.explore_rate = explore_rate
        self._rng = random.Random()

        # Experience buffer for current game
        self.experiences: list[Experience] = []
        self._training = True

    @property
    def name(self) -> str:
        return self._name

    def set_training(self, training: bool) -> None:
        self._training = training
        self.network.train(training)

    # ------------------------------------------------------------------
    # Generic decision helper
    # ------------------------------------------------------------------

    def _decide(
        self,
        state: GameState,
        player_idx: int,
        legal_mask: torch.Tensor,
        decision_type: DecisionType,
    ) -> int:
        """Run the network for *decision_type*, record experience, return action index."""
        state_tensor = encode_state(state, player_idx, decision_type).to(self.device)
        mask = legal_mask.to(self.device)

        # Epsilon-greedy exploration during training
        if self._training and self._rng.random() < self.explore_rate:
            legal_indices = mask.nonzero(as_tuple=True)[0].tolist()
            action_idx = self._rng.choice(legal_indices)
            with torch.no_grad():
                _, value = self.network(state_tensor.unsqueeze(0), decision_type)
            self.experiences.append(Experience(
                state=state_tensor,
                action=action_idx,
                log_prob=torch.tensor(0.0),
                value=value.squeeze(),
                decision_type=decision_type,
            ))
            return action_idx

        ctx = torch.no_grad() if not self._training else torch.enable_grad()
        with ctx:
            action_idx, log_prob, value = self.network.get_action(
                state_tensor.unsqueeze(0), mask.unsqueeze(0), decision_type,
            )

        if self._training:
            self.experiences.append(Experience(
                state=state_tensor.detach(),
                action=action_idx,
                log_prob=log_prob.detach(),
                value=value.detach(),
                decision_type=decision_type,
            ))

        return action_idx

    # ------------------------------------------------------------------
    # PlayerPort implementation
    # ------------------------------------------------------------------

    async def choose_bid(
        self, state: GameState, player_idx: int, legal_bids: list[Contract | None]
    ) -> Contract | None:
        mask = encode_bid_mask(legal_bids)
        action_idx = self._decide(state, player_idx, mask, DecisionType.BID)
        bid = BID_ACTIONS[action_idx]
        # Safety: make sure the decoded bid is actually legal
        if bid not in legal_bids:
            bid = None  # fall back to pass
        return bid

    async def choose_king(
        self, state: GameState, player_idx: int, callable_kings: list[Card]
    ) -> Card:
        mask = encode_king_mask(callable_kings)
        action_idx = self._decide(state, player_idx, mask, DecisionType.KING_CALL)
        chosen_suit = KING_ACTIONS[action_idx]
        # Find the king (or queen) matching that suit
        for king in callable_kings:
            if king.suit == chosen_suit:
                return king
        return callable_kings[0]  # safety fallback

    async def choose_talon_group(
        self, state: GameState, player_idx: int, talon_groups: list[list[Card]]
    ) -> int:
        mask = encode_talon_mask(len(talon_groups))
        action_idx = self._decide(state, player_idx, mask, DecisionType.TALON_PICK)
        if action_idx >= len(talon_groups):
            action_idx = 0  # safety fallback
        return action_idx

    async def choose_discard(
        self, state: GameState, player_idx: int, must_discard: int
    ) -> list[Card]:
        """Heuristic discard — kept out of RL due to combinatorial action space."""
        hand = state.hands[player_idx]
        discardable = [
            c for c in hand if not c.is_king and c.card_type != CardType.TAROK
        ]
        if len(discardable) < must_discard:
            discardable = [c for c in hand if not c.is_king]
        discardable.sort(key=lambda c: c.points)
        return discardable[:must_discard]

    async def choose_announcements(
        self, state: GameState, player_idx: int
    ) -> list[Announcement]:
        return []

    async def choose_announce_action(
        self, state: GameState, player_idx: int
    ) -> int:
        """Choose a single announcement action (0=pass, 1-4=announce, 5-9=kontra).

        Called repeatedly by the game loop until the player passes.
        """
        mask = encode_announce_mask(state, player_idx)
        # If only pass is legal, just return pass without going through the network
        if mask.sum().item() <= 1.0:
            return ANNOUNCE_PASS
        return self._decide(state, player_idx, mask, DecisionType.ANNOUNCE)

    async def choose_card(
        self, state: GameState, player_idx: int, legal_plays: list[Card]
    ) -> Card:
        mask = encode_legal_mask(legal_plays)
        action_idx = self._decide(state, player_idx, mask, DecisionType.CARD_PLAY)
        card = card_idx_to_card(action_idx)
        if card not in legal_plays:
            card = legal_plays[0]
        return card

    # ------------------------------------------------------------------
    # Experience management
    # ------------------------------------------------------------------

    def finalize_game(self, reward: float) -> None:
        """Set the reward for all experiences in this game."""
        for exp in self.experiences:
            exp.reward = reward
            exp.done = True

    def clear_experiences(self) -> None:
        self.experiences.clear()

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/ai/encoding.py`
```python
"""State encoding — converts game state into tensor representation for the neural network.

Supports multiple decision types: bidding, king calling, talon selection, and card play.
"""

from __future__ import annotations

from enum import Enum

import torch

from tarok.entities.card import Card, CardType, DECK, Suit
from tarok.entities.game_state import Announcement, Contract, GameState, KontraLevel, Phase, Team

# Build a card-to-index mapping
CARD_TO_IDX: dict[Card, int] = {card: idx for idx, card in enumerate(DECK)}


class DecisionType(Enum):
    BID = 0
    KING_CALL = 1
    TALON_PICK = 2
    CARD_PLAY = 3
    ANNOUNCE = 4


# Action space sizes per decision type
BID_ACTION_SIZE = 8       # pass + 7 biddable contracts
KING_ACTION_SIZE = 4      # one per suit
TALON_ACTION_SIZE = 6     # max talon groups (contract ONE → 6 groups of 1)
CARD_ACTION_SIZE = 54     # one per card in deck
ANNOUNCE_ACTION_SIZE = 10 # pass + 4 announcements + 5 kontras (game + 4 bonuses)

# Announcement action indices
ANNOUNCE_PASS = 0
ANNOUNCE_TRULA = 1
ANNOUNCE_KINGS = 2
ANNOUNCE_PAGAT = 3
ANNOUNCE_VALAT = 4
KONTRA_GAME = 5
KONTRA_TRULA = 6
KONTRA_KINGS = 7
KONTRA_PAGAT = 8
KONTRA_VALAT = 9

# Maps from action index to Announcement enum (for the announce actions)
ANNOUNCE_IDX_TO_ANN: dict[int, Announcement] = {
    ANNOUNCE_TRULA: Announcement.TRULA,
    ANNOUNCE_KINGS: Announcement.KINGS,
    ANNOUNCE_PAGAT: Announcement.PAGAT_ULTIMO,
    ANNOUNCE_VALAT: Announcement.VALAT,
}

# Maps from action index to kontra target key
KONTRA_IDX_TO_KEY: dict[int, str] = {
    KONTRA_GAME: "game",
    KONTRA_TRULA: Announcement.TRULA.value,
    KONTRA_KINGS: Announcement.KINGS.value,
    KONTRA_PAGAT: Announcement.PAGAT_ULTIMO.value,
    KONTRA_VALAT: Announcement.VALAT.value,
}

# Mapping from bid action index to Contract | None
_BIDDABLE_CONTRACTS = [c for c in Contract if c.is_biddable]
BID_ACTIONS: list[Contract | None] = [None] + _BIDDABLE_CONTRACTS  # [pass, THREE, TWO, ONE, S3, S2, S1, SOLO]

BID_TO_IDX: dict[Contract | None, int] = {action: idx for idx, action in enumerate(BID_ACTIONS)}

# King action index → suit
KING_ACTIONS: list[Suit] = list(Suit)  # [HEARTS, DIAMONDS, CLUBS, SPADES]
SUIT_TO_IDX: dict[Suit, int] = {s: i for i, s in enumerate(KING_ACTIONS)}


def encode_state(state: GameState, player_idx: int, decision_type: DecisionType = DecisionType.CARD_PLAY) -> torch.Tensor:
    """Encode visible game state for a specific player as a flat tensor.

    Expanded to include bidding context, talon visibility, hand strength,
    and the current decision type.
    """
    features: list[float] = []

    # Cards in hand (54 binary)
    hand_vec = [0.0] * 54
    for card in state.hands[player_idx]:
        hand_vec[CARD_TO_IDX[card]] = 1.0
    features.extend(hand_vec)

    # Cards already played (54 binary)
    played_vec = [0.0] * 54
    for trick in state.tricks:
        for _, card in trick.cards:
            played_vec[CARD_TO_IDX[card]] = 1.0
    features.extend(played_vec)

    # Cards in current trick (54 binary)
    trick_vec = [0.0] * 54
    if state.current_trick:
        for _, card in state.current_trick.cards:
            trick_vec[CARD_TO_IDX[card]] = 1.0
    features.extend(trick_vec)

    # Talon cards visible to this player (54 binary)
    talon_vec = [0.0] * 54
    if state.talon_revealed and player_idx == state.declarer:
        for group in state.talon_revealed:
            for card in group:
                talon_vec[CARD_TO_IDX[card]] = 1.0
    features.extend(talon_vec)

    # Player position relative to dealer (4 one-hot)
    pos_vec = [0.0] * 4
    relative_pos = (player_idx - state.dealer) % state.num_players
    pos_vec[relative_pos] = 1.0
    features.extend(pos_vec)

    # Contract (8 one-hot: none + KLOP + 7 biddable)
    contract_vec = [0.0] * 8
    if state.contract:
        contract_list = list(Contract)
        idx = contract_list.index(state.contract)
        if idx < 8:
            contract_vec[idx] = 1.0
    features.extend(contract_vec)

    # Phase encoding (3 features: bidding, trick_play, other)
    phase_vec = [0.0] * 3
    if state.phase == Phase.BIDDING:
        phase_vec[0] = 1.0
    elif state.phase == Phase.TRICK_PLAY:
        phase_vec[1] = 1.0
    else:
        phase_vec[2] = 1.0
    features.extend(phase_vec)

    # Partner known
    features.append(1.0 if state.is_partner_revealed else 0.0)

    # Tricks won by my team (normalized 0-1)
    my_team = state.get_team(player_idx)
    my_tricks = sum(
        1 for t in state.tricks if state.get_team(t.winner()) == my_team
    )
    features.append(my_tricks / 12.0)

    # Tricks played (normalized 0-1)
    features.append(state.tricks_played / 12.0)

    # Decision type (5 one-hot)
    dt_vec = [0.0] * 5
    dt_vec[decision_type.value] = 1.0
    features.extend(dt_vec)

    # Bidding context: highest bid so far (8 one-hot: no_bid + 7 contracts)
    bid_vec = [0.0] * 8
    bids_with_contract = [b.contract for b in state.bids if b.contract is not None]
    if bids_with_contract:
        highest = max(bids_with_contract, key=lambda c: c.strength)
        bid_vec[BID_TO_IDX.get(highest, 0)] = 1.0
    else:
        bid_vec[0] = 1.0  # No bid yet → index 0 (pass slot, meaning "nothing bid")
    features.extend(bid_vec)

    # Which players have passed (4 binary, relative to dealer)
    passed_set = {b.player for b in state.bids if b.contract is None}
    for i in range(state.num_players):
        p = (state.dealer + 1 + i) % state.num_players
        features.append(1.0 if p in passed_set else 0.0)

    # Hand strength features (normalized, always useful)
    hand = state.hands[player_idx]
    tarok_count = sum(1 for c in hand if c.card_type == CardType.TAROK)
    high_taroks = sum(1 for c in hand if c.card_type == CardType.TAROK and c.value >= 15)
    king_count = sum(1 for c in hand if c.is_king)
    suits_in_hand = {c.suit for c in hand if c.suit is not None}
    void_count = 4 - len(suits_in_hand)

    features.append(tarok_count / 12.0)
    features.append(high_taroks / 7.0)
    features.append(king_count / 4.0)
    features.append(void_count / 4.0)

    # Announcements made (4 binary: trula, kings, pagat, valat — by either team)
    ann_set: set[Announcement] = set()
    for anns in state.announcements.values():
        ann_set.update(anns)
    for ann in [Announcement.TRULA, Announcement.KINGS, Announcement.PAGAT_ULTIMO, Announcement.VALAT]:
        features.append(1.0 if ann in ann_set else 0.0)

    # Kontra levels (5 features, normalized: game + 4 bonuses, each 0/0.33/0.67/1)
    _KONTRA_KEYS = ["game", Announcement.TRULA.value, Announcement.KINGS.value,
                    Announcement.PAGAT_ULTIMO.value, Announcement.VALAT.value]
    for key in _KONTRA_KEYS:
        level = state.kontra_levels.get(key, KontraLevel.NONE)
        features.append((level.value - 1) / 7.0)  # 0→0, 1→0.14, 3→0.43, 7→1.0

    return torch.tensor(features, dtype=torch.float32)


# Compute STATE_SIZE from the feature layout
STATE_SIZE = (
    54 +  # hand
    54 +  # played
    54 +  # current_trick
    54 +  # talon_visible
    4 +   # player_position
    8 +   # contract
    3 +   # phase
    1 +   # partner_known
    1 +   # tricks_won_by_team
    1 +   # tricks_played
    5 +   # decision_type (now 5: bid, king, talon, card, announce)
    8 +   # highest_bid
    4 +   # passed_players
    4 +   # hand_strength
    4 +   # announcements_made
    5     # kontra_levels
)  # = 263


def encode_legal_mask(legal_cards: list[Card]) -> torch.Tensor:
    """Create a binary mask over the 54-card action space."""
    mask = torch.zeros(54, dtype=torch.float32)
    for card in legal_cards:
        mask[CARD_TO_IDX[card]] = 1.0
    return mask


def encode_bid_mask(legal_bids: list[Contract | None]) -> torch.Tensor:
    """Create a binary mask over the 8-bid action space."""
    mask = torch.zeros(BID_ACTION_SIZE, dtype=torch.float32)
    for bid in legal_bids:
        idx = BID_TO_IDX.get(bid)
        if idx is not None:
            mask[idx] = 1.0
    return mask


def encode_king_mask(callable_kings: list[Card]) -> torch.Tensor:
    """Create a binary mask over the 4-king action space (by suit)."""
    mask = torch.zeros(KING_ACTION_SIZE, dtype=torch.float32)
    for king in callable_kings:
        if king.suit is not None:
            mask[SUIT_TO_IDX[king.suit]] = 1.0
    return mask


def encode_talon_mask(num_groups: int) -> torch.Tensor:
    """Create a binary mask over the 6-talon-group action space."""
    mask = torch.zeros(TALON_ACTION_SIZE, dtype=torch.float32)
    for i in range(num_groups):
        mask[i] = 1.0
    return mask


def card_idx_to_card(idx: int) -> Card:
    return DECK[idx]


def encode_announce_mask(
    state: GameState,
    player_idx: int,
) -> torch.Tensor:
    """Create a binary mask over the 10-action announcement space.

    Actions: PASS(0), TRULA(1), KINGS(2), PAGAT(3), VALAT(4),
             K_GAME(5), K_TRULA(6), K_KINGS(7), K_PAGAT(8), K_VALAT(9)

    Rules:
    - Can always pass
    - Declarer team can announce bonuses they haven't already announced
    - For each announced bonus (or the base game), the other team may kontra
      if the current kontra level allows escalation and it's their turn
    """
    mask = torch.zeros(ANNOUNCE_ACTION_SIZE, dtype=torch.float32)
    mask[ANNOUNCE_PASS] = 1.0  # always legal

    player_team = state.get_team(player_idx)
    is_declarer_team = player_team == Team.DECLARER_TEAM

    # Already-announced set
    already_announced: set[Announcement] = set()
    for anns in state.announcements.values():
        already_announced.update(anns)

    # Declarer team can make new announcements
    if is_declarer_team:
        for action_idx, ann in ANNOUNCE_IDX_TO_ANN.items():
            if ann not in already_announced:
                mask[action_idx] = 1.0

    # Kontra escalation: check each target
    for action_idx, key in KONTRA_IDX_TO_KEY.items():
        # For bonus kontras, only allow if the bonus was announced
        if action_idx != KONTRA_GAME:
            # Map key back to Announcement to check if it was announced
            ann = next((a for a in Announcement if a.value == key), None)
            if ann is None or ann not in already_announced:
                continue

        level = state.kontra_levels.get(key, KontraLevel.NONE)
        next_level = level.next_level
        if next_level is None:
            continue  # Already at SUB, can't escalate

        # Check whose turn it is to escalate
        if level.is_opponent_turn and not is_declarer_team:
            mask[action_idx] = 1.0
        elif not level.is_opponent_turn and is_declarer_team:
            mask[action_idx] = 1.0

    return mask

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/ai/network.py`
```python
"""Neural network for the RL agent — multi-head actor-critic architecture.

Supports four decision types with separate action heads:
  - Bidding (8 actions: pass + 7 contracts)
  - King calling (4 actions: one per suit)
  - Talon group selection (6 actions)
  - Card play (54 actions: one per card)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tarok.adapters.ai.encoding import (
    STATE_SIZE,
    BID_ACTION_SIZE,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
    CARD_ACTION_SIZE,
    ANNOUNCE_ACTION_SIZE,
    DecisionType,
)


class TarokNet(nn.Module):
    """Multi-head Actor-Critic network for all Tarok decisions."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        half = hidden_size // 2

        # Actor heads — one per decision type
        self.bid_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, BID_ACTION_SIZE),
        )
        self.king_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, KING_ACTION_SIZE),
        )
        self.talon_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, TALON_ACTION_SIZE),
        )
        self.card_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, CARD_ACTION_SIZE),
        )
        self.announce_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, ANNOUNCE_ACTION_SIZE),
        )

        # Critic head — shared value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )

        self._heads = {
            DecisionType.BID: self.bid_head,
            DecisionType.KING_CALL: self.king_head,
            DecisionType.TALON_PICK: self.talon_head,
            DecisionType.CARD_PLAY: self.card_head,
            DecisionType.ANNOUNCE: self.announce_head,
        }

    def forward(
        self, state: torch.Tensor, decision_type: DecisionType = DecisionType.CARD_PLAY
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(state)
        logits = self._heads[decision_type](shared)
        value = self.critic(shared)
        return logits, value

    def get_action(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType = DecisionType.CARD_PLAY,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select an action from legal moves. Returns (action_idx, log_prob, value)."""
        logits, value = self(state, decision_type)
        # Mask illegal actions with -inf
        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = float("-inf")

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action).squeeze(), value.squeeze(-1).squeeze()

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType = DecisionType.CARD_PLAY,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a batch of actions. Returns (log_probs, values, entropy)."""
        logits, values = self(state, decision_type)
        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = float("-inf")

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/ai/random_agent.py`
```python
"""Random player adapter — baseline agent that makes random legal moves."""

from __future__ import annotations

import random

from tarok.entities.card import Card, CardType
from tarok.entities.game_state import Announcement, Contract, GameState


class RandomPlayer:
    def __init__(self, name: str = "Random", rng: random.Random | None = None):
        self._name = name
        self._rng = rng or random.Random()

    @property
    def name(self) -> str:
        return self._name

    async def choose_bid(
        self, state: GameState, player_idx: int, legal_bids: list[Contract | None]
    ) -> Contract | None:
        # Bias toward passing slightly
        if None in legal_bids and self._rng.random() < 0.6:
            return None
        return self._rng.choice(legal_bids)

    async def choose_king(
        self, state: GameState, player_idx: int, callable_kings: list[Card]
    ) -> Card:
        return self._rng.choice(callable_kings)

    async def choose_talon_group(
        self, state: GameState, player_idx: int, talon_groups: list[list[Card]]
    ) -> int:
        return self._rng.randint(0, len(talon_groups) - 1)

    async def choose_discard(
        self, state: GameState, player_idx: int, must_discard: int
    ) -> list[Card]:
        hand = state.hands[player_idx]
        # Discard lowest non-king, non-tarok cards
        discardable = [
            c for c in hand
            if not c.is_king and c.card_type != CardType.TAROK
        ]
        if len(discardable) < must_discard:
            # Must discard taroks (allowed when no suit cards)
            discardable = [c for c in hand if not c.is_king]
        discardable.sort(key=lambda c: c.points)
        return discardable[:must_discard]

    async def choose_announcements(
        self, state: GameState, player_idx: int
    ) -> list[Announcement]:
        return []  # Random player never announces

    async def choose_card(
        self, state: GameState, player_idx: int, legal_plays: list[Card]
    ) -> Card:
        return self._rng.choice(legal_plays)

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/ai/trainer.py`
```python
"""PPO Trainer — session-based self-play training with multi-decision heads.

Training is organized into *sessions* of N games (default 20).  The agent
learns to maximise cumulative score across a session, which naturally teaches
strategic bidding: when to pass (berac/klop), when to bid conservatively
(tri/dva), and when to go for high-value solo contracts.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tarok.adapters.ai.agent import RLAgent, Experience
from tarok.adapters.ai.encoding import (
    DecisionType,
    BID_ACTION_SIZE,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
    CARD_ACTION_SIZE,
    ANNOUNCE_ACTION_SIZE,
)
from tarok.use_cases.game_loop import GameLoop


_ACTION_SIZES = {
    DecisionType.BID: BID_ACTION_SIZE,
    DecisionType.KING_CALL: KING_ACTION_SIZE,
    DecisionType.TALON_PICK: TALON_ACTION_SIZE,
    DecisionType.CARD_PLAY: CARD_ACTION_SIZE,
    DecisionType.ANNOUNCE: ANNOUNCE_ACTION_SIZE,
}


@dataclass
class ContractStats:
    """Per-contract stats, split by role (declarer vs defender)."""
    # As declarer (P0 won the bidding for this contract)
    decl_played: int = 0
    decl_won: int = 0
    decl_total_score: int = 0
    # As defender (another player declared this contract)
    def_played: int = 0
    def_won: int = 0
    def_total_score: int = 0

    @property
    def played(self) -> int:
        return self.decl_played + self.def_played

    @property
    def decl_win_rate(self) -> float:
        return self.decl_won / max(self.decl_played, 1)

    @property
    def def_win_rate(self) -> float:
        return self.def_won / max(self.def_played, 1)

    @property
    def decl_avg_score(self) -> float:
        return self.decl_total_score / max(self.decl_played, 1)

    @property
    def def_avg_score(self) -> float:
        return self.def_total_score / max(self.def_played, 1)

    def to_dict(self) -> dict:
        return {
            "played": self.played,
            "decl_played": self.decl_played,
            "decl_won": self.decl_won,
            "decl_win_rate": round(self.decl_win_rate, 4),
            "decl_avg_score": round(self.decl_avg_score, 1),
            "def_played": self.def_played,
            "def_won": self.def_won,
            "def_win_rate": round(self.def_win_rate, 4),
            "def_avg_score": round(self.def_avg_score, 1),
        }


# Contract names we individually track
_TRACKED_CONTRACTS = ["klop", "three", "two", "one", "solo_three", "solo_two", "solo_one", "solo"]


@dataclass
class TrainingMetrics:
    episode: int = 0
    total_episodes: int = 0
    session: int = 0
    total_sessions: int = 0
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    entropy: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    games_per_second: float = 0.0
    bid_rate: float = 0.0
    klop_rate: float = 0.0
    solo_rate: float = 0.0
    contract_stats: dict[str, ContractStats] = field(
        default_factory=lambda: {c: ContractStats() for c in _TRACKED_CONTRACTS}
    )
    reward_history: list[float] = field(default_factory=list)
    win_rate_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    bid_rate_history: list[float] = field(default_factory=list)
    klop_rate_history: list[float] = field(default_factory=list)
    solo_rate_history: list[float] = field(default_factory=list)
    contract_win_rate_history: dict[str, list[float]] = field(
        default_factory=lambda: {c: [] for c in _TRACKED_CONTRACTS}
    )
    session_avg_score_history: list[float] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "total_episodes": self.total_episodes,
            "session": self.session,
            "total_sessions": self.total_sessions,
            "avg_reward": round(self.avg_reward, 2),
            "avg_loss": round(self.avg_loss, 4),
            "win_rate": round(self.win_rate, 4),
            "entropy": round(self.entropy, 4),
            "value_loss": round(self.value_loss, 4),
            "policy_loss": round(self.policy_loss, 4),
            "games_per_second": round(self.games_per_second, 2),
            "bid_rate": round(self.bid_rate, 4),
            "klop_rate": round(self.klop_rate, 4),
            "solo_rate": round(self.solo_rate, 4),
            "contract_stats": {k: v.to_dict() for k, v in self.contract_stats.items()},
            "reward_history": self.reward_history[-500:],
            "win_rate_history": self.win_rate_history[-500:],
            "loss_history": self.loss_history[-500:],
            "bid_rate_history": self.bid_rate_history[-500:],
            "klop_rate_history": self.klop_rate_history[-500:],
            "solo_rate_history": self.solo_rate_history[-500:],
            "contract_win_rate_history": {
                k: v[-500:] for k, v in self.contract_win_rate_history.items()
            },
            "session_avg_score_history": self.session_avg_score_history[-500:],
            "snapshots": self.snapshots,
        }


class PPOTrainer:
    """Session-based self-play PPO trainer for Tarok agents.

    Each training session plays *games_per_session* games, collects all
    experiences (bids + king calls + talon picks + card plays), then
    performs a PPO update.  This teaches the agent the expected value of
    each bid given a hand, and the long-run payoff of conservative vs
    aggressive play.
    """

    def __init__(
        self,
        agents: list[RLAgent],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        games_per_session: int = 20,
        device: str = "cpu",
        save_dir: str = "checkpoints",
    ):
        self.agents = agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.games_per_session = games_per_session
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # All agents share the same network for self-play
        self.shared_network = agents[0].network
        self.optimizer = optim.Adam(self.shared_network.parameters(), lr=lr)

        # Sync all agents to use the same network
        for agent in agents:
            agent.network = self.shared_network

        self.metrics = TrainingMetrics()
        self._running = False
        self._metrics_callback: list = []

    def add_metrics_callback(self, callback) -> None:
        self._metrics_callback.append(callback)

    async def train(self, num_sessions: int) -> TrainingMetrics:
        """Run session-based self-play training.

        Each session = *games_per_session* games → one PPO update.
        Metrics update after every game for live dashboard feedback.
        """
        self._running = True
        total_games = num_sessions * self.games_per_session
        self.metrics.total_episodes = total_games
        self.metrics.total_sessions = num_sessions

        recent_rewards: list[float] = []
        recent_wins: list[float] = []
        recent_bids: list[float] = []
        recent_klops: list[float] = []
        recent_solos: list[float] = []
        start_time = time.time()
        game_count = 0
        # Running sums for O(1) rolling-window metrics
        _rsum = 0.0; _wsum = 0.0; _bsum = 0.0; _ksum = 0.0; _ssum = 0.0

        for agent in self.agents:
            agent.set_training(True)

        snapshot_interval = max(1, num_sessions // 10)

        for session_idx in range(num_sessions):
            if not self._running:
                break

            self.metrics.session = session_idx + 1
            all_experiences: list[Experience] = []
            session_scores: list[int] = []

            for g in range(self.games_per_session):
                if not self._running:
                    break

                # Clear experiences from previous game
                for agent in self.agents:
                    agent.clear_experiences()

                # Play one game
                game = GameLoop(self.agents)
                state, scores = await game.run(dealer=(game_count + g) % 4)

                # Extract stats
                is_klop = state.contract is not None and state.contract.is_klop
                is_solo = state.contract is not None and state.contract.is_solo
                agent0_bids = [b for b in state.bids if b.player == 0 and b.contract is not None]
                contract_name = state.contract.name.lower() if state.contract else "klop"
                raw_score = scores.get(0, 0)
                declarer_p0 = state.declarer == 0

                # Finalize rewards and collect experiences
                for i, agent in enumerate(self.agents):
                    reward = scores.get(i, 0) / 100.0
                    agent.finalize_game(reward)
                    all_experiences.extend(agent.experiences)

                game_count += 1
                session_scores.append(raw_score)
                r = raw_score / 100.0
                w = 1.0 if raw_score > 0 else 0.0
                b = 1.0 if agent0_bids else 0.0
                k = 1.0 if is_klop else 0.0
                s = 1.0 if is_solo else 0.0
                recent_rewards.append(r); recent_wins.append(w)
                recent_bids.append(b); recent_klops.append(k); recent_solos.append(s)
                _rsum += r; _wsum += w; _bsum += b; _ksum += k; _ssum += s

                # Evict oldest entry once we exceed the window
                window = self.games_per_session * 10
                if len(recent_rewards) > window:
                    _rsum -= recent_rewards[-window - 1]
                    _wsum -= recent_wins[-window - 1]
                    _bsum -= recent_bids[-window - 1]
                    _ksum -= recent_klops[-window - 1]
                    _ssum -= recent_solos[-window - 1]

                # Per-contract tracking (split by role)
                if contract_name in self.metrics.contract_stats:
                    cs = self.metrics.contract_stats[contract_name]
                    if declarer_p0:
                        cs.decl_played += 1
                        if raw_score > 0:
                            cs.decl_won += 1
                        cs.decl_total_score += raw_score
                    else:
                        cs.def_played += 1
                        if raw_score > 0:
                            cs.def_won += 1
                        cs.def_total_score += raw_score

                # Update live metrics (O(1) — no recomputation)
                self.metrics.episode = game_count
                n = min(len(recent_rewards), window)
                self.metrics.avg_reward = _rsum / n
                self.metrics.win_rate = _wsum / n
                self.metrics.bid_rate = _bsum / n
                self.metrics.klop_rate = _ksum / n
                self.metrics.solo_rate = _ssum / n

                # Throttle callbacks: every 10 games (print flush + async yield is expensive)
                if game_count % 10 == 0 or g == self.games_per_session - 1:
                    elapsed = time.time() - start_time
                    self.metrics.games_per_second = game_count / max(elapsed, 1e-6)
                    for cb in self._metrics_callback:
                        await cb(self.metrics)
                    # Yield to the event loop so FastAPI can serve HTTP requests
                    await asyncio.sleep(0)

            # Per-session average score
            if session_scores:
                self.metrics.session_avg_score_history.append(
                    round(sum(session_scores) / len(session_scores), 2)
                )

            # --- PPO update on the full session ---
            if all_experiences:
                loss_info = self._ppo_update(all_experiences)
                self.metrics.policy_loss = loss_info["policy_loss"]
                self.metrics.value_loss = loss_info["value_loss"]
                self.metrics.entropy = loss_info["entropy"]
                self.metrics.avg_loss = loss_info["total_loss"]

            # --- Append per-session history for charts ---
            self.metrics.reward_history.append(self.metrics.avg_reward)
            self.metrics.win_rate_history.append(self.metrics.win_rate)
            self.metrics.loss_history.append(self.metrics.avg_loss)
            self.metrics.bid_rate_history.append(self.metrics.bid_rate)
            self.metrics.klop_rate_history.append(self.metrics.klop_rate)
            self.metrics.solo_rate_history.append(self.metrics.solo_rate)

            # Per-contract declarer win rate history
            for cname in _TRACKED_CONTRACTS:
                cs = self.metrics.contract_stats[cname]
                self.metrics.contract_win_rate_history[cname].append(
                    round(cs.decl_win_rate, 4)
                )

            # Periodic snapshot checkpoint
            if (session_idx + 1) % snapshot_interval == 0 or session_idx == num_sessions - 1:
                snap_info = self._save_checkpoint(game_count, is_snapshot=True)
                self.metrics.snapshots.append(snap_info)

        self._save_checkpoint(game_count, is_snapshot=False)
        return self.metrics

    def _ppo_update(self, experiences: list[Experience]) -> dict[str, float]:
        """Perform PPO update on collected experiences, grouped by decision type."""
        if not experiences:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        # Group experiences by decision type for correct head routing
        grouped: dict[DecisionType, list[Experience]] = defaultdict(list)
        for exp in experiences:
            grouped[exp.decision_type].append(exp)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for dt, dt_exps in grouped.items():
            if not dt_exps:
                continue

            action_size = _ACTION_SIZES[dt]

            states = torch.stack([e.state for e in dt_exps]).to(self.device)
            actions = torch.tensor([e.action for e in dt_exps], dtype=torch.long).to(self.device)
            old_log_probs = torch.stack([e.log_prob for e in dt_exps]).to(self.device)
            rewards = torch.tensor([e.reward for e in dt_exps], dtype=torch.float32).to(self.device)
            old_values = torch.stack([e.value for e in dt_exps]).to(self.device)

            # Advantages
            advantages = rewards - old_values.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            returns = rewards

            # Legal masks (all-ones placeholder — actions were legal at play time)
            legal_masks = torch.ones(len(dt_exps), action_size, dtype=torch.float32).to(self.device)

            for _ in range(self.epochs_per_update):
                indices = torch.randperm(len(dt_exps))

                for start in range(0, len(dt_exps), self.batch_size):
                    end = min(start + self.batch_size, len(dt_exps))
                    batch_idx = indices[start:end]

                    b_states = states[batch_idx]
                    b_actions = actions[batch_idx]
                    b_old_log_probs = old_log_probs[batch_idx]
                    b_advantages = advantages[batch_idx]
                    b_returns = returns[batch_idx]
                    b_masks = legal_masks[batch_idx]

                    new_log_probs, new_values, entropy = self.shared_network.evaluate_action(
                        b_states, b_actions, b_masks, dt,
                    )

                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    ) * b_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.functional.mse_loss(new_values, b_returns)

                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy.mean()
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.shared_network.parameters(), 0.5)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates += 1

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "total_loss": (total_policy_loss + total_value_loss) / n,
        }

    def _save_checkpoint(self, episode: int, is_snapshot: bool = False) -> dict:
        data = {
            "episode": episode,
            "session": self.metrics.session,
            "model_state_dict": self.shared_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics.to_dict(),
        }

        # Always save as 'latest'
        latest = self.save_dir / "tarok_agent_latest.pt"
        torch.save(data, latest)

        # Numbered snapshot
        path = self.save_dir / f"tarok_agent_ep{episode}.pt"
        torch.save(data, path)

        info = {
            "filename": path.name,
            "episode": episode,
            "session": self.metrics.session,
            "win_rate": round(self.metrics.win_rate, 4),
            "avg_reward": round(self.metrics.avg_reward, 2),
            "games_per_second": round(self.metrics.games_per_second, 1),
        }
        return info

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.shared_network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def stop(self) -> None:
        self._running = False

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/ai/worker.py`
```python
"""Parallel self-play worker — runs games in a subprocess and returns experiences.

Workers are long-lived: the pool initializer creates agents once per process,
and each ``play_games_worker`` call just loads fresh weights and plays games.
This avoids the overhead of re-importing modules and rebuilding neural networks
on every session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch

from tarok.adapters.ai.agent import RLAgent, Experience
from tarok.adapters.ai.encoding import DecisionType
from tarok.use_cases.game_loop import GameLoop


@dataclass
class GameStats:
    """Per-game statistics returned by a worker."""
    reward_p0: float
    won_p0: bool
    bid_p0: bool
    is_klop: bool
    is_solo: bool
    contract_name: str  # e.g. "klop", "three", "two", "one", "solo_three", etc.
    score_p0: int       # raw score (not /100)
    declarer_p0: bool   # True if Player 0 was the declarer (won the bidding)


@dataclass
class WorkerResult:
    """Everything a worker returns after playing its batch of games."""
    experiences: list[dict]  # serialized Experience dicts (CPU tensors)
    game_stats: list[GameStats]


# ---------------------------------------------------------------------------
# Per-process global state — initialised once by ``init_worker``
# ---------------------------------------------------------------------------
_worker_agents: list[RLAgent] | None = None


def init_worker(hidden_size: int, explore_rate: float) -> None:
    """Pool initializer — called once in each worker subprocess."""
    global _worker_agents
    _worker_agents = [
        RLAgent(name=f"W-{i}", hidden_size=hidden_size, device="cpu",
                explore_rate=explore_rate)
        for i in range(4)
    ]
    shared_net = _worker_agents[0].network
    for agent in _worker_agents[1:]:
        agent.network = shared_net
    for agent in _worker_agents:
        agent.set_training(True)


def _serialize_experience(exp: Experience) -> dict:
    """Convert Experience to a plain dict with CPU tensors for pickling."""
    return {
        "state": exp.state.cpu(),
        "action": exp.action,
        "log_prob": exp.log_prob.cpu(),
        "value": exp.value.cpu(),
        "decision_type": exp.decision_type.value,
        "reward": exp.reward,
        "done": exp.done,
    }


def deserialize_experience(d: dict) -> Experience:
    """Reconstruct Experience from a serialized dict."""
    return Experience(
        state=d["state"],
        action=d["action"],
        log_prob=d["log_prob"],
        value=d["value"],
        decision_type=DecisionType(d["decision_type"]),
        reward=d["reward"],
        done=d["done"],
    )


def play_games_worker(args: tuple) -> WorkerResult:
    """Top-level function for multiprocessing — must be picklable.

    Args is a tuple of (state_dict, num_games, dealer_offset).
    Agents are already initialised via ``init_worker``.
    """
    state_dict, num_games, dealer_offset = args
    assert _worker_agents is not None, "init_worker was not called"

    agents = _worker_agents
    # Load the latest shared weights
    agents[0].network.load_state_dict(state_dict)

    all_experiences: list[dict] = []
    all_stats: list[GameStats] = []

    async def _run():
        for g in range(num_games):
            for agent in agents:
                agent.clear_experiences()

            game = GameLoop(agents)
            state, scores = await game.run(dealer=(dealer_offset + g) % 4)

            is_klop = state.contract is not None and state.contract.is_klop
            is_solo = state.contract is not None and state.contract.is_solo
            agent0_bids = [b for b in state.bids if b.player == 0 and b.contract is not None]
            contract_name = state.contract.name.lower() if state.contract else "klop"
            raw_score = scores.get(0, 0)
            declarer_p0 = state.declarer == 0

            for i, agent in enumerate(agents):
                reward = scores.get(i, 0) / 100.0
                agent.finalize_game(reward)
                for exp in agent.experiences:
                    all_experiences.append(_serialize_experience(exp))

            all_stats.append(GameStats(
                reward_p0=raw_score / 100.0,
                won_p0=raw_score > 0,
                bid_p0=bool(agent0_bids),
                is_klop=is_klop,
                is_solo=is_solo,
                contract_name=contract_name,
                score_p0=raw_score,
                declarer_p0=declarer_p0,
            ))

    asyncio.run(_run())
    return WorkerResult(experiences=all_experiences, game_stats=all_stats)

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/api/__init__.py`
```python

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/api/human_player.py`
```python
"""Human player adapter — bridges WebSocket input to PlayerPort."""

from __future__ import annotations

import asyncio

from tarok.entities.card import Card, CardType, Suit, SuitRank
from tarok.entities.game_state import Announcement, Contract, GameState


class HumanPlayer:
    """Player controlled via WebSocket. Waits for human input."""

    def __init__(self, name: str = "Human"):
        self._name = name
        self._pending_action: asyncio.Future | None = None

    @property
    def name(self) -> str:
        return self._name

    def submit_action(self, action) -> None:
        """Called by the WebSocket handler when the human submits a move."""
        if self._pending_action and not self._pending_action.done():
            self._pending_action.set_result(action)

    async def _wait_for_input(self):
        loop = asyncio.get_event_loop()
        self._pending_action = loop.create_future()
        result = await self._pending_action
        self._pending_action = None
        return result

    async def choose_bid(
        self, state: GameState, player_idx: int, legal_bids: list[Contract | None]
    ) -> Contract | None:
        return await self._wait_for_input()

    async def choose_king(
        self, state: GameState, player_idx: int, callable_kings: list[Card]
    ) -> Card:
        return await self._wait_for_input()

    async def choose_talon_group(
        self, state: GameState, player_idx: int, talon_groups: list[list[Card]]
    ) -> int:
        return await self._wait_for_input()

    async def choose_discard(
        self, state: GameState, player_idx: int, must_discard: int
    ) -> list[Card]:
        return await self._wait_for_input()

    async def choose_announcements(
        self, state: GameState, player_idx: int
    ) -> list[Announcement]:
        return []  # Simplified for now

    async def choose_card(
        self, state: GameState, player_idx: int, legal_plays: list[Card]
    ) -> Card:
        return await self._wait_for_input()

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/api/schemas.py`
```python
"""Pydantic schemas for the API adapter."""

from __future__ import annotations

from pydantic import BaseModel


class CardSchema(BaseModel):
    card_type: str  # "tarok" or "suit"
    value: int
    suit: str | None = None
    label: str
    points: int


class TrickSchema(BaseModel):
    cards: list[tuple[int, CardSchema]]
    lead_player: int
    winner: int | None = None


class GameStateSchema(BaseModel):
    phase: str
    hand: list[CardSchema]
    hand_sizes: list[int]
    talon_groups: list[list[CardSchema]] | None = None
    bids: list[dict]
    contract: str | None = None
    declarer: int | None = None
    called_king: CardSchema | None = None
    partner_revealed: bool = False
    partner: int | None = None
    current_trick: list[tuple[int, CardSchema]]
    tricks_played: int = 0
    current_player: int = 0
    scores: dict[str, int] | None = None
    legal_plays: list[CardSchema] = []
    player_names: list[str] = []


class PlayCardRequest(BaseModel):
    card_type: str
    value: int
    suit: str | None = None


class BidRequest(BaseModel):
    contract: int | None  # Contract value or None for pass


class CallKingRequest(BaseModel):
    suit: str


class TalonChoiceRequest(BaseModel):
    group_index: int


class DiscardRequest(BaseModel):
    cards: list[PlayCardRequest]


class TrainingRequest(BaseModel):
    num_sessions: int = 100
    games_per_session: int = 100
    learning_rate: float = 3e-4
    hidden_size: int = 256
    resume: bool = False


class TrainingMetricsSchema(BaseModel):
    episode: int = 0
    total_episodes: int = 0
    session: int = 0
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    entropy: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    games_per_second: float = 0.0
    bid_rate: float = 0.0
    klop_rate: float = 0.0
    solo_rate: float = 0.0
    reward_history: list[float] = []
    win_rate_history: list[float] = []
    loss_history: list[float] = []

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/api/server.py`
```python
"""FastAPI server — REST + WebSocket adapter for the Tarok game."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.trainer import PPOTrainer, TrainingMetrics
from tarok.adapters.api.human_player import HumanPlayer
from tarok.adapters.api.ws_observer import WebSocketObserver
from tarok.adapters.api.schemas import (
    TrainingRequest,
    TrainingMetricsSchema,
)
from tarok.entities.card import Card, CardType, Suit, SuitRank, DECK
from tarok.entities.game_state import Contract, GameState, Phase, PlayerRole, Trick
from tarok.use_cases.game_loop import GameLoop

# --- Globals managed by lifespan ---
_trainer: PPOTrainer | None = None
_training_task: asyncio.Task | None = None
_latest_metrics: TrainingMetrics | None = None
_active_games: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _trainer, _training_task
    if _trainer:
        _trainer.stop()
    if _training_task and not _training_task.done():
        _training_task.cancel()


app = FastAPI(title="Tarok API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Training endpoints ----

@app.post("/api/training/start")
async def start_training(req: TrainingRequest):
    global _trainer, _training_task, _latest_metrics

    if _training_task and not _training_task.done():
        return {"status": "already_running", "metrics": _latest_metrics.to_dict() if _latest_metrics else None}

    agents = [RLAgent(name=f"Agent-{i}", hidden_size=req.hidden_size) for i in range(4)]

    # Resume from latest checkpoint if available
    checkpoint_path = Path("checkpoints/tarok_agent_latest.pt")
    if req.resume and checkpoint_path.exists():
        import torch as _torch
        ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agents[0].network.load_state_dict(ckpt["model_state_dict"])

    _trainer = PPOTrainer(
        agents, lr=req.learning_rate, device="cpu",
        games_per_session=req.games_per_session,
    )

    async def on_metrics(metrics: TrainingMetrics):
        global _latest_metrics
        _latest_metrics = metrics

    _trainer.add_metrics_callback(on_metrics)

    async def run_training():
        global _latest_metrics
        result = await _trainer.train(req.num_sessions)
        _latest_metrics = result

    _training_task = asyncio.create_task(run_training())
    return {"status": "started", "num_sessions": req.num_sessions, "games_per_session": req.games_per_session}


@app.post("/api/training/stop")
async def stop_training():
    global _trainer, _training_task
    if _trainer:
        _trainer.stop()
    return {"status": "stopped"}


@app.get("/api/training/metrics")
async def get_metrics() -> dict:
    if _latest_metrics:
        return _latest_metrics.to_dict()
    return TrainingMetrics().to_dict()


@app.get("/api/training/status")
async def training_status():
    running = _training_task is not None and not _training_task.done()
    return {"running": running}


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List all saved checkpoint files."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return {"checkpoints": []}
    files = sorted(ckpt_dir.glob("tarok_agent_ep*.pt"))
    result = []
    for f in files:
        import torch as _torch
        try:
            meta = _torch.load(f, map_location="cpu", weights_only=False)
            result.append({
                "filename": f.name,
                "episode": meta.get("episode", 0),
                "session": meta.get("session", 0),
                "win_rate": meta.get("metrics", {}).get("win_rate", 0),
                "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
            })
        except Exception:
            result.append({"filename": f.name, "episode": 0})
    return {"checkpoints": result}


# ---- Game endpoints ----

@app.post("/api/game/new")
async def new_game():
    """Create a new human-vs-AI game."""
    game_id = f"game-{len(_active_games)}"

    human = HumanPlayer(name="You")

    # Load trained agent if available
    agents: list = [human]
    checkpoint_path = Path("checkpoints/tarok_agent_latest.pt")

    for i in range(3):
        agent = RLAgent(name=f"AI-{i+1}")
        agent.set_training(False)
        if checkpoint_path.exists():
            import torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            agent.network.load_state_dict(checkpoint["model_state_dict"])
        agents.append(agent)

    _active_games[game_id] = {
        "human": human,
        "agents": agents,
        "game_loop": None,
        "state": None,
    }
    return {"game_id": game_id}


@app.websocket("/ws/game/{game_id}")
async def game_websocket(ws: WebSocket, game_id: str):
    await ws.accept()

    if game_id not in _active_games:
        await ws.close(code=4004, reason="Game not found")
        return

    game_info = _active_games[game_id]
    human: HumanPlayer = game_info["human"]
    agents = game_info["agents"]
    player_names = [a.name for a in agents]

    observer = WebSocketObserver(ws, player_idx=0, player_names=player_names)
    game_loop = GameLoop(agents, observer=observer)

    # Start game in background
    game_task = asyncio.create_task(game_loop.run())

    try:
        while True:
            data = await ws.receive_json()
            action_type = data.get("action")

            if action_type == "bid":
                contract_val = data.get("contract")
                if contract_val is None:
                    human.submit_action(None)
                else:
                    human.submit_action(Contract(contract_val))

            elif action_type == "call_king":
                suit = Suit(data["suit"])
                king = Card(CardType.SUIT, SuitRank.KING.value, suit)
                human.submit_action(king)

            elif action_type == "choose_talon":
                human.submit_action(data["group_index"])

            elif action_type == "discard":
                cards = []
                for c in data["cards"]:
                    card = Card(
                        CardType(c["card_type"]),
                        c["value"],
                        Suit(c["suit"]) if c.get("suit") else None,
                    )
                    cards.append(card)
                human.submit_action(cards)

            elif action_type == "play_card":
                c = data["card"]
                card = Card(
                    CardType(c["card_type"]),
                    c["value"],
                    Suit(c["suit"]) if c.get("suit") else None,
                )
                human.submit_action(card)

    except WebSocketDisconnect:
        game_task.cancel()
        if game_id in _active_games:
            del _active_games[game_id]


@app.websocket("/ws/training")
async def training_websocket(ws: WebSocket):
    """Stream training metrics to the frontend."""
    await ws.accept()
    try:
        while True:
            if _latest_metrics:
                await ws.send_json({
                    "event": "metrics",
                    "data": _latest_metrics.to_dict(),
                })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


# ---- Health check ----

@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---- Camera Agent: analyze hand and recommend play ----

class CardInput(BaseModel):
    card_type: str  # "tarok" or "suit"
    value: int
    suit: str | None = None


class AnalyzeRequest(BaseModel):
    """Cards the user holds + the current trick + game context."""
    hand: list[CardInput]
    trick: list[CardInput] = []  # cards already played in current trick
    contract: str = "three"  # contract name
    position: int = 0  # 0=declarer, 1=partner, 2/3=opponent
    tricks_played: int = 0
    played_cards: list[CardInput] = []  # all previously played cards


def _parse_card(ci: CardInput) -> Card:
    ct = CardType(ci.card_type)
    s = Suit(ci.suit) if ci.suit else None
    return Card(ct, ci.value, s)


def _card_to_dict(card: Card) -> dict:
    return {
        "card_type": card.card_type.value,
        "value": card.value,
        "suit": card.suit.value if card.suit else None,
        "label": card.label,
        "points": card.points,
    }


# Map of known cards in the DECK for quick label-to-card lookup
_LABEL_TO_CARD: dict[str, Card] = {c.label: c for c in DECK}

CONTRACT_NAME_MAP = {
    "three": Contract.THREE,
    "two": Contract.TWO,
    "one": Contract.ONE,
    "solo_three": Contract.SOLO_THREE,
    "solo_two": Contract.SOLO_TWO,
    "solo_one": Contract.SOLO_ONE,
    "solo": Contract.SOLO,
}


@app.post("/api/analyze")
async def analyze_hand(req: AnalyzeRequest):
    """Given a hand of cards and game context, return the AI's recommended play.

    This endpoint lets users photograph a real Tarok hand, input the cards,
    and get the trained agent's recommendation for what to play.
    """
    # Parse cards
    hand = [_parse_card(c) for c in req.hand]
    trick_cards = [_parse_card(c) for c in req.trick]
    played = [_parse_card(c) for c in req.played_cards]

    # Build a synthetic game state for the agent
    state = GameState(phase=Phase.TRICK_PLAY)
    contract = CONTRACT_NAME_MAP.get(req.contract, Contract.THREE)
    state.contract = contract
    state.declarer = 0 if req.position == 0 else 1

    # Set player 0 as the user
    state.hands[0] = list(hand)
    state.current_player = 0
    state.roles = {
        0: PlayerRole.DECLARER if req.position == 0 else (
            PlayerRole.PARTNER if req.position == 1 else PlayerRole.OPPONENT
        ),
        1: PlayerRole.PARTNER if req.position != 1 else PlayerRole.DECLARER,
        2: PlayerRole.OPPONENT,
        3: PlayerRole.OPPONENT,
    }

    # Build current trick if cards have been played
    if trick_cards:
        state.current_trick = Trick(lead_player=(4 - len(trick_cards)) % 4)
        for i, card in enumerate(trick_cards):
            player_idx = (state.current_trick.lead_player + i) % 4
            state.current_trick.cards.append((player_idx, card))
    else:
        state.current_trick = Trick(lead_player=0)

    # Record previously played tricks
    state.tricks = []
    # Approximate: create empty trick records for played tricks count
    for i in range(req.tricks_played):
        t = Trick(lead_player=0)
        # Minimal trick data
        state.tricks.append(t)

    # Compute legal plays
    legal = state.legal_plays(0)

    # Load the trained agent and get its recommendation
    agent = RLAgent(name="Advisor")
    agent.set_training(False)
    checkpoint_path = Path("checkpoints/tarok_agent_latest.pt")
    if checkpoint_path.exists():
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agent.network.load_state_dict(checkpoint["model_state_dict"])

    # Get the agent's card choice
    recommended = await agent.choose_card(state, 0)

    # Also rank all legal plays by the agent's policy
    from tarok.adapters.ai.encoding import encode_state, encode_legal_mask, CARD_TO_IDX
    import torch

    state_tensor = torch.tensor(encode_state(state, 0), dtype=torch.float32).unsqueeze(0)
    legal_mask = torch.tensor(encode_legal_mask(legal), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, value = agent.network(state_tensor)

    # Mask illegal actions
    masked = logits.clone()
    masked[legal_mask == 0] = float('-inf')
    probs = torch.softmax(masked, dim=-1).squeeze(0)

    # Build ranked recommendations
    ranked = []
    for card in legal:
        idx = CARD_TO_IDX.get(card)
        if idx is not None:
            prob = probs[idx].item()
            ranked.append({
                "card": _card_to_dict(card),
                "probability": round(prob, 4),
            })

    ranked.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "recommended": _card_to_dict(recommended),
        "legal_plays": [_card_to_dict(c) for c in legal],
        "ranked_plays": ranked,
        "position_value": round(value.item(), 4) if value is not None else None,
        "has_trained_model": checkpoint_path.exists(),
    }

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/adapters/api/ws_observer.py`
```python
"""WebSocket observer — broadcasts game events to connected clients."""

from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket

from tarok.entities.card import Card
from tarok.entities.game_state import Contract, GameState, Trick


def _card_to_dict(card: Card) -> dict:
    return {
        "card_type": card.card_type.value,
        "value": card.value,
        "suit": card.suit.value if card.suit else None,
        "label": card.label,
        "points": card.points,
    }


def _state_for_player(state: GameState, player_idx: int, player_names: list[str]) -> dict:
    legal = state.legal_plays(player_idx) if state.current_player == player_idx else []
    return {
        "phase": state.phase.value,
        "hand": [_card_to_dict(c) for c in state.hands[player_idx]],
        "hand_sizes": [len(h) for h in state.hands],
        "talon_groups": (
            [[_card_to_dict(c) for c in g] for g in state.talon_revealed]
            if state.talon_revealed
            else None
        ),
        "bids": [
            {"player": b.player, "contract": b.contract.value if b.contract else None}
            for b in state.bids
        ],
        "contract": state.contract.value if state.contract else None,
        "declarer": state.declarer,
        "called_king": _card_to_dict(state.called_king) if state.called_king else None,
        "partner_revealed": state.is_partner_revealed,
        "partner": state.partner if state.is_partner_revealed else None,
        "current_trick": (
            [(p, _card_to_dict(c)) for p, c in state.current_trick.cards]
            if state.current_trick
            else []
        ),
        "tricks_played": state.tricks_played,
        "current_player": state.current_player,
        "scores": state.scores if state.scores else None,
        "legal_plays": [_card_to_dict(c) for c in legal],
        "player_names": player_names,
    }


class WebSocketObserver:
    """Broadcasts game events to a connected WebSocket client."""

    def __init__(self, ws: WebSocket, player_idx: int, player_names: list[str]):
        self._ws = ws
        self._player_idx = player_idx
        self._player_names = player_names

    async def _send(self, event: str, data: Any, state: GameState) -> None:
        msg = {
            "event": event,
            "data": data,
            "state": _state_for_player(state, self._player_idx, self._player_names),
        }
        await self._ws.send_json(msg)

    async def on_game_start(self, state: GameState) -> None:
        await self._send("game_start", {}, state)

    async def on_deal(self, state: GameState) -> None:
        await self._send("deal", {}, state)

    async def on_bid(self, player: int, bid: Contract | None, state: GameState) -> None:
        await self._send("bid", {
            "player": player,
            "contract": bid.value if bid else None,
        }, state)

    async def on_contract_won(self, player: int, contract: Contract, state: GameState) -> None:
        await self._send("contract_won", {
            "player": player,
            "contract": contract.value,
        }, state)

    async def on_king_called(self, player: int, king: Card, state: GameState) -> None:
        await self._send("king_called", {
            "player": player,
            "king": _card_to_dict(king),
        }, state)

    async def on_talon_revealed(self, groups: list[list[Card]], state: GameState) -> None:
        await self._send("talon_revealed", {
            "groups": [[_card_to_dict(c) for c in g] for g in groups],
        }, state)

    async def on_talon_exchanged(self, state: GameState) -> None:
        await self._send("talon_exchanged", {}, state)

    async def on_card_played(self, player: int, card: Card, state: GameState) -> None:
        await self._send("card_played", {
            "player": player,
            "card": _card_to_dict(card),
        }, state)

    async def on_trick_won(self, trick: Trick, winner: int, state: GameState) -> None:
        await self._send("trick_won", {
            "winner": winner,
            "cards": [(p, _card_to_dict(c)) for p, c in trick.cards],
        }, state)

    async def on_game_end(self, scores: dict[int, int], state: GameState) -> None:
        await self._send("game_end", {
            "scores": {str(k): v for k, v in scores.items()},
        }, state)

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/entities/__init__.py`
```python
from tarok.entities.card import Card, CardType, Suit, SuitRank, DECK, tarok, suit_card
from tarok.entities.game_state import (
    GameState,
    Phase,
    Contract,
    Trick,
    Team,
    PlayerRole,
    Announcement,
)
from tarok.entities.scoring import compute_card_points, score_game

__all__ = [
    "Card",
    "CardType",
    "Suit",
    "SuitRank",
    "DECK",
    "tarok",
    "suit_card",
    "GameState",
    "Phase",
    "Contract",
    "Trick",
    "Team",
    "PlayerRole",
    "Announcement",
    "compute_card_points",
    "score_game",
]

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/entities/card.py`
```python
"""Card domain entity for Slovenian Tarok.

54 cards total:
  - 22 Taroks: I (Pagat) through XXI (Mond) + Škis (the Fool)
  - 32 Suit cards: 4 suits × 8 cards (King, Queen, Knight, Jack + 4 pips)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CardType(Enum):
    TAROK = "tarok"
    SUIT = "suit"


class Suit(Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


class SuitRank(Enum):
    """Ranks within a suit, ordered lowest to highest."""
    PIP_1 = 1  # Red: 1, Black: 7
    PIP_2 = 2  # Red: 2, Black: 8
    PIP_3 = 3  # Red: 3, Black: 9
    PIP_4 = 4  # Red: 4, Black: 10
    JACK = 5
    KNIGHT = 6
    QUEEN = 7
    KING = 8


# Display labels for pip cards by suit color
_RED_PIP_LABELS = {SuitRank.PIP_1: "1", SuitRank.PIP_2: "2", SuitRank.PIP_3: "3", SuitRank.PIP_4: "4"}
_BLACK_PIP_LABELS = {SuitRank.PIP_1: "7", SuitRank.PIP_2: "8", SuitRank.PIP_3: "9", SuitRank.PIP_4: "10"}

_FACE_LABELS = {
    SuitRank.JACK: "J",
    SuitRank.KNIGHT: "C",  # Kavall/Cavalier
    SuitRank.QUEEN: "Q",
    SuitRank.KING: "K",
}

_TAROK_ROMAN = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
    6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X",
    11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV",
    16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX",
    21: "XXI", 22: "Škis",
}

SUIT_SYMBOLS = {
    Suit.HEARTS: "♥",
    Suit.DIAMONDS: "♦",
    Suit.CLUBS: "♣",
    Suit.SPADES: "♠",
}

_POINT_VALUES = {
    SuitRank.KING: 5,
    SuitRank.QUEEN: 4,
    SuitRank.KNIGHT: 3,
    SuitRank.JACK: 2,
}

SKIS = 22
MOND = 21
PAGAT = 1


@dataclass(frozen=True, slots=True)
class Card:
    card_type: CardType
    value: int  # Taroks: 1–22 (22=Škis). Suit cards: SuitRank.value
    suit: Suit | None = None

    @property
    def points(self) -> int:
        if self.card_type == CardType.TAROK:
            if self.value in (PAGAT, MOND, SKIS):
                return 5
            return 1
        rank = SuitRank(self.value)
        return _POINT_VALUES.get(rank, 1)

    @property
    def is_trula(self) -> bool:
        return self.card_type == CardType.TAROK and self.value in (PAGAT, MOND, SKIS)

    @property
    def is_king(self) -> bool:
        return self.card_type == CardType.SUIT and self.value == SuitRank.KING.value

    @property
    def label(self) -> str:
        if self.card_type == CardType.TAROK:
            return _TAROK_ROMAN[self.value]
        assert self.suit is not None
        rank = SuitRank(self.value)
        if rank.value <= 4:
            is_red = self.suit in (Suit.HEARTS, Suit.DIAMONDS)
            pip_labels = _RED_PIP_LABELS if is_red else _BLACK_PIP_LABELS
            return f"{pip_labels[rank]}{SUIT_SYMBOLS[self.suit]}"
        return f"{_FACE_LABELS[rank]}{SUIT_SYMBOLS[self.suit]}"

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Sort key: taroks first by value, then suits grouped."""
        if self.card_type == CardType.TAROK:
            return (0, self.value, 0)
        assert self.suit is not None
        return (1, list(Suit).index(self.suit), self.value)

    def beats(self, other: Card, lead_suit: Suit | None) -> bool:
        """Does this card beat `other` given the lead suit?"""
        if self.card_type == CardType.TAROK and other.card_type == CardType.TAROK:
            # Škis always wins... except it's special (captured if played last trick)
            if self.value == SKIS:
                return True
            if other.value == SKIS:
                return False
            return self.value > other.value
        if self.card_type == CardType.TAROK:
            return True  # Tarok beats any suit card
        if other.card_type == CardType.TAROK:
            return False
        # Both suit cards
        if self.suit == other.suit:
            return self.value > other.value
        # Different suits — only the lead suit wins
        return self.suit == lead_suit

    def __repr__(self) -> str:
        return f"Card({self.label})"


def tarok(value: int) -> Card:
    """Create a tarok card."""
    assert 1 <= value <= 22
    return Card(CardType.TAROK, value)


def suit_card(suit: Suit, rank: SuitRank) -> Card:
    """Create a suit card."""
    return Card(CardType.SUIT, rank.value, suit)


def _build_deck() -> tuple[Card, ...]:
    cards: list[Card] = []
    for v in range(1, 23):
        cards.append(tarok(v))
    for s in Suit:
        for r in SuitRank:
            cards.append(suit_card(s, r))
    return tuple(cards)


DECK: tuple[Card, ...] = _build_deck()
assert len(DECK) == 54

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/entities/game_state.py`
```python
"""Game state domain entity.

Represents the full state machine for a 4-player Slovenian Tarok game.
Partners are determined by king calling (2v2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tarok.entities.card import Card, CardType, Suit, SuitRank, SKIS


class Phase(Enum):
    DEALING = "dealing"
    BIDDING = "bidding"
    KING_CALLING = "king_calling"
    TALON_EXCHANGE = "talon_exchange"
    ANNOUNCEMENTS = "announcements"
    TRICK_PLAY = "trick_play"
    SCORING = "scoring"
    FINISHED = "finished"


class Contract(Enum):
    """Contracts ordered by ascending strength (higher outbids lower)."""
    KLOP = -99  # All pass → each player for themselves, avoid taking points
    THREE = 3   # Take 3 talon cards
    TWO = 2     # Take 2 talon cards
    ONE = 1     # Take 1 talon card
    SOLO_THREE = -3  # Solo but pick 3 (no partner)
    SOLO_TWO = -2
    SOLO_ONE = -1
    SOLO = 0    # No talon, no partner

    @property
    def is_solo(self) -> bool:
        return self.value <= 0 and self != Contract.KLOP

    @property
    def is_klop(self) -> bool:
        return self == Contract.KLOP

    @property
    def talon_cards(self) -> int:
        if self == Contract.KLOP:
            return 0
        return abs(self.value)

    @property
    def strength(self) -> int:
        """Higher number = stronger bid. KLOP is not biddable."""
        order = {
            Contract.KLOP: 0,
            Contract.THREE: 1,
            Contract.TWO: 2,
            Contract.ONE: 3,
            Contract.SOLO_THREE: 4,
            Contract.SOLO_TWO: 5,
            Contract.SOLO_ONE: 6,
            Contract.SOLO: 7,
        }
        return order[self]

    @property
    def is_biddable(self) -> bool:
        """Can a player actively bid this contract?"""
        return self != Contract.KLOP


class PlayerRole(Enum):
    DECLARER = "declarer"
    PARTNER = "partner"      # Holds the called king
    OPPONENT = "opponent"


class Team(Enum):
    DECLARER_TEAM = "declarer_team"
    OPPONENT_TEAM = "opponent_team"


class Announcement(Enum):
    TRULA = "trula"          # Declarer team will collect all 3 trula cards
    KINGS = "kings"          # Declarer team will collect all 4 kings
    PAGAT_ULTIMO = "pagat_ultimo"  # Pagat played in the last trick and wins
    VALAT = "valat"          # Win all 12 tricks


class KontraLevel(Enum):
    """Counter-doubling levels on the base game or individual announcements.

    Opponents say KONTRA (double), declarers respond with RE (4×),
    opponents can respond with SUB/MORT (8×).
    """
    NONE = 1
    KONTRA = 2      # Opponents double
    RE = 4          # Declarers re-double
    SUB = 8         # Opponents re-re-double (sub-kontra / mort)

    @property
    def next_level(self) -> "KontraLevel | None":
        """The next escalation level, or None if maxed out."""
        _chain = {
            KontraLevel.NONE: KontraLevel.KONTRA,
            KontraLevel.KONTRA: KontraLevel.RE,
            KontraLevel.RE: KontraLevel.SUB,
            KontraLevel.SUB: None,
        }
        return _chain[self]

    @property
    def is_opponent_turn(self) -> bool:
        """Is it the opponents' turn to escalate?"""
        return self in (KontraLevel.NONE, KontraLevel.RE)


@dataclass
class Trick:
    lead_player: int
    cards: list[tuple[int, Card]] = field(default_factory=list)  # (player_idx, card)

    @property
    def lead_suit(self) -> Suit | None:
        if not self.cards:
            return None
        _, lead_card = self.cards[0]
        if lead_card.card_type == CardType.TAROK:
            return None  # Taroks don't establish a suit lead
        return lead_card.suit

    @property
    def is_complete(self) -> bool:
        return len(self.cards) == 4

    def winner(self) -> int:
        assert self.is_complete
        best_player, best_card = self.cards[0]
        for player, card in self.cards[1:]:
            if card.beats(best_card, self.lead_suit):
                best_player, best_card = player, card
        return best_player

    @property
    def points(self) -> int:
        return sum(c.points for _, c in self.cards)


@dataclass
class Bid:
    player: int
    contract: Contract | None  # None = pass


@dataclass
class GameState:
    """Immutable-ish representation of the full game state."""

    num_players: int = 4
    phase: Phase = Phase.DEALING

    # Cards
    hands: list[list[Card]] = field(default_factory=lambda: [[] for _ in range(4)])
    talon: list[Card] = field(default_factory=list)

    # Bidding
    bids: list[Bid] = field(default_factory=list)
    current_bidder: int = 0  # forehand (first to bid)
    declarer: int | None = None
    contract: Contract | None = None

    # King calling
    called_king: Card | None = None
    partner: int | None = None  # Revealed when king is played (or never for solo)

    # Talon exchange
    talon_revealed: list[list[Card]] = field(default_factory=list)
    put_down: list[Card] = field(default_factory=list)  # Cards declarer put back

    # Announcements
    announcements: dict[int, list[Announcement]] = field(default_factory=dict)
    # Kontra/Re/Sub levels: 'game' key for base contract, Announcement keys for bonuses
    kontra_levels: dict[str, KontraLevel] = field(default_factory=dict)

    # Trick play
    tricks: list[Trick] = field(default_factory=list)
    current_trick: Trick | None = None
    current_player: int = 0

    # Scoring
    declarer_team_tricks: list[Trick] = field(default_factory=list)
    opponent_team_tricks: list[Trick] = field(default_factory=list)
    scores: dict[int, int] = field(default_factory=dict)  # player -> score delta

    # Roles
    roles: dict[int, PlayerRole] = field(default_factory=dict)

    # Tracking
    round_number: int = 0
    dealer: int = 0

    def get_team(self, player: int) -> Team:
        role = self.roles.get(player, PlayerRole.OPPONENT)
        if role in (PlayerRole.DECLARER, PlayerRole.PARTNER):
            return Team.DECLARER_TEAM
        return Team.OPPONENT_TEAM

    def legal_plays(self, player: int) -> list[Card]:
        """Which cards can this player legally play in the current trick?"""
        hand = self.hands[player]
        if not hand:
            return []

        if self.current_trick is None or not self.current_trick.cards:
            return list(hand)  # Leading: can play anything

        lead_card = self.current_trick.cards[0][1]

        # Must follow suit if possible
        if lead_card.card_type == CardType.TAROK:
            taroks = [c for c in hand if c.card_type == CardType.TAROK]
            if taroks:
                return taroks
        else:
            same_suit = [c for c in hand if c.suit == lead_card.suit]
            if same_suit:
                return same_suit
            # Can't follow suit — must play tarok if possible
            taroks = [c for c in hand if c.card_type == CardType.TAROK]
            if taroks:
                return taroks

        # Can't follow suit or play tarok — play anything
        return list(hand)

    def legal_bids(self, player: int) -> list[Contract | None]:
        """Which bids can this player make? None = pass."""
        biddable = [c for c in Contract if c.is_biddable]

        if not self.bids:
            # First bidder can bid anything or pass
            return [None] + biddable

        # Find the current highest bid
        highest = max(
            (b.contract for b in self.bids if b.contract is not None),
            key=lambda c: c.strength,
            default=None,
        )

        options: list[Contract | None] = [None]  # Can always pass
        for c in biddable:
            if highest is None or c.strength > highest.strength:
                options.append(c)

        return options

    def callable_kings(self) -> list[Card]:
        """Which kings can the declarer call?"""
        assert self.declarer is not None
        hand = self.hands[self.declarer]
        all_kings = [
            Card(CardType.SUIT, SuitRank.KING.value, s) for s in Suit
        ]
        # Can call any king NOT in own hand
        callable = [k for k in all_kings if k not in hand]
        if not callable:
            # Has all 4 kings — can call a queen
            callable = [
                Card(CardType.SUIT, SuitRank.QUEEN.value, s)
                for s in Suit
                if Card(CardType.SUIT, SuitRank.QUEEN.value, s) not in hand
            ]
        return callable

    @property
    def is_partner_revealed(self) -> bool:
        if self.called_king is None:
            return False
        for trick in self.tricks:
            for _, card in trick.cards:
                if card == self.called_king:
                    return True
        if self.current_trick:
            for _, card in self.current_trick.cards:
                if card == self.called_king:
                    return True
        return False

    @property
    def tricks_played(self) -> int:
        return len(self.tricks)

    @property
    def is_last_trick(self) -> bool:
        return self.tricks_played == 11 and self.current_trick is not None

    def visible_state_for(self, player: int) -> dict:
        """Return the game state visible to a specific player (for AI observation)."""
        return {
            "phase": self.phase.value,
            "hand": list(self.hands[player]),
            "hand_size": [len(h) for h in self.hands],
            "talon_size": len(self.talon),
            "bids": [(b.player, b.contract.value if b.contract else None) for b in self.bids],
            "contract": self.contract.value if self.contract else None,
            "declarer": self.declarer,
            "called_king": self.called_king,
            "partner_revealed": self.is_partner_revealed,
            "partner": self.partner if self.is_partner_revealed else None,
            "current_trick": (
                [(p, c) for p, c in self.current_trick.cards]
                if self.current_trick
                else []
            ),
            "tricks_played": self.tricks_played,
            "my_role": self.roles.get(player, PlayerRole.OPPONENT).value,
            "played_cards": [
                (p, c) for trick in self.tricks for p, c in trick.cards
            ],
        }

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/entities/scoring.py`
```python
"""Scoring rules for Slovenian Tarok.

Cards are counted in groups of 3: (sum of 3 cards) - 2.
Total game points = 70. Declarer team wins with > 35 points (i.e. ≥ 36).
Differences are always computed from 35.
"""

from __future__ import annotations

from tarok.entities.card import Card, PAGAT, MOND, SKIS, CardType, SuitRank
from tarok.entities.game_state import (
    Announcement,
    Contract,
    GameState,
    KontraLevel,
    Team,
    Trick,
)

TOTAL_GAME_POINTS = 70
POINT_HALF = 35  # Diff is computed from 35; winning means > 35 (≥ 36)


def compute_card_points(cards: list[Card]) -> int:
    """Count card points using the 'groups of 3' method.

    Standard Tarok counting:
      groups of 3 → sum(points) − 2
      leftover 2  → sum(points) − 1
      leftover 1  → point value as-is
    Total over all 54 cards = 70.
    """
    raw = sum(c.points for c in cards)
    n = len(cards)
    deduction = (n // 3) * 2
    if n % 3 == 2:
        deduction += 1
    return raw - deduction


def _collect_team_cards(tricks: list[Trick], state: GameState, team: Team) -> list[Card]:
    """Collect all cards won by a team."""
    cards: list[Card] = []
    for trick in tricks:
        winner = trick.winner()
        if state.get_team(winner) == team:
            cards.extend(c for _, c in trick.cards)
    return cards


def _has_trula(cards: list[Card]) -> bool:
    tarok_values = {c.value for c in cards if c.card_type == CardType.TAROK}
    return {PAGAT, MOND, SKIS}.issubset(tarok_values)


def _has_all_kings(cards: list[Card]) -> bool:
    kings = [c for c in cards if c.is_king]
    return len(kings) == 4


def _pagat_ultimo(tricks: list[Trick], team: Team, state: GameState) -> bool:
    """Did the team win the last trick with Pagat?"""
    if not tricks:
        return False
    last_trick = tricks[-1]
    winner = last_trick.winner()
    if state.get_team(winner) != team:
        return False
    # Check if Pagat was in the last trick played by the winning team
    for player, card in last_trick.cards:
        if card.card_type == CardType.TAROK and card.value == PAGAT:
            return state.get_team(player) == team
    return False


def _contract_multiplier(contract: Contract) -> int:
    """Base game value per the official table (docs/basics.md).

    tri=10, dva=20, ena=30
    solo tri=20, solo dva=30, solo ena=40, solo brez talona=50
    """
    return {
        Contract.KLOP: 0,
        Contract.THREE: 10,
        Contract.TWO: 20,
        Contract.ONE: 30,
        Contract.SOLO_THREE: 20,
        Contract.SOLO_TWO: 30,
        Contract.SOLO_ONE: 40,
        Contract.SOLO: 50,
    }[contract]


# --- Silent (tihi) bonus values ---
_SILENT_TRULA = 10
_SILENT_KINGS = 10
_SILENT_PAGAT_ULTIMO = 25
# --- Announced (napovedani) bonus values ---
_ANNOUNCED_TRULA = 20
_ANNOUNCED_KINGS = 20
_ANNOUNCED_PAGAT_ULTIMO = 50
_ANNOUNCED_VALAT = 500
_SILENT_VALAT = 250


def _score_klop(state: GameState) -> dict[int, int]:
    """Score a klop game (all players passed).

    Each player plays for themselves. Goal: avoid taking card points.
    Score = -(counted card points captured). If one player wins all tricks,
    they get +70 instead (klop valat).
    """
    # Collect cards per player
    player_cards: dict[int, list[Card]] = {p: [] for p in range(state.num_players)}
    for trick in state.tricks:
        winner = trick.winner()
        player_cards[winner].extend(c for _, c in trick.cards)

    # Check for klop valat: one player won all tricks
    all_tricks = state.tricks
    if all_tricks and all(t.winner() == all_tricks[0].winner() for t in all_tricks):
        valat_player = all_tricks[0].winner()
        others = [p for p in range(state.num_players) if p != valat_player]
        scores: dict[int, int] = {}
        per_other = TOTAL_GAME_POINTS // len(others)
        remainder = TOTAL_GAME_POINTS % len(others)
        for i, p in enumerate(others):
            scores[p] = -(per_other + (1 if i < remainder else 0))
        scores[valat_player] = -sum(scores[p] for p in others)
        return scores

    # Normal klop: each player loses their counted card points
    # Counting in groups-of-3 is not additive, so we adjust rounding to keep zero-sum
    raw_scores = {p: -compute_card_points(cards) for p, cards in player_cards.items()}
    total = sum(raw_scores.values())
    if total != 0:
        # Spread rounding error to player(s) who captured the most points
        worst = max(raw_scores, key=lambda p: -raw_scores[p])  # most negative
        raw_scores[worst] -= total
    return raw_scores


def _get_kontra(state: GameState, key: str) -> int:
    """Return the kontra multiplier for a given target ('game' or announcement name)."""
    level = state.kontra_levels.get(key, KontraLevel.NONE)
    return level.value


def score_game(state: GameState) -> dict[int, int]:
    """Compute final scores for all players. Returns player_idx -> point delta."""
    assert state.contract is not None

    if state.contract.is_klop:
        return _score_klop(state)

    assert state.declarer is not None

    all_tricks = state.tricks
    declarer_cards: list[Card] = []
    opponent_cards: list[Card] = []

    for trick in all_tricks:
        winner = trick.winner()
        team = state.get_team(winner)
        for _, card in trick.cards:
            if team == Team.DECLARER_TEAM:
                declarer_cards.append(card)
            else:
                opponent_cards.append(card)

    # Add put-down cards to declarer's pile
    declarer_cards.extend(state.put_down)

    declarer_points = compute_card_points(declarer_cards)
    declarer_won = declarer_points > POINT_HALF  # > 35 means ≥ 36

    # Base game score: contract value + |difference from 35|
    point_diff = abs(declarer_points - POINT_HALF)
    base_score = _contract_multiplier(state.contract) + point_diff

    if not declarer_won:
        base_score = -base_score

    # Apply kontra/re/sub to the base game
    base_score *= _get_kontra(state, "game")

    # --- Bonus scoring (silent & announced) ---
    bonus = 0

    # Collect which announcements were made and by which team
    announced_by_team: dict[Announcement, Team] = {}
    for player, announcements in state.announcements.items():
        team = state.get_team(player)
        for ann in announcements:
            announced_by_team[ann] = team

    # Trula
    decl_has_trula = _has_trula(declarer_cards)
    opp_has_trula = _has_trula(opponent_cards)
    trula_bonus = 0
    if Announcement.TRULA in announced_by_team:
        ann_team = announced_by_team[Announcement.TRULA]
        if ann_team == Team.DECLARER_TEAM:
            trula_bonus = _ANNOUNCED_TRULA if decl_has_trula else -_ANNOUNCED_TRULA
        else:
            trula_bonus = -_ANNOUNCED_TRULA if opp_has_trula else _ANNOUNCED_TRULA
        trula_bonus *= _get_kontra(state, Announcement.TRULA.value)
    else:
        if decl_has_trula:
            trula_bonus = _SILENT_TRULA
        elif opp_has_trula:
            trula_bonus = -_SILENT_TRULA
    bonus += trula_bonus

    # Kings
    decl_has_kings = _has_all_kings(declarer_cards)
    opp_has_kings = _has_all_kings(opponent_cards)
    kings_bonus = 0
    if Announcement.KINGS in announced_by_team:
        ann_team = announced_by_team[Announcement.KINGS]
        if ann_team == Team.DECLARER_TEAM:
            kings_bonus = _ANNOUNCED_KINGS if decl_has_kings else -_ANNOUNCED_KINGS
        else:
            kings_bonus = -_ANNOUNCED_KINGS if opp_has_kings else _ANNOUNCED_KINGS
        kings_bonus *= _get_kontra(state, Announcement.KINGS.value)
    else:
        if decl_has_kings:
            kings_bonus = _SILENT_KINGS
        elif opp_has_kings:
            kings_bonus = -_SILENT_KINGS
    bonus += kings_bonus

    # Pagat ultimo
    decl_pagat = _pagat_ultimo(all_tricks, Team.DECLARER_TEAM, state)
    opp_pagat = _pagat_ultimo(all_tricks, Team.OPPONENT_TEAM, state)
    pagat_bonus = 0
    if Announcement.PAGAT_ULTIMO in announced_by_team:
        ann_team = announced_by_team[Announcement.PAGAT_ULTIMO]
        if ann_team == Team.DECLARER_TEAM:
            pagat_bonus = _ANNOUNCED_PAGAT_ULTIMO if decl_pagat else -_ANNOUNCED_PAGAT_ULTIMO
        else:
            pagat_bonus = -_ANNOUNCED_PAGAT_ULTIMO if opp_pagat else _ANNOUNCED_PAGAT_ULTIMO
        pagat_bonus *= _get_kontra(state, Announcement.PAGAT_ULTIMO.value)
    else:
        if decl_pagat:
            pagat_bonus = _SILENT_PAGAT_ULTIMO
        elif opp_pagat:
            pagat_bonus = -_SILENT_PAGAT_ULTIMO
    bonus += pagat_bonus

    # Valat
    valat_bonus = 0
    if Announcement.VALAT in announced_by_team:
        ann_team = announced_by_team[Announcement.VALAT]
        all_won = all(state.get_team(t.winner()) == ann_team for t in all_tricks)
        if ann_team == Team.DECLARER_TEAM:
            valat_bonus = _ANNOUNCED_VALAT if all_won else -_ANNOUNCED_VALAT
        else:
            valat_bonus = -_ANNOUNCED_VALAT if all_won else _ANNOUNCED_VALAT
        valat_bonus *= _get_kontra(state, Announcement.VALAT.value)
    else:
        # Silent valat
        decl_all = all(state.get_team(t.winner()) == Team.DECLARER_TEAM for t in all_tricks)
        opp_all = all(state.get_team(t.winner()) == Team.OPPONENT_TEAM for t in all_tricks)
        if decl_all:
            valat_bonus = _SILENT_VALAT
        elif opp_all:
            valat_bonus = -_SILENT_VALAT
    bonus += valat_bonus

    total_declarer = base_score + bonus

    # Determine if it's effectively solo (solo contract OR king in talon)
    effectively_solo = state.contract.is_solo or state.partner is None

    # Distribute scores
    scores: dict[int, int] = {}
    for p in range(state.num_players):
        team = state.get_team(p)
        if effectively_solo:
            # Solo / king-in-talon: declarer gets/loses 3× against each opponent
            if p == state.declarer:
                scores[p] = total_declarer * 3
            else:
                scores[p] = -total_declarer
        else:
            # 2v2: each player on a team gets the same
            if team == Team.DECLARER_TEAM:
                scores[p] = total_declarer
            else:
                scores[p] = -total_declarer

    return scores

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/ports/__init__.py`
```python
from tarok.ports.player_port import PlayerPort
from tarok.ports.game_repo_port import GameRepoPort
from tarok.ports.observer_port import GameObserverPort

__all__ = ["PlayerPort", "GameRepoPort", "GameObserverPort"]

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/ports/game_repo_port.py`
```python
"""Port for game state persistence."""

from __future__ import annotations

from typing import Protocol

from tarok.entities.game_state import GameState


class GameRepoPort(Protocol):
    async def save(self, game_id: str, state: GameState) -> None: ...
    async def load(self, game_id: str) -> GameState | None: ...
    async def list_games(self) -> list[str]: ...

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/ports/observer_port.py`
```python
"""Port for observing game events (UI updates, metrics collection)."""

from __future__ import annotations

from typing import Protocol

from tarok.entities.card import Card
from tarok.entities.game_state import Contract, GameState, Trick


class GameObserverPort(Protocol):
    """Notified about game events — implemented by WebSocket broadcaster, training metrics, etc."""

    async def on_game_start(self, state: GameState) -> None: ...
    async def on_deal(self, state: GameState) -> None: ...
    async def on_bid(self, player: int, bid: Contract | None, state: GameState) -> None: ...
    async def on_contract_won(self, player: int, contract: Contract, state: GameState) -> None: ...
    async def on_king_called(self, player: int, king: Card, state: GameState) -> None: ...
    async def on_talon_revealed(self, groups: list[list[Card]], state: GameState) -> None: ...
    async def on_talon_exchanged(self, state: GameState) -> None: ...
    async def on_card_played(self, player: int, card: Card, state: GameState) -> None: ...
    async def on_trick_won(self, trick: Trick, winner: int, state: GameState) -> None: ...
    async def on_game_end(self, scores: dict[int, int], state: GameState) -> None: ...

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/ports/player_port.py`
```python
"""Port for player decision-making.

Implemented by: HumanPlayer (via WebSocket), AIPlayer (RL agent), RandomPlayer.
"""

from __future__ import annotations

from typing import Protocol

from tarok.entities.card import Card
from tarok.entities.game_state import Contract, GameState, Announcement


class PlayerPort(Protocol):
    """Interface for any player (human, AI, random)."""

    @property
    def name(self) -> str: ...

    async def choose_bid(
        self,
        state: GameState,
        player_idx: int,
        legal_bids: list[Contract | None],
    ) -> Contract | None:
        """Choose a bid or pass (None)."""
        ...

    async def choose_king(
        self,
        state: GameState,
        player_idx: int,
        callable_kings: list[Card],
    ) -> Card:
        """Choose which king to call."""
        ...

    async def choose_talon_group(
        self,
        state: GameState,
        player_idx: int,
        talon_groups: list[list[Card]],
    ) -> int:
        """Choose which talon group to pick up (index)."""
        ...

    async def choose_discard(
        self,
        state: GameState,
        player_idx: int,
        must_discard: int,
    ) -> list[Card]:
        """Choose which cards to put down after picking up talon."""
        ...

    async def choose_announcements(
        self,
        state: GameState,
        player_idx: int,
    ) -> list[Announcement]:
        """Choose announcements (can be empty)."""
        ...

    async def choose_announce_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> int:
        """Choose a single announcement/kontra action.

        0=pass, 1=trula, 2=kings, 3=pagat, 4=valat,
        5=kontra_game, 6=kontra_trula, 7=kontra_kings, 8=kontra_pagat, 9=kontra_valat.
        Called repeatedly until the player passes.
        """
        ...

    async def choose_card(
        self,
        state: GameState,
        player_idx: int,
        legal_plays: list[Card],
    ) -> Card:
        """Choose a card to play in the current trick."""
        ...

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/use_cases/__init__.py`
```python
from tarok.use_cases.game_loop import GameLoop

__all__ = ["GameLoop"]

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/use_cases/bid.py`
```python
"""Bidding use case — players bid ascending contracts or pass."""

from __future__ import annotations

from tarok.entities.game_state import Bid, Contract, GameState, Phase, PlayerRole


def place_bid(state: GameState, player: int, contract: Contract | None) -> GameState:
    """Player places a bid or passes. Returns updated state."""
    assert state.phase == Phase.BIDDING
    assert player == state.current_bidder

    state.bids.append(Bid(player=player, contract=contract))

    # Check if bidding is over
    if _bidding_complete(state):
        _resolve_bidding(state)
    else:
        state.current_bidder = _next_bidder(state)
        state.current_player = state.current_bidder

    return state


def _next_bidder(state: GameState) -> int:
    """Find next player who hasn't passed."""
    passed = {b.player for b in state.bids if b.contract is None}
    start = (state.current_bidder + 1) % state.num_players
    for i in range(state.num_players):
        candidate = (start + i) % state.num_players
        if candidate not in passed:
            return candidate
    return state.current_bidder  # Shouldn't reach here


def _bidding_complete(state: GameState) -> bool:
    """Bidding ends when 3 players have passed (one winner) or all passed."""
    passed = {b.player for b in state.bids if b.contract is None}
    active_bidders = set(range(state.num_players)) - passed

    # Everyone passed
    if len(passed) == state.num_players:
        return True

    # Only one bidder remains and at least one full round completed
    if len(active_bidders) == 1 and len(state.bids) >= state.num_players:
        return True

    # All 4 have acted at least once, and 3 have passed
    if len(passed) >= state.num_players - 1:
        return True

    return False


def _resolve_bidding(state: GameState) -> None:
    """Determine the winner and contract."""
    bids_with_contract = [b for b in state.bids if b.contract is not None]

    if not bids_with_contract:
        # Everyone passed → klop
        state.contract = Contract.KLOP
        state.phase = Phase.TRICK_PLAY
        # No declarer in klop — everyone plays for themselves
        for p in range(state.num_players):
            state.roles[p] = PlayerRole.OPPONENT
        state.current_player = (state.dealer + 1) % state.num_players
        return

    # Highest bid wins
    winning_bid = max(bids_with_contract, key=lambda b: b.contract.strength)  # type: ignore
    state.declarer = winning_bid.player
    state.contract = winning_bid.contract
    state.current_player = state.declarer

    # Set declarer role
    state.roles[state.declarer] = PlayerRole.DECLARER

    if state.contract.is_solo:
        # Solo: no partner, all others are opponents
        for p in range(state.num_players):
            if p != state.declarer:
                state.roles[p] = PlayerRole.OPPONENT
        if state.contract.talon_cards > 0:
            state.phase = Phase.TALON_EXCHANGE
        else:
            state.phase = Phase.ANNOUNCEMENTS
    else:
        state.phase = Phase.KING_CALLING

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/use_cases/call_king.py`
```python
"""King calling use case — declarer names a king to find a partner."""

from __future__ import annotations

from tarok.entities.card import Card
from tarok.entities.game_state import GameState, Phase, PlayerRole


def call_king(state: GameState, king: Card) -> GameState:
    assert state.phase == Phase.KING_CALLING
    assert state.declarer is not None
    assert king in state.callable_kings()

    state.called_king = king

    # Find partner (secret until king is played)
    for p in range(state.num_players):
        if p == state.declarer:
            continue
        if king in state.hands[p]:
            state.partner = p
            state.roles[p] = PlayerRole.PARTNER
            break

    # If called king is in talon, declarer plays alone
    if state.partner is None:
        for p in range(state.num_players):
            if p != state.declarer:
                state.roles[p] = PlayerRole.OPPONENT

    # Set remaining players as opponents
    for p in range(state.num_players):
        if p not in state.roles:
            state.roles[p] = PlayerRole.OPPONENT

    state.phase = Phase.TALON_EXCHANGE
    state.current_player = state.declarer
    return state

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/use_cases/deal.py`
```python
"""Deal use case — shuffle and distribute 54 cards."""

from __future__ import annotations

import random

from tarok.entities.card import DECK, Card
from tarok.entities.game_state import GameState, Phase

CARDS_PER_PLAYER = 12
TALON_SIZE = 6


def deal(state: GameState, rng: random.Random | None = None) -> GameState:
    """Deal cards to 4 players + talon. Returns new state."""
    assert state.phase == Phase.DEALING

    rng = rng or random.Random()
    cards = list(DECK)
    rng.shuffle(cards)

    hands: list[list[Card]] = []
    idx = 0
    for _ in range(state.num_players):
        hands.append(sorted(cards[idx : idx + CARDS_PER_PLAYER], key=lambda c: c.sort_key))
        idx += CARDS_PER_PLAYER

    talon = cards[idx : idx + TALON_SIZE]
    assert idx + TALON_SIZE == len(cards)

    state.hands = hands
    state.talon = talon
    state.phase = Phase.BIDDING
    # First bidder is the player after the dealer (forehand)
    state.current_bidder = (state.dealer + 1) % state.num_players
    state.current_player = state.current_bidder
    return state

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/use_cases/exchange_talon.py`
```python
"""Talon exchange use case — uses talon strategy plugins."""

from __future__ import annotations

from tarok.entities.card import Card, CardType, SuitRank
from tarok.entities.game_state import GameState, Phase


def reveal_talon(state: GameState) -> list[list[Card]]:
    """Split the talon into groups based on contract."""
    assert state.contract is not None
    n = state.contract.talon_cards
    if n == 0:
        return []

    group_size = 6 // (6 // n) if n in (1, 2, 3) else n
    # Three: 2 groups of 3
    # Two: 3 groups of 2
    # One: 6 groups of 1
    groups: list[list[Card]] = []
    for i in range(0, 6, group_size):
        groups.append(state.talon[i : i + group_size])

    state.talon_revealed = groups
    return groups


def pick_talon_group(state: GameState, group_idx: int) -> GameState:
    """Declarer picks up a talon group."""
    assert state.phase == Phase.TALON_EXCHANGE
    assert state.declarer is not None
    assert 0 <= group_idx < len(state.talon_revealed)

    picked = state.talon_revealed[group_idx]
    state.hands[state.declarer].extend(picked)
    return state


def discard_cards(state: GameState, cards: list[Card]) -> GameState:
    """Declarer discards cards back (cannot discard kings or taroks, with exceptions)."""
    assert state.phase == Phase.TALON_EXCHANGE
    assert state.declarer is not None
    assert state.contract is not None
    assert len(cards) == state.contract.talon_cards

    for card in cards:
        # Cannot discard kings
        assert not card.is_king, f"Cannot discard a king: {card}"
        # Cannot discard taroks (unless hand is all taroks + kings)
        if card.card_type == CardType.TAROK:
            non_tarok_non_king = [
                c for c in state.hands[state.declarer]
                if c.card_type != CardType.TAROK and not c.is_king
            ]
            assert len(non_tarok_non_king) == 0, "Cannot discard taroks if you have suit cards"

    hand = state.hands[state.declarer]
    for card in cards:
        hand.remove(card)
    state.hands[state.declarer] = sorted(hand, key=lambda c: c.sort_key)
    state.put_down = cards

    state.phase = Phase.ANNOUNCEMENTS
    return state

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/use_cases/game_loop.py`
```python
"""Main game loop use case — orchestrates a full game from deal to score.

This is the core orchestrator that depends on ports (PlayerPort, GameObserverPort)
but knows nothing about adapters (WebSocket, RL agents, etc.).
"""

from __future__ import annotations

import random

from tarok.entities.game_state import GameState, KontraLevel, Phase, Announcement
from tarok.entities.scoring import score_game
from tarok.ports.player_port import PlayerPort
from tarok.ports.observer_port import GameObserverPort
from tarok.use_cases.deal import deal
from tarok.use_cases.bid import place_bid
from tarok.use_cases.call_king import call_king
from tarok.use_cases.exchange_talon import reveal_talon, pick_talon_group, discard_cards
from tarok.use_cases.play_trick import start_trick, play_card


class NullObserver:
    """Default observer that does nothing."""

    async def on_game_start(self, state): pass
    async def on_deal(self, state): pass
    async def on_bid(self, player, bid, state): pass
    async def on_contract_won(self, player, contract, state): pass
    async def on_king_called(self, player, king, state): pass
    async def on_talon_revealed(self, groups, state): pass
    async def on_talon_exchanged(self, state): pass
    async def on_card_played(self, player, card, state): pass
    async def on_trick_won(self, trick, winner, state): pass
    async def on_game_end(self, scores, state): pass


class GameLoop:
    """Runs a single game of Tarok end-to-end."""

    def __init__(
        self,
        players: list[PlayerPort],
        observer: GameObserverPort | None = None,
        rng: random.Random | None = None,
    ):
        assert len(players) == 4
        self._players = players
        self._observer: GameObserverPort = observer or NullObserver()  # type: ignore
        self._rng = rng or random.Random()

    async def run(self, dealer: int = 0) -> tuple[GameState, dict[int, int]]:
        """Play one full game. Returns (final_state, scores)."""
        state = GameState(dealer=dealer)
        await self._observer.on_game_start(state)

        # === DEAL ===
        state.phase = Phase.DEALING
        state = deal(state, self._rng)
        await self._observer.on_deal(state)

        # === BIDDING ===
        max_rounds = 20  # Safety limit
        rounds = 0
        while state.phase == Phase.BIDDING and rounds < max_rounds:
            player_idx = state.current_bidder
            legal = state.legal_bids(player_idx)
            bid = await self._players[player_idx].choose_bid(state, player_idx, legal)
            assert bid is None or bid in legal
            state = place_bid(state, player_idx, bid)
            await self._observer.on_bid(player_idx, bid, state)
            rounds += 1

        if state.phase == Phase.DEALING:
            # Everyone passed — re-deal
            # (This shouldn't happen now that klop exists, but keep as safety)
            return await self.run(dealer=(dealer + 1) % 4)

        assert state.contract is not None

        # Klop: skip king/talon/announcements, go straight to tricks
        if state.contract.is_klop:
            await self._observer.on_contract_won(-1, state.contract, state)
        else:
            assert state.declarer is not None
            await self._observer.on_contract_won(state.declarer, state.contract, state)

        # === KING CALLING (if not solo and not klop) ===
        if state.phase == Phase.KING_CALLING:
            callable = state.callable_kings()
            if callable:
                king = await self._players[state.declarer].choose_king(
                    state, state.declarer, callable
                )
                state = call_king(state, king)
                await self._observer.on_king_called(state.declarer, king, state)
            else:
                state.phase = Phase.TALON_EXCHANGE

        # === TALON EXCHANGE ===
        if state.phase == Phase.TALON_EXCHANGE and state.contract.talon_cards > 0:
            groups = reveal_talon(state)
            await self._observer.on_talon_revealed(groups, state)

            group_idx = await self._players[state.declarer].choose_talon_group(
                state, state.declarer, groups
            )
            state = pick_talon_group(state, group_idx)

            discards = await self._players[state.declarer].choose_discard(
                state, state.declarer, state.contract.talon_cards
            )
            state = discard_cards(state, discards)
            await self._observer.on_talon_exchanged(state)
        elif state.phase == Phase.TALON_EXCHANGE:
            state.phase = Phase.ANNOUNCEMENTS

        # === ANNOUNCEMENTS + KONTRA/RE/SUB ===
        if state.phase == Phase.ANNOUNCEMENTS:
            # Round-robin: each player can announce or kontra, repeat until all pass.
            # Max rounds to prevent infinite loops (5 announcement types × 4 kontra levels = bounded).
            MAX_ROUNDS = 20
            for _round in range(MAX_ROUNDS):
                any_action = False
                for p in range(state.num_players):
                    if not hasattr(self._players[p], 'choose_announce_action'):
                        # Legacy path: player doesn't support the new protocol
                        continue
                    action = await self._players[p].choose_announce_action(state, p)
                    if action == 0:  # PASS
                        continue

                    any_action = True

                    # Import action constants (avoid circular)
                    from tarok.adapters.ai.encoding import (
                        ANNOUNCE_IDX_TO_ANN,
                        KONTRA_IDX_TO_KEY,
                    )

                    if action in ANNOUNCE_IDX_TO_ANN:
                        ann = ANNOUNCE_IDX_TO_ANN[action]
                        if p not in state.announcements:
                            state.announcements[p] = []
                        if ann not in state.announcements[p]:
                            state.announcements[p].append(ann)
                    elif action in KONTRA_IDX_TO_KEY:
                        key = KONTRA_IDX_TO_KEY[action]
                        current = state.kontra_levels.get(key, KontraLevel.NONE)
                        next_level = current.next_level
                        if next_level is not None:
                            state.kontra_levels[key] = next_level

                if not any_action:
                    break

            state.phase = Phase.TRICK_PLAY
            # Forehand leads first trick
            state.current_player = (state.dealer + 1) % state.num_players

        # === TRICK PLAY ===
        while state.phase == Phase.TRICK_PLAY:
            state = start_trick(state)
            for _ in range(state.num_players):
                player_idx = state.current_player
                legal = state.legal_plays(player_idx)
                card = await self._players[player_idx].choose_card(
                    state, player_idx, legal
                )
                state = play_card(state, player_idx, card)
                await self._observer.on_card_played(player_idx, card, state)

            if state.tricks:
                last_trick = state.tricks[-1]
                winner = last_trick.winner()
                await self._observer.on_trick_won(last_trick, winner, state)

        # === SCORING ===
        scores = score_game(state)
        state.scores = scores
        state.phase = Phase.FINISHED
        await self._observer.on_game_end(scores, state)

        return state, scores

```

### File: `/Users/swozny/work/tarok/backend/src/tarok/use_cases/play_trick.py`
```python
"""Trick play use case — individual card plays and trick resolution."""

from __future__ import annotations

from tarok.entities.card import Card
from tarok.entities.game_state import GameState, Phase, Trick


def start_trick(state: GameState) -> GameState:
    """Begin a new trick led by current_player."""
    assert state.phase == Phase.TRICK_PLAY
    state.current_trick = Trick(lead_player=state.current_player)
    return state


def play_card(state: GameState, player: int, card: Card) -> GameState:
    """Play a card into the current trick."""
    assert state.phase == Phase.TRICK_PLAY
    assert state.current_trick is not None
    assert player == state.current_player
    assert card in state.legal_plays(player), (
        f"Illegal play: {card} not in {state.legal_plays(player)}"
    )

    state.hands[player].remove(card)
    state.current_trick.cards.append((player, card))

    if state.current_trick.is_complete:
        _resolve_trick(state)
    else:
        state.current_player = (player + 1) % state.num_players

    return state


def _resolve_trick(state: GameState) -> None:
    """Resolve completed trick — determine winner and start next or end game."""
    assert state.current_trick is not None
    assert state.current_trick.is_complete

    winner = state.current_trick.winner()
    state.tricks.append(state.current_trick)
    state.current_trick = None
    state.current_player = winner

    if state.tricks_played == 12:
        state.phase = Phase.SCORING

```

### File: `/Users/swozny/work/tarok/frontend/src/App.tsx`
```tsx
import React, { useState } from 'react';
import GameBoard from './components/GameBoard';
import GameLog from './components/GameLog';
import TrainingDashboard from './components/TrainingDashboard';
import CameraAgent from './components/CameraAgent';
import { useGame } from './hooks/useGame';
import type { CardData } from './types/game';
import './App.css';

type Page = 'home' | 'training' | 'play' | 'camera';

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const game = useGame();

  const handleStartGame = async () => {
    await game.startNewGame();
    setPage('play');
  };

  if (page === 'training') {
    return <TrainingDashboard onBack={() => setPage('home')} />;
  }

  if (page === 'camera') {
    return <CameraAgent onBack={() => setPage('home')} />;
  }

  if (page === 'play') {
    return (
      <div className="app">
        <div className="app-bar">
          <button className="btn-secondary btn-sm" onClick={() => setPage('home')}>← Menu</button>
          <span className="connection-status">
            {game.connected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
        </div>
        <div className="play-layout">
          <div className="play-main">
            <GameBoard
              state={game.gameState}
              onPlayCard={(card: CardData) => game.playCard(card)}
              onBid={(contract) => game.bid(contract)}
              onCallKing={(suit) => game.callKing(suit)}
              onChooseTalon={(idx) => game.chooseTalon(idx)}
              onDiscard={(cards: CardData[]) => game.discard(cards)}
            />
          </div>
          <GameLog entries={game.logEntries} />
        </div>
      </div>
    );
  }

  // Home page
  return (
    <div className="app">
      <div className="home-page">
        <div className="hero">
          <h1 className="title">Slovenian Tarok</h1>
          <p className="subtitle">A 4-player trick-taking card game with AI agents that learn by self-play</p>

          <div className="hero-cards">
            <div className="hero-card hero-card-1">★</div>
            <div className="hero-card hero-card-2">♠</div>
            <div className="hero-card hero-card-3">XXI</div>
          </div>
        </div>

        <div className="home-actions">
          <button className="btn-gold btn-large" onClick={() => setPage('training')}>
            <span className="btn-icon">🧠</span>
            <span>
              <strong>Train AI Agents</strong>
              <small>Watch agents learn through self-play</small>
            </span>
          </button>

          <button className="btn-primary btn-large" onClick={handleStartGame}>
            <span className="btn-icon">🃏</span>
            <span>
              <strong>Play vs AI</strong>
              <small>Challenge the trained agents</small>
            </span>
          </button>

          <button className="btn-secondary btn-large" onClick={() => setPage('camera')}>
            <span className="btn-icon">📸</span>
            <span>
              <strong>Camera Agent</strong>
              <small>Get AI advice for a real-world hand</small>
            </span>
          </button>
        </div>

        <div className="rules-summary">
          <h3>How it works</h3>
          <div className="rules-grid">
            <div className="rule-card">
              <div className="rule-icon">📦</div>
              <h4>54 Cards</h4>
              <p>22 Taroks (trumps) + 32 suit cards across 4 suits</p>
            </div>
            <div className="rule-card">
              <div className="rule-icon">👥</div>
              <h4>2v2 Teams</h4>
              <p>Declarer calls a king — the holder becomes their secret partner</p>
            </div>
            <div className="rule-card">
              <div className="rule-icon">🎯</div>
              <h4>12 Tricks</h4>
              <p>Win tricks to collect card points. Team with 36+ points wins</p>
            </div>
            <div className="rule-card">
              <div className="rule-icon">🤖</div>
              <h4>Self-Play RL</h4>
              <p>AI agents learn optimal play through PPO deep reinforcement learning</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

```

### File: `/Users/swozny/work/tarok/frontend/src/components/BiddingPanel.tsx`
```tsx
import React from 'react';
import type { CardData } from '../types/game';
import { CONTRACT_NAMES, SUIT_SYMBOLS } from '../types/game';
import './BiddingPanel.css';

interface BiddingPanelProps {
  phase: string;
  bids: { player: number; contract: number | null }[];
  legalBids?: (number | null)[];
  onBid: (contract: number | null) => void;
  playerNames: string[];
  callableKings?: CardData[];
  onCallKing?: (suit: string) => void;
}

const BID_OPTIONS = [
  { value: 3, label: 'Three', description: 'Pick 3 talon cards (2v2)' },
  { value: 2, label: 'Two', description: 'Pick 2 talon cards (2v2)' },
  { value: 1, label: 'One', description: 'Pick 1 talon card (2v2)' },
  { value: 0, label: 'Solo', description: 'No talon, play alone (1v3)' },
];

export default function BiddingPanel({
  phase, bids, legalBids, onBid, playerNames, callableKings, onCallKing,
}: BiddingPanelProps) {
  if (phase === 'king_calling' && callableKings && onCallKing) {
    return (
      <div className="bidding-panel">
        <h3>Call a King</h3>
        <p className="bidding-subtitle">Choose which king to call — the holder becomes your partner</p>
        <div className="king-options">
          {callableKings.map(king => (
            <button
              key={king.suit}
              className="btn-gold king-btn"
              onClick={() => onCallKing(king.suit!)}
            >
              {SUIT_SYMBOLS[king.suit!]} King of {king.suit}
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (phase !== 'bidding') return null;

  return (
    <div className="bidding-panel">
      <h3>Bidding</h3>

      {bids.length > 0 && (
        <div className="bid-history">
          {bids.map((bid, i) => (
            <div key={i} className="bid-entry">
              <span className="bid-player">{playerNames[bid.player] || `P${bid.player}`}</span>
              <span className="bid-value">
                {bid.contract !== null ? CONTRACT_NAMES[bid.contract] || `${bid.contract}` : 'Pass'}
              </span>
            </div>
          ))}
        </div>
      )}

      {legalBids && (
        <div className="bid-actions">
          <button className="btn-secondary" data-testid="bid-pass" onClick={() => onBid(null)}>
            Pass
          </button>
          {BID_OPTIONS.filter(opt => legalBids.includes(opt.value)).map(opt => (
            <button key={opt.value} className="btn-primary bid-btn" onClick={() => onBid(opt.value)}>
              <span className="bid-btn-label">{opt.label}</span>
              <span className="bid-btn-desc">{opt.description}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

```

### File: `/Users/swozny/work/tarok/frontend/src/components/CameraAgent.tsx`
```tsx
import React, { useState, useRef } from 'react';
import type { CardData } from '../types/game';
import Card from './Card';
import { SUIT_SYMBOLS } from '../types/game';
import './CameraAgent.css';

interface RankedPlay {
  card: CardData;
  probability: number;
}

interface AnalysisResult {
  recommended: CardData;
  legal_plays: CardData[];
  ranked_plays: RankedPlay[];
  position_value: number | null;
  has_trained_model: boolean;
}

const ALL_TAROKS: CardData[] = Array.from({ length: 22 }, (_, i) => ({
  card_type: 'tarok' as const,
  value: i + 1,
  suit: null,
  label: i + 1 === 22 ? 'Škis' : ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX','XXI'][i],
  points: [1, 21, 22].includes(i + 1) ? 5 : 1,
}));

const SUITS: ('hearts' | 'diamonds' | 'clubs' | 'spades')[] = ['hearts', 'diamonds', 'clubs', 'spades'];
const SUIT_RANKS = [
  { value: 8, label: 'K', points: 5 },
  { value: 7, label: 'Q', points: 4 },
  { value: 6, label: 'C', points: 3 },
  { value: 5, label: 'J', points: 2 },
  { value: 4, label: '4', points: 1 },
  { value: 3, label: '3', points: 1 },
  { value: 2, label: '2', points: 1 },
  { value: 1, label: '1', points: 1 },
];

function makeSuitCard(suit: string, rank: typeof SUIT_RANKS[0]): CardData {
  const isRed = suit === 'hearts' || suit === 'diamonds';
  const pipLabels: Record<number, string> = isRed
    ? { 1: '1', 2: '2', 3: '3', 4: '4' }
    : { 1: '7', 2: '8', 3: '9', 4: '10' };
  const faceLabels: Record<number, string> = { 5: 'J', 6: 'C', 7: 'Q', 8: 'K' };
  const sym = SUIT_SYMBOLS[suit as keyof typeof SUIT_SYMBOLS] || suit;
  const lbl = rank.value <= 4 ? pipLabels[rank.value] : faceLabels[rank.value];
  return {
    card_type: 'suit',
    value: rank.value,
    suit: suit as CardData['suit'],
    label: `${lbl}${sym}`,
    points: rank.points,
  };
}

const CONTRACTS: { id: string; name: string }[] = [
  { id: 'three', name: 'Three' },
  { id: 'two', name: 'Two' },
  { id: 'one', name: 'One' },
  { id: 'solo_three', name: 'Solo Three' },
  { id: 'solo_two', name: 'Solo Two' },
  { id: 'solo_one', name: 'Solo One' },
  { id: 'solo', name: 'Solo' },
];
const POSITIONS = ['Declarer', 'Partner', 'Opponent 1', 'Opponent 2'];

export default function CameraAgent({ onBack }: { onBack: () => void }) {
  const [hand, setHand] = useState<CardData[]>([]);
  const [trick, setTrick] = useState<CardData[]>([]);
  const [contract, setContract] = useState('three');
  const [position, setPosition] = useState(0);
  const [tricksPlayed, setTricksPlayed] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'hand' | 'trick'>('hand');
  const [tab, setTab] = useState<'taroks' | 'hearts' | 'diamonds' | 'clubs' | 'spades'>('taroks');

  const isCardSelected = (card: CardData, list: CardData[]) =>
    list.some(c => c.card_type === card.card_type && c.value === card.value && c.suit === card.suit);

  const toggleCard = (card: CardData) => {
    const target = mode === 'hand' ? hand : trick;
    const setter = mode === 'hand' ? setHand : setTrick;
    if (isCardSelected(card, target)) {
      setter(target.filter(c => !(c.card_type === card.card_type && c.value === card.value && c.suit === card.suit)));
    } else {
      if (mode === 'trick' && target.length >= 3) return; // max 3 cards in trick before us
      setter([...target, card]);
    }
    setResult(null);
  };

  const analyze = async () => {
    if (hand.length === 0) {
      setError('Select at least one card in your hand');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hand: hand.map(c => ({ card_type: c.card_type, value: c.value, suit: c.suit })),
          trick: trick.map(c => ({ card_type: c.card_type, value: c.value, suit: c.suit })),
          contract,
          position,
          tricks_played: tricksPlayed,
          played_cards: [],
        }),
      });
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
      const data = await resp.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const allCards: { label: string; cards: CardData[] }[] = [
    { label: 'Taroks', cards: ALL_TAROKS },
    ...SUITS.map(s => ({
      label: `${SUIT_SYMBOLS[s]} ${s.charAt(0).toUpperCase() + s.slice(1)}`,
      cards: SUIT_RANKS.map(r => makeSuitCard(s, r)),
    })),
  ];

  const tabMap: Record<string, number> = { taroks: 0, hearts: 1, diamonds: 2, clubs: 3, spades: 4 };
  const currentCards = allCards[tabMap[tab]].cards;

  return (
    <div className="camera-agent">
      <div className="ca-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
        <h2>Camera Agent</h2>
        <p className="ca-subtitle">Input your real Tarok hand and get AI move recommendations</p>
      </div>

      {/* Game context */}
      <div className="ca-context">
        <div className="ca-field">
          <label>Contract</label>
          <select value={contract} onChange={e => setContract(e.target.value)}>
            {CONTRACTS.map(c => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
        </div>
        <div className="ca-field">
          <label>Your role</label>
          <select value={position} onChange={e => setPosition(Number(e.target.value))}>
            {POSITIONS.map((p, i) => (
              <option key={i} value={i}>{p}</option>
            ))}
          </select>
        </div>
        <div className="ca-field">
          <label>Tricks played</label>
          <input type="number" min={0} max={11} value={tricksPlayed}
            onChange={e => setTricksPlayed(Number(e.target.value))} />
        </div>
      </div>

      {/* Mode toggle */}
      <div className="ca-mode-toggle">
        <button className={`btn-sm ${mode === 'hand' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setMode('hand')}>
          Your Hand ({hand.length})
        </button>
        <button className={`btn-sm ${mode === 'trick' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setMode('trick')}>
          Current Trick ({trick.length})
        </button>
      </div>

      {/* Card picker tabs */}
      <div className="ca-tabs">
        {(['taroks', 'hearts', 'diamonds', 'clubs', 'spades'] as const).map(t => (
          <button key={t} className={`ca-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'taroks' ? '🃏 Taroks' : `${SUIT_SYMBOLS[t]} ${t.charAt(0).toUpperCase() + t.slice(1)}`}
          </button>
        ))}
      </div>

      {/* Card grid */}
      <div className="ca-card-grid">
        {currentCards.map((card, i) => {
          const inHand = isCardSelected(card, hand);
          const inTrick = isCardSelected(card, trick);
          const selected = mode === 'hand' ? inHand : inTrick;
          const disabled = mode === 'hand' ? inTrick : inHand;
          return (
            <div key={i}
              className={`ca-card-slot ${selected ? 'selected' : ''} ${disabled ? 'disabled' : ''}`}
              onClick={() => !disabled && toggleCard(card)}>
              <Card card={card} />
              {inHand && <span className="ca-badge hand-badge">H</span>}
              {inTrick && <span className="ca-badge trick-badge">T</span>}
            </div>
          );
        })}
      </div>

      {/* Selected cards summary */}
      <div className="ca-selected">
        <div className="ca-selected-section">
          <h4>Your Hand</h4>
          <div className="ca-selected-cards">
            {hand.length === 0 ? <span className="ca-empty">Tap cards above to add</span> :
              hand.map((c, i) => <Card key={i} card={c} />)}
          </div>
        </div>
        {trick.length > 0 && (
          <div className="ca-selected-section">
            <h4>Cards on Table</h4>
            <div className="ca-selected-cards">
              {trick.map((c, i) => <Card key={i} card={c} />)}
            </div>
          </div>
        )}
      </div>

      {/* Analyze button */}
      <button className="btn-gold btn-large ca-analyze-btn" onClick={analyze} disabled={loading || hand.length === 0}>
        {loading ? 'Analyzing...' : '🤖 Get AI Recommendation'}
      </button>

      {error && <div className="ca-error">{error}</div>}

      {/* Results */}
      {result && (
        <div className="ca-results">
          <div className="ca-recommendation">
            <h3>
              {result.has_trained_model ? '🎯 AI Recommends' : '🎲 Random (no trained model yet)'}
            </h3>
            <div className="ca-rec-card">
              <Card card={result.recommended} highlighted={true} />
            </div>
            {result.position_value !== null && (
              <p className="ca-position-eval">
                Position evaluation: <strong>{result.position_value > 0 ? '+' : ''}{result.position_value}</strong>
              </p>
            )}
          </div>

          {result.ranked_plays.length > 1 && (
            <div className="ca-ranking">
              <h4>All Legal Plays (ranked by AI preference)</h4>
              <div className="ca-ranked-list">
                {result.ranked_plays.map((rp, i) => (
                  <div key={i} className="ca-ranked-item">
                    <span className="ca-rank">#{i + 1}</span>
                    <Card card={rp.card}
                      highlighted={rp.card.label === result.recommended.label} />
                    <span className="ca-prob">{(rp.probability * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

```

### File: `/Users/swozny/work/tarok/frontend/src/components/Card.tsx`
```tsx
import React from 'react';
import type { CardData } from '../types/game';
import './Card.css';

interface CardProps {
  card: CardData;
  onClick?: () => void;
  disabled?: boolean;
  highlighted?: boolean;
  faceDown?: boolean;
  small?: boolean;
}

const SUIT_SYMBOLS: Record<string, string> = {
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
  spades: '♠',
};

const TAROK_NUMERALS: Record<number, string> = {
  1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
  6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
  11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV', 15: 'XV',
  16: 'XVI', 17: 'XVII', 18: 'XVIII', 19: 'XIX', 20: 'XX',
  21: 'XXI', 22: 'Škis',
};

function isTrula(card: CardData): boolean {
  return card.card_type === 'tarok' && [1, 21, 22].includes(card.value);
}

export default function Card({ card, onClick, disabled, highlighted, faceDown, small }: CardProps) {
  if (faceDown) {
    return (
      <div className={`card card-back ${small ? 'card-small' : ''}`}>
        <div className="card-back-pattern">
          <div className="card-back-inner">✦</div>
        </div>
      </div>
    );
  }

  const isTarok = card.card_type === 'tarok';
  const isRed = card.suit === 'hearts' || card.suit === 'diamonds';
  const trula = isTrula(card);

  const classes = [
    'card',
    isTarok ? 'card-tarok' : 'card-suit',
    isRed ? 'card-red' : 'card-black',
    trula ? 'card-trula' : '',
    highlighted ? 'card-highlighted' : '',
    disabled ? 'card-disabled' : '',
    onClick && !disabled ? 'card-clickable' : '',
    small ? 'card-small' : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={classes} onClick={!disabled && onClick ? onClick : undefined} data-testid={`card-${card.card_type}-${card.value}-${card.suit ?? 'none'}`}>
      {isTarok ? (
        <div className="card-content card-content-tarok">
          <div className="card-corner card-corner-top">
            <span className="card-numeral">{TAROK_NUMERALS[card.value]}</span>
          </div>
          <div className="card-center">
            <div className="tarok-emblem">
              {trula ? (
                <span className="tarok-star">★</span>
              ) : (
                <span className="tarok-number">{TAROK_NUMERALS[card.value]}</span>
              )}
            </div>
          </div>
          <div className="card-corner card-corner-bottom">
            <span className="card-numeral">{TAROK_NUMERALS[card.value]}</span>
          </div>
          <div className="card-points">{card.points}pt</div>
        </div>
      ) : (
        <div className="card-content card-content-suit">
          <div className="card-corner card-corner-top">
            <span className="card-rank">{card.label.replace(/[♥♦♣♠]/g, '')}</span>
            <span className="card-suit-symbol">{SUIT_SYMBOLS[card.suit!]}</span>
          </div>
          <div className="card-center">
            <span className="suit-large">{SUIT_SYMBOLS[card.suit!]}</span>
          </div>
          <div className="card-corner card-corner-bottom">
            <span className="card-rank">{card.label.replace(/[♥♦♣♠]/g, '')}</span>
            <span className="card-suit-symbol">{SUIT_SYMBOLS[card.suit!]}</span>
          </div>
          <div className="card-points">{card.points}pt</div>
        </div>
      )}
    </div>
  );
}

```

### File: `/Users/swozny/work/tarok/frontend/src/components/GameBoard.tsx`
```tsx
import React from 'react';
import type { GameState, CardData } from '../types/game';
import { CONTRACT_NAMES } from '../types/game';
import Hand from './Hand';
import TrickArea from './TrickArea';
import BiddingPanel from './BiddingPanel';
import Card from './Card';
import './GameBoard.css';

interface GameBoardProps {
  state: GameState;
  onPlayCard: (card: CardData) => void;
  onBid: (contract: number | null) => void;
  onCallKing: (suit: string) => void;
  onChooseTalon: (groupIndex: number) => void;
  onDiscard: (cards: CardData[]) => void;
}

export default function GameBoard({
  state, onPlayCard, onBid, onCallKing, onChooseTalon, onDiscard,
}: GameBoardProps) {
  const isMyTurn = state.current_player === 0;
  const names = state.player_names.length > 0 ? state.player_names : ['You', 'AI-1', 'AI-2', 'AI-3'];

  return (
    <div className="game-board" data-testid="game-board" data-phase={state.phase}>      {/* Game info bar */}
      <div className="game-info-bar">
        <div className="info-item">
          <span className="info-label">Tricks</span>
          <span className="info-value">{state.tricks_played}/12</span>
        </div>
        {state.contract !== null && (
          <div className="info-item">
            <span className="info-label">Contract</span>
            <span className="info-value">{CONTRACT_NAMES[state.contract] ?? state.contract}</span>
          </div>
        )}
        {state.declarer !== null && (
          <div className="info-item">
            <span className="info-label">Declarer</span>
            <span className="info-value">{names[state.declarer]}</span>
          </div>
        )}
        {state.called_king && (
          <div className="info-item">
            <span className="info-label">Called</span>
            <span className="info-value">{state.called_king.label}</span>
          </div>
        )}
        <div className="info-item">
          <span className="info-label">Phase</span>
          <span className="info-value phase-badge">{state.phase.replace(/_/g, ' ')}</span>
        </div>
      </div>

      {/* Table layout */}
      <div className="table">
        {/* Top player (P2) */}
        <div className="table-top">
          <Hand cards={[]} faceDown cardCount={state.hand_sizes[2]} position="top" label={names[2]} />
        </div>

        {/* Left player (P1) */}
        <div className="table-left">
          <Hand cards={[]} faceDown cardCount={state.hand_sizes[1]} position="left" label={names[1]} />
        </div>

        {/* Center — trick area */}
        <div className="table-center">
          {state.phase === 'trick_play' && (
            <TrickArea
              trickCards={state.current_trick}
              playerNames={names}
              playerIndex={0}
            />
          )}

          {/* Bidding */}
          {(state.phase === 'bidding' || state.phase === 'king_calling') && (
            <BiddingPanel
              phase={state.phase}
              bids={state.bids}
              legalBids={isMyTurn && state.phase === 'bidding' ? getLegalBidValues(state) : undefined}
              onBid={onBid}
              playerNames={names}
              callableKings={isMyTurn && state.phase === 'king_calling' ? state.legal_plays : undefined}
              onCallKing={onCallKing}
            />
          )}

          {/* Talon selection */}
          {state.phase === 'talon_exchange' && state.talon_groups && isMyTurn && (
            <div className="talon-selection">
              <h3>Choose a talon group</h3>
              <div className="talon-groups">
                {state.talon_groups.map((group, i) => (
                  <div key={i} className="talon-group" onClick={() => onChooseTalon(i)}>
                    {group.map((card, j) => (
                      <Card key={j} card={card} small />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scores */}
          {state.phase === 'finished' && state.scores && (
            <div className="score-display" data-testid="score-display">
              <h3>Game Over!</h3>
              <div className="score-list">
                {Object.entries(state.scores).map(([pid, score]) => (
                  <div key={pid} className={`score-entry ${Number(score) > 0 ? 'score-positive' : 'score-negative'}`}>
                    <span>{names[Number(pid)]}</span>
                    <span className="score-value">{Number(score) > 0 ? '+' : ''}{score}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right player (P3) */}
        <div className="table-right">
          <Hand cards={[]} faceDown cardCount={state.hand_sizes[3]} position="right" label={names[3]} />
        </div>

        {/* Bottom player (human, P0) */}
        <div className="table-bottom">
          <Hand
            cards={state.hand}
            legalPlays={isMyTurn && state.phase === 'trick_play' ? state.legal_plays : undefined}
            onCardClick={isMyTurn && state.phase === 'trick_play' ? onPlayCard : undefined}
            position="bottom"
            label={names[0]}
          />
        </div>
      </div>

      {/* Turn indicator */}
      {state.phase === 'trick_play' && (
        <div className={`turn-indicator ${isMyTurn ? 'your-turn' : ''}`} data-testid="turn-indicator">
          {isMyTurn ? '🎯 Your turn — play a card' : `Waiting for ${names[state.current_player]}...`}
        </div>
      )}
    </div>
  );
}

function getLegalBidValues(state: GameState): (number | null)[] {
  // Derived from the bids so far
  const highestBid = state.bids.reduce((max, b) => {
    if (b.contract !== null && (max === null || bidStrength(b.contract) > bidStrength(max))) {
      return b.contract;
    }
    return max;
  }, null as number | null);

  const options: (number | null)[] = [null]; // pass
  const allContracts = [3, 2, 1, 0];
  for (const c of allContracts) {
    if (highestBid === null || bidStrength(c) > bidStrength(highestBid)) {
      options.push(c);
    }
  }
  return options;
}

function bidStrength(contract: number): number {
  const strengths: Record<number, number> = { 3: 1, 2: 2, 1: 3, '-3': 4, '-2': 5, '-1': 6, 0: 7 };
  return strengths[contract] ?? 0;
}

```

### File: `/Users/swozny/work/tarok/frontend/src/components/GameLog.tsx`
```tsx
import React, { useEffect, useRef } from 'react';
import type { LogEntry } from '../hooks/useGame';
import './GameLog.css';

interface GameLogProps {
  entries: LogEntry[];
}

const CATEGORY_ICONS: Record<string, string> = {
  system: '⚙️',
  bid: '🗣️',
  king: '👑',
  talon: '📦',
  play: '🃏',
  trick: '✅',
  score: '🏆',
};

export default function GameLog({ entries }: GameLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries]);

  return (
    <div className="game-log" data-testid="game-log">
      <div className="game-log-header">
        <h3>Game Log</h3>
      </div>
      <div className="game-log-entries">
        {entries.length === 0 && (
          <div className="game-log-empty">Waiting for game to start…</div>
        )}
        {entries.map((entry) => (
          <div
            key={entry.id}
            className={`game-log-entry log-${entry.category}${entry.isHuman ? ' log-human' : ''}`}
          >
            <span className="log-icon">{CATEGORY_ICONS[entry.category] ?? '•'}</span>
            <span className="log-message">{entry.message}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

```

### File: `/Users/swozny/work/tarok/frontend/src/components/Hand.tsx`
```tsx
import React from 'react';
import Card from './Card';
import type { CardData } from '../types/game';
import './Hand.css';

interface HandProps {
  cards: CardData[];
  legalPlays?: CardData[];
  onCardClick?: (card: CardData) => void;
  faceDown?: boolean;
  position?: 'bottom' | 'top' | 'left' | 'right';
  label?: string;
  cardCount?: number;
}

function cardKey(c: CardData): string {
  return `${c.card_type}-${c.value}-${c.suit ?? 'none'}`;
}

function isLegal(card: CardData, legalPlays?: CardData[]): boolean {
  if (!legalPlays) return true;
  return legalPlays.some(
    lp => lp.card_type === card.card_type && lp.value === card.value && lp.suit === card.suit
  );
}

export default function Hand({ cards, legalPlays, onCardClick, faceDown, position = 'bottom', label, cardCount }: HandProps) {
  const isHorizontal = position === 'bottom' || position === 'top';
  const count = cardCount ?? cards.length;

  return (
    <div className={`hand hand-${position}`}>
      {label && <div className="hand-label">{label}</div>}
      <div className={`hand-cards ${isHorizontal ? 'hand-horizontal' : 'hand-vertical'}`}>
        {faceDown ? (
          Array.from({ length: count }).map((_, i) => (
            <Card key={i} card={{ card_type: 'tarok', value: 0, suit: null, label: '', points: 0 }} faceDown small />
          ))
        ) : (
          cards.map(card => (
            <Card
              key={cardKey(card)}
              card={card}
              onClick={onCardClick && isLegal(card, legalPlays) ? () => onCardClick(card) : undefined}
              disabled={legalPlays !== undefined && !isLegal(card, legalPlays)}
              highlighted={legalPlays !== undefined && isLegal(card, legalPlays)}
            />
          ))
        )}
      </div>
    </div>
  );
}

```

### File: `/Users/swozny/work/tarok/frontend/src/components/TrainingDashboard.tsx`
```tsx
import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell,
} from 'recharts';
import type { TrainingMetrics, ContractStat } from '../types/game';
import './TrainingDashboard.css';

interface Props { onBack: () => void }

const CONTRACT_LABELS: Record<string, string> = {
  klop: 'Klop', three: 'Tri', two: 'Dve', one: 'Ena',
  solo_three: 'Solo 3', solo_two: 'Solo 2', solo_one: 'Solo 1', solo: 'Solo',
};
const CONTRACT_COLORS: Record<string, string> = {
  klop: '#888', three: '#4caf50', two: '#2196f3', one: '#ff9800',
  solo_three: '#9c27b0', solo_two: '#e91e63', solo_one: '#f44336', solo: '#d4a843',
};

const API = '';

export default function TrainingDashboard({ onBack }: Props) {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [sessions, setSessions] = useState(100);
  const [gamesPerSession, setGamesPerSession] = useState(100);
  const [resume, setResume] = useState(false);
  const [tab, setTab] = useState<'overview' | 'contracts' | 'loss' | 'snapshots'>('overview');

  const poll = useCallback(async () => {
    try {
      const [mRes, sRes] = await Promise.all([
        fetch(`${API}/api/training/metrics`),
        fetch(`${API}/api/training/status`),
      ]);
      const mData = await mRes.json();
      const sData = await sRes.json();
      setMetrics(mData);
      setIsTraining(sData.running);
    } catch { /* server not up */ }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 800);
    return () => clearInterval(id);
  }, [poll]);

  const startTraining = async () => {
    await fetch(`${API}/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_sessions: sessions, games_per_session: gamesPerSession, resume }),
    });
    setIsTraining(true);
  };

  const stopTraining = async () => {
    await fetch(`${API}/api/training/stop`, { method: 'POST' });
    setIsTraining(false);
  };

  // Chart data
  const rewardData = metrics?.reward_history.map((v, i) => ({ s: i + 1, reward: v })) ?? [];
  const winRateData = metrics?.win_rate_history.map((v, i) => ({ s: i + 1, winRate: +(v * 100).toFixed(1) })) ?? [];
  const lossData = metrics?.loss_history.map((v, i) => ({ s: i + 1, loss: v })) ?? [];
  const sessionScoreData = metrics?.session_avg_score_history?.map((v, i) => ({ s: i + 1, avgScore: v })) ?? [];
  const bidKlopData = metrics?.bid_rate_history?.map((v, i) => ({
    s: i + 1,
    bid: +((metrics.bid_rate_history[i] ?? 0) * 100).toFixed(1),
    klop: +((metrics.klop_rate_history?.[i] ?? 0) * 100).toFixed(1),
    solo: +((metrics.solo_rate_history?.[i] ?? 0) * 100).toFixed(1),
  })) ?? [];

  // Contract bar data (declarer stats only — the meaningful metric)
  const contractBarData = metrics?.contract_stats ? Object.entries(metrics.contract_stats)
    .filter(([, cs]) => cs.decl_played > 0)
    .map(([name, cs]) => ({
      name: CONTRACT_LABELS[name] || name,
      key: name,
      played: cs.decl_played,
      winRate: +(cs.decl_win_rate * 100).toFixed(1),
      avgScore: +cs.decl_avg_score.toFixed(1),
    })) : [];

  // Contract win-rate over time
  const contractWinData = metrics?.contract_win_rate_history
    ? (metrics.win_rate_history || []).map((_, i) => {
        const row: Record<string, number> = { s: i + 1 };
        for (const [cname, arr] of Object.entries(metrics.contract_win_rate_history)) {
          if (arr[i] !== undefined) row[cname] = +(arr[i] * 100).toFixed(1);
        }
        return row;
      })
    : [];

  const sessionPct = metrics && metrics.total_sessions > 0
    ? (metrics.session / metrics.total_sessions) * 100
    : 0;

  return (
    <div className="training-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
        <h2>AI Training Dashboard</h2>
      </div>

      {/* Controls */}
      <div className="training-controls-bar">
        <label className="td-field">
          <span>Sessions</span>
          <input type="number" value={sessions} onChange={e => setSessions(Number(e.target.value))}
            disabled={isTraining} min={10} step={10} />
        </label>
        <label className="td-field">
          <span>Games/Session</span>
          <input type="number" value={gamesPerSession} onChange={e => setGamesPerSession(Number(e.target.value))}
            disabled={isTraining} min={10} step={10} />
        </label>
        <label className="td-check">
          <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} disabled={isTraining} />
          <span>Resume from checkpoint</span>
        </label>
        {isTraining ? (
          <button className="btn-danger" onClick={stopTraining}>Stop</button>
        ) : (
          <button className="btn-gold" onClick={startTraining}>Start Training</button>
        )}
      </div>

      {/* Progress */}
      {metrics && metrics.total_episodes > 0 && (
        <div className="td-progress">
          <div className="td-progress-bar">
            <div className="td-progress-fill" style={{ width: `${sessionPct}%` }} />
          </div>
          <span className="td-progress-text">
            Session {metrics.session}/{metrics.total_sessions} · {metrics.episode.toLocaleString()} games · {metrics.games_per_second.toFixed(1)} g/s
          </span>
        </div>
      )}

      {/* Stat cards */}
      <div className="td-stats">
        <StatCard label="Win Rate" value={`${((metrics?.win_rate ?? 0) * 100).toFixed(1)}%`} highlight />
        <StatCard label="Avg Reward" value={(metrics?.avg_reward ?? 0).toFixed(2)} />
        <StatCard label="Sess. Avg Score"
          value={metrics?.session_avg_score_history?.length
            ? metrics.session_avg_score_history[metrics.session_avg_score_history.length - 1].toFixed(1)
            : '—'}
        />
        <StatCard label="Bid Rate" value={`${((metrics?.bid_rate ?? 0) * 100).toFixed(0)}%`} />
        <StatCard label="Klop Rate" value={`${((metrics?.klop_rate ?? 0) * 100).toFixed(0)}%`} />
        <StatCard label="Solo Rate" value={`${((metrics?.solo_rate ?? 0) * 100).toFixed(0)}%`} />
        <StatCard label="Games/sec" value={(metrics?.games_per_second ?? 0).toFixed(1)} />
        <StatCard label="Entropy" value={(metrics?.entropy ?? 0).toFixed(4)} />
        <StatCard label="Policy Loss" value={(metrics?.policy_loss ?? 0).toFixed(4)} />
      </div>

      {/* Tabs */}
      <div className="td-tabs">
        {(['overview', 'contracts', 'loss', 'snapshots'] as const).map(t => (
          <button key={t} className={`td-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'overview' ? '📈 Overview' : t === 'contracts' ? '🃏 Contracts' : t === 'loss' ? '📉 Loss & Entropy' : '💾 Snapshots'}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="td-tab-content">
        {tab === 'overview' && (
          <div className="chart-grid">
            <ChartCard title="Win Rate Over Time">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={winRateData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} domain={[0, 100]} unit="%" />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="winRate" stroke="#4caf50" strokeWidth={2} dot={false} name="Win %" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Average Reward">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rewardData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="reward" stroke="#d4a843" strokeWidth={2} dot={false} name="Reward" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Session Avg Score (P0)">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={sessionScoreData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="avgScore" stroke="#ff9800" strokeWidth={2} dot={false} name="Avg Score" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Bid / Klop / Solo Rates">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={bidKlopData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} domain={[0, 100]} unit="%" />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Line type="monotone" dataKey="bid" stroke="#2196f3" strokeWidth={2} dot={false} name="Bid %" />
                  <Line type="monotone" dataKey="klop" stroke="#888" strokeWidth={2} dot={false} name="Klop %" />
                  <Line type="monotone" dataKey="solo" stroke="#e91e63" strokeWidth={2} dot={false} name="Solo %" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}

        {tab === 'contracts' && (
          <div className="chart-grid">
            <ChartCard title="Declarer: Play Count & Win Rate">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={contractBarData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis type="number" stroke="#666" fontSize={11} />
                  <YAxis type="category" dataKey="name" stroke="#666" fontSize={12} width={70} />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Bar dataKey="played" name="Declared" fill="#4a9eff" />
                  <Bar dataKey="winRate" name="Win %" fill="#4caf50" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Declarer: Average Score by Contract">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={contractBarData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis type="number" stroke="#666" fontSize={11} />
                  <YAxis type="category" dataKey="name" stroke="#666" fontSize={12} width={70} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey="avgScore" name="Avg Score">
                    {contractBarData.map((entry) => (
                      <Cell key={entry.key} fill={CONTRACT_COLORS[entry.key] || '#888'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Declarer Win Rates Over Time" wide>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={contractWinData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} domain={[0, 100]} unit="%" />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  {Object.keys(CONTRACT_LABELS).map(cname => (
                    <Line
                      key={cname}
                      type="monotone"
                      dataKey={cname}
                      stroke={CONTRACT_COLORS[cname]}
                      strokeWidth={1.5}
                      dot={false}
                      name={CONTRACT_LABELS[cname]}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Contract Breakdown (Declarer vs Defender)" wide>
              <table className="td-table">
                <thead>
                  <tr>
                    <th rowSpan={2}>Contract</th>
                    <th colSpan={3} className="th-group">As Declarer</th>
                    <th colSpan={3} className="th-group">As Defender</th>
                  </tr>
                  <tr>
                    <th>Played</th>
                    <th>Win %</th>
                    <th>Avg Score</th>
                    <th>Played</th>
                    <th>Win %</th>
                    <th>Avg Score</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics?.contract_stats && Object.entries(metrics.contract_stats).map(([name, cs]) => (
                    <tr key={name} className={cs.played === 0 ? 'dimmed' : ''}>
                      <td>
                        <span className="td-dot" style={{ background: CONTRACT_COLORS[name] }} />
                        {CONTRACT_LABELS[name] || name}
                      </td>
                      <td>{cs.decl_played || '—'}</td>
                      <td>{cs.decl_played > 0 ? `${(cs.decl_win_rate * 100).toFixed(1)}%` : '—'}</td>
                      <td>{cs.decl_played > 0 ? cs.decl_avg_score.toFixed(1) : '—'}</td>
                      <td>{cs.def_played || '—'}</td>
                      <td>{cs.def_played > 0 ? `${(cs.def_win_rate * 100).toFixed(1)}%` : '—'}</td>
                      <td>{cs.def_played > 0 ? cs.def_avg_score.toFixed(1) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ChartCard>
          </div>
        )}

        {tab === 'loss' && (
          <div className="chart-grid">
            <ChartCard title="Total Loss">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="loss" stroke="#e94560" strokeWidth={2} dot={false} name="Loss" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}

        {tab === 'snapshots' && (
          <div className="td-snapshots">
            <p className="td-snap-info">
              Snapshots are saved periodically during training. Use them to resume training or play against a specific version.
            </p>
            {metrics?.snapshots && metrics.snapshots.length > 0 ? (
              <table className="td-table">
                <thead>
                  <tr>
                    <th>Checkpoint</th>
                    <th>Session</th>
                    <th>Games</th>
                    <th>Win %</th>
                    <th>Avg Reward</th>
                    <th>Speed</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.snapshots.map((snap, i) => (
                    <tr key={i}>
                      <td className="td-snap-file">{snap.filename}</td>
                      <td>{snap.session}</td>
                      <td>{snap.episode.toLocaleString()}</td>
                      <td>{(snap.win_rate * 100).toFixed(1)}%</td>
                      <td>{snap.avg_reward.toFixed(2)}</td>
                      <td>{snap.games_per_second.toFixed(1)} g/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="td-empty">No snapshots yet. Start training to generate checkpoints.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`td-stat ${highlight ? 'td-stat-hl' : ''}`}>
      <div className="td-stat-val">{value}</div>
      <div className="td-stat-lbl">{label}</div>
    </div>
  );
}

function ChartCard({ title, children, wide }: { title: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={`td-chart ${wide ? 'td-chart-wide' : ''}`}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

const tooltipStyle = {
  contentStyle: { background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 13 },
  labelStyle: { color: '#aaa' },
};

```

### File: `/Users/swozny/work/tarok/frontend/src/components/TrickArea.tsx`
```tsx
import React from 'react';
import Card from './Card';
import type { CardData, TrickCard } from '../types/game';
import './TrickArea.css';

interface TrickAreaProps {
  trickCards: TrickCard[];
  playerNames: string[];
  playerIndex: number;
}

const POSITIONS = ['bottom', 'left', 'top', 'right'] as const;

export default function TrickArea({ trickCards, playerNames, playerIndex }: TrickAreaProps) {
  // Map absolute player indices to relative positions
  // Player 0 (human) is always at bottom
  function relativePosition(absIdx: number): typeof POSITIONS[number] {
    const rel = (absIdx - playerIndex + 4) % 4;
    return POSITIONS[rel];
  }

  return (
    <div className="trick-area">
      {trickCards.length === 0 && (
        <div className="trick-empty">Play a card</div>
      )}
      {trickCards.map(([playerIdx, card]) => (
        <div key={`${playerIdx}-${card.label}`} className={`trick-card trick-card-${relativePosition(playerIdx)}`}>
          <Card card={card} small />
          <span className="trick-player-name">{playerNames[playerIdx] || `P${playerIdx}`}</span>
        </div>
      ))}
    </div>
  );
}

```

### File: `/Users/swozny/work/tarok/frontend/src/hooks/useGame.ts`
```typescript
import { useState, useEffect, useRef, useCallback } from 'react';
import type { GameState, GameEvent, CardData } from '../types/game';
import { CONTRACT_NAMES, SUIT_SYMBOLS } from '../types/game';

export interface LogEntry {
  id: number;
  message: string;
  category: 'system' | 'bid' | 'king' | 'talon' | 'play' | 'trick' | 'score';
  player?: number;
  isHuman?: boolean;
}

const INITIAL_STATE: GameState = {
  phase: 'waiting',
  hand: [],
  hand_sizes: [0, 0, 0, 0],
  talon_groups: null,
  bids: [],
  contract: null,
  declarer: null,
  called_king: null,
  partner_revealed: false,
  partner: null,
  current_trick: [],
  tricks_played: 0,
  current_player: 0,
  scores: null,
  legal_plays: [],
  player_names: [],
};

function cardLabel(card: CardData): string {
  if (card.card_type === 'tarok') return card.label;
  return `${card.label}`;
}

function formatEvent(event: string, data: Record<string, unknown>, names: string[]): LogEntry | null {
  const name = (idx: number) => names[idx] ?? `P${idx}`;
  const isHuman = (idx: number) => idx === 0;
  let nextId = Date.now();

  switch (event) {
    case 'game_start':
      return { id: nextId, message: 'Game started. Dealing cards...', category: 'system' };
    case 'deal':
      return { id: nextId, message: 'Cards dealt to all players.', category: 'system' };
    case 'bid': {
      const p = data.player as number;
      const c = data.contract as number | null;
      const bidText = c !== null ? (CONTRACT_NAMES[c] ?? `${c}`) : 'Pass';
      return { id: nextId, message: `${name(p)} bids: ${bidText}`, category: 'bid', player: p, isHuman: isHuman(p) };
    }
    case 'contract_won': {
      const p = data.player as number;
      const c = data.contract as number;
      return { id: nextId, message: `${name(p)} wins the contract: ${CONTRACT_NAMES[c] ?? c}`, category: 'bid', player: p, isHuman: isHuman(p) };
    }
    case 'king_called': {
      const p = data.player as number;
      const king = data.king as CardData;
      const suit = king.suit ? (SUIT_SYMBOLS[king.suit] ?? king.suit) : '';
      return { id: nextId, message: `${name(p)} calls ${suit} King — the holder is the secret partner!`, category: 'king', player: p, isHuman: isHuman(p) };
    }
    case 'talon_revealed':
      return { id: nextId, message: 'Talon revealed.', category: 'talon' };
    case 'talon_exchanged':
      return { id: nextId, message: 'Talon exchange complete.', category: 'talon' };
    case 'card_played': {
      const p = data.player as number;
      const card = data.card as CardData;
      return { id: nextId, message: `${name(p)} plays ${cardLabel(card)}`, category: 'play', player: p, isHuman: isHuman(p) };
    }
    case 'trick_won': {
      const w = data.winner as number;
      return { id: nextId, message: `${name(w)} wins the trick!`, category: 'trick', player: w, isHuman: isHuman(w) };
    }
    case 'game_end': {
      const scores = data.scores as Record<string, number>;
      const lines = Object.entries(scores)
        .map(([pid, s]) => `${name(Number(pid))}: ${s > 0 ? '+' : ''}${s}`)
        .join(' | ');
      return { id: nextId, message: `Game over — ${lines}`, category: 'score' };
    }
    default:
      return null;
  }
}

export function useGame() {
  const [gameState, setGameState] = useState<GameState>(INITIAL_STATE);
  const [gameId, setGameId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<string[]>([]);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const addEvent = useCallback((msg: string) => {
    setEvents(prev => [...prev.slice(-50), msg]);
  }, []);

  const addLogEntry = useCallback((entry: LogEntry) => {
    setLogEntries(prev => [...prev.slice(-100), entry]);
  }, []);

  const connect = useCallback((id: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/game/${id}`);

    ws.onopen = () => {
      setConnected(true);
      addEvent('Connected to game');
    };

    ws.onmessage = (e) => {
      const data: GameEvent = JSON.parse(e.data);
      setGameState(data.state);
      addEvent(`Event: ${data.event}`);
      const names = data.state.player_names.length > 0 ? data.state.player_names : ['You', 'AI-1', 'AI-2', 'AI-3'];
      const entry = formatEvent(data.event, data.data, names);
      if (entry) addLogEntry(entry);
    };

    ws.onclose = () => {
      setConnected(false);
      addEvent('Disconnected');
    };

    ws.onerror = () => {
      addEvent('WebSocket error');
    };

    wsRef.current = ws;
  }, [addEvent, addLogEntry]);

  const startNewGame = useCallback(async () => {
    try {
      const res = await fetch('/api/game/new', { method: 'POST' });
      const data = await res.json();
      setGameId(data.game_id);
      connect(data.game_id);
    } catch (e) {
      addEvent('Failed to create game');
    }
  }, [connect, addEvent]);

  const sendAction = useCallback((action: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(action));
    }
  }, []);

  const playCard = useCallback((card: { card_type: string; value: number; suit: string | null }) => {
    sendAction({ action: 'play_card', card });
  }, [sendAction]);

  const bid = useCallback((contract: number | null) => {
    sendAction({ action: 'bid', contract });
  }, [sendAction]);

  const callKing = useCallback((suit: string) => {
    sendAction({ action: 'call_king', suit });
  }, [sendAction]);

  const chooseTalon = useCallback((groupIndex: number) => {
    sendAction({ action: 'choose_talon', group_index: groupIndex });
  }, [sendAction]);

  const discard = useCallback((cards: { card_type: string; value: number; suit: string | null }[]) => {
    sendAction({ action: 'discard', cards });
  }, [sendAction]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return {
    gameState,
    gameId,
    connected,
    events,
    logEntries,
    startNewGame,
    playCard,
    bid,
    callKing,
    chooseTalon,
    discard,
  };
}

```

### File: `/Users/swozny/work/tarok/frontend/src/main.tsx`
```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/global.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

```

### File: `/Users/swozny/work/tarok/frontend/src/types/game.ts`
```typescript
export interface CardData {
  card_type: 'tarok' | 'suit';
  value: number;
  suit: 'hearts' | 'diamonds' | 'clubs' | 'spades' | null;
  label: string;
  points: number;
}

export type TrickCard = [number, CardData];

export interface GameState {
  phase: string;
  hand: CardData[];
  hand_sizes: number[];
  talon_groups: CardData[][] | null;
  bids: { player: number; contract: number | null }[];
  contract: number | null;
  declarer: number | null;
  called_king: CardData | null;
  partner_revealed: boolean;
  partner: number | null;
  current_trick: TrickCard[];
  tricks_played: number;
  current_player: number;
  scores: Record<string, number> | null;
  legal_plays: CardData[];
  player_names: string[];
}

export interface GameEvent {
  event: string;
  data: Record<string, unknown>;
  state: GameState;
}

export interface ContractStat {
  played: number;
  decl_played: number;
  decl_won: number;
  decl_win_rate: number;
  decl_avg_score: number;
  def_played: number;
  def_won: number;
  def_win_rate: number;
  def_avg_score: number;
}

export interface SnapshotInfo {
  filename: string;
  episode: number;
  session: number;
  win_rate: number;
  avg_reward: number;
  games_per_second: number;
}

export interface TrainingMetrics {
  episode: number;
  total_episodes: number;
  session: number;
  total_sessions: number;
  avg_reward: number;
  avg_loss: number;
  win_rate: number;
  entropy: number;
  value_loss: number;
  policy_loss: number;
  games_per_second: number;
  bid_rate: number;
  klop_rate: number;
  solo_rate: number;
  contract_stats: Record<string, ContractStat>;
  reward_history: number[];
  win_rate_history: number[];
  loss_history: number[];
  bid_rate_history: number[];
  klop_rate_history: number[];
  solo_rate_history: number[];
  contract_win_rate_history: Record<string, number[]>;
  session_avg_score_history: number[];
  snapshots: SnapshotInfo[];
}

export const CONTRACT_NAMES: Record<number, string> = {
  '-99': 'Klop',
  3: 'Three',
  2: 'Two',
  1: 'One',
  '-3': 'Solo Three',
  '-2': 'Solo Two',
  '-1': 'Solo One',
  0: 'Solo',
};

export const SUIT_SYMBOLS: Record<string, string> = {
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
  spades: '♠',
};

```

### File: `/Users/swozny/work/tarok/frontend/src/vite-env.d.ts`
```typescript
/// <reference types="vite/client" />

```

