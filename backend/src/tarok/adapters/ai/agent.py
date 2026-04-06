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
from tarok.adapters.ai.behavioral_profile import (
    BehavioralProfile, apply_behavioral_bias, apply_temperature,
)
from tarok.adapters.ai.network import TarokNet
from tarok.adapters.ai.encoding import (
    DecisionType,
    encode_state,
    encode_oracle_state,
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
    oracle_state: torch.Tensor | None = None
    game_id: int = 0          # which game this experience belongs to
    step_in_game: int = 0     # temporal position within game


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
        oracle_critic: bool = False,
        profile: BehavioralProfile | None = None,
    ):
        self._name = name
        self.device = torch.device(device)
        self.network = TarokNet(hidden_size, oracle_critic=oracle_critic).to(self.device)
        self.explore_rate = explore_rate
        self.profile = profile
        self._rng = random.Random()

        # Experience buffer for current game
        self.experiences: list[Experience] = []
        self._training = True
        self._game_id = 0
        self._step_counter = 0

    @property
    def name(self) -> str:
        return self._name

    def set_training(self, training: bool) -> None:
        self._training = training
        self.network.train(training)

    @staticmethod
    def from_checkpoint(
        path: str | "Path",
        name: str = "RL-Agent",
        device: str = "cpu",
        explore_rate: float = 0.1,
        oracle_critic: bool = False,
        profile: BehavioralProfile | None = None,
    ) -> "RLAgent":
        """Create an RLAgent with hidden_size inferred from checkpoint weights."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state_dict = ckpt["model_state_dict"]
        # Infer hidden_size from the first layer of the shared backbone
        hidden_size = state_dict["shared.0.weight"].shape[0]
        # Detect oracle critic from checkpoint keys
        has_oracle = any(k.startswith("critic_backbone.") for k in state_dict)
        agent = RLAgent(
            name=name,
            hidden_size=hidden_size,
            device=device,
            explore_rate=explore_rate,
            oracle_critic=has_oracle or oracle_critic,
            profile=profile,
        )
        agent.network.load_state_dict(state_dict)
        return agent

    # ------------------------------------------------------------------
    # Generic decision helper
    # ------------------------------------------------------------------

    def _decide_from_tensors(
        self,
        state_tensor: torch.Tensor,
        legal_mask: torch.Tensor,
        decision_type: DecisionType,
        oracle_tensor: torch.Tensor | None = None,
    ) -> int:
        """Same as _decide but with pre-encoded tensors (for Rust engine)."""
        state_tensor = state_tensor.to(self.device)
        mask = legal_mask.to(self.device)
        if oracle_tensor is not None:
            oracle_tensor = oracle_tensor.to(self.device)

        return self._decide_core(state_tensor, mask, decision_type, oracle_tensor, is_defender=False)

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

        # Oracle state for critic (perfect information — all hands visible)
        oracle_tensor = None
        if self._training and self.network.oracle_critic_enabled:
            oracle_tensor = encode_oracle_state(state, player_idx, decision_type).to(self.device)

        is_defender = (
            state.declarer is not None
            and state.declarer != player_idx
            and not (state.partner is not None and state.partner == player_idx)
        )

        return self._decide_core(state_tensor, mask, decision_type, oracle_tensor, is_defender=is_defender)

    def _decide_core(
        self,
        state_tensor: torch.Tensor,
        mask: torch.Tensor,
        decision_type: DecisionType,
        oracle_tensor: torch.Tensor | None,
        is_defender: bool = False,
    ) -> int:
        """Core decision logic shared by _decide and _decide_from_tensors."""
        # Epsilon-greedy exploration during training
        if self._training and self._rng.random() < self.explore_rate:
            legal_indices = mask.nonzero(as_tuple=True)[0].tolist()
            action_idx = self._rng.choice(legal_indices)
            with torch.no_grad():
                _, value = self.network(
                    state_tensor.unsqueeze(0), decision_type,
                    oracle_state=oracle_tensor.unsqueeze(0) if oracle_tensor is not None else None,
                )
            self.experiences.append(Experience(
                state=state_tensor,
                action=action_idx,
                log_prob=torch.tensor(0.0),
                value=value.squeeze(),
                decision_type=decision_type,
                oracle_state=oracle_tensor,
                game_id=self._game_id,
                step_in_game=self._step_counter,
            ))
            self._step_counter += 1
            return action_idx

        if self.profile is not None:
            # Behavioral profile path: get raw logits, apply bias + temperature
            ctx = torch.no_grad() if not self._training else torch.enable_grad()
            with ctx:
                logits, value = self.network(
                    state_tensor.unsqueeze(0), decision_type,
                    oracle_state=oracle_tensor.unsqueeze(0) if oracle_tensor is not None else None,
                )
                biased = apply_behavioral_bias(
                    logits, self.profile, decision_type, mask.unsqueeze(0),
                    is_defender=is_defender,
                )
                tempered = apply_temperature(biased, self.profile.temperature)
                masked = tempered.clone()
                masked[mask.unsqueeze(0) == 0] = float("-inf")
                probs = torch.softmax(masked, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                action_idx = action.item()
                log_prob = dist.log_prob(action).squeeze()
                value = value.squeeze(-1).squeeze()
        else:
            ctx = torch.no_grad() if not self._training else torch.enable_grad()
            with ctx:
                action_idx, log_prob, value = self.network.get_action(
                    state_tensor.unsqueeze(0), mask.unsqueeze(0), decision_type,
                    oracle_state=oracle_tensor.unsqueeze(0) if oracle_tensor is not None else None,
                )

        if self._training:
            self.experiences.append(Experience(
                state=state_tensor.detach(),
                action=action_idx,
                log_prob=log_prob.detach() if isinstance(log_prob, torch.Tensor) else torch.tensor(log_prob),
                value=value.detach() if isinstance(value, torch.Tensor) else torch.tensor(value),
                decision_type=decision_type,
                oracle_state=oracle_tensor.detach() if oracle_tensor is not None else None,
                game_id=self._game_id,
                step_in_game=self._step_counter,
            ))
            self._step_counter += 1

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
        """Set the terminal reward on only the last experience; others get 0."""
        if self.experiences:
            # Only the final step gets the game reward; intermediates stay 0
            self.experiences[-1].reward = reward
            self.experiences[-1].done = True
        self._game_id += 1
        self._step_counter = 0

    def clear_experiences(self) -> None:
        self.experiences.clear()

    def decay_exploration(self) -> None:
        """Decay explore_rate using the behavioral profile's schedule."""
        if self.profile is not None:
            self.explore_rate = max(
                self.profile.explore_floor,
                self.explore_rate * self.profile.explore_decay,
            )
        else:
            # Default decay: 0.5% per call, floor at 2%
            self.explore_rate = max(0.02, self.explore_rate * 0.995)
