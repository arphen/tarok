"""Lookahead (Monte Carlo search) agent — exhaustive legal-move evaluation.

For each decision, this agent samples possible distributions of unknown cards
among opponents, then evaluates every legal move by simulating games to
completion with random playout. It picks the move that maximises its expected
score.

Can be used as:
  - A strong opponent for human play
  - A teacher for training RL agents (via the standard GameLoop)
  - A baseline to benchmark other agents against

The ``n_simulations`` parameter controls strength vs speed:
  - 1–10:   fast but noisy
  - 50–200: strong play, suitable for training
  - 500+:   near-optimal for the trick-play phase
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from tarok.entities.card import Card, CardType, SuitRank, DECK
from tarok.entities.game_state import (
    Announcement,
    Contract,
    GameState,
    Phase,
    Trick,
)
from tarok.entities.scoring import score_game
from tarok.use_cases.play_trick import play_card, start_trick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unseen_cards(state: GameState, player_idx: int) -> list[Card]:
    """Cards that *player_idx* knows exist but hasn't seen (opponents' hands)."""
    all_cards = set(DECK)

    # Remove cards we can see
    seen = set(state.hands[player_idx])            # our hand
    seen.update(state.put_down)                     # cards put down in talon exchange
    for trick in state.tricks:                      # completed tricks
        for _, c in trick.cards:
            seen.add(c)
    if state.current_trick:                         # current trick in progress
        for _, c in state.current_trick.cards:
            seen.add(c)
    # Talon cards that were revealed to us
    for group in state.talon_revealed:
        for c in group:
            seen.add(c)
    # Cards still in the talon (unrevealed) are unknown but not in opponent hands
    seen.update(state.talon)

    return list(all_cards - seen)


def _deal_unknown(
    state: GameState,
    player_idx: int,
    rng: random.Random,
) -> GameState:
    """Create a copy of *state* with unknown opponent cards randomly assigned.

    Cards that *player_idx* can see in an opponent's hand (e.g. talon cards
    picked by the declarer) stay put; only the truly unknown slots are
    filled from the shuffled unseen pool.
    """
    st = copy.deepcopy(state)
    unseen = _unseen_cards(state, player_idx)
    unseen_set = set(unseen)
    rng.shuffle(unseen)

    idx = 0
    for p in range(st.num_players):
        if p == player_idx:
            continue
        # Keep cards we already know are in this hand (e.g. talon picks)
        known = [c for c in st.hands[p] if c not in unseen_set]
        need = len(st.hands[p]) - len(known)
        st.hands[p] = known + unseen[idx : idx + need]
        idx += need

    return st


def _simulate_random_playout(
    state: GameState,
    rng: random.Random,
) -> dict[int, int]:
    """Play out the remaining tricks with random moves. Returns scores."""
    st = copy.deepcopy(state)

    max_iters = 60  # 12 tricks × 4 cards + safety margin
    iters = 0
    while st.phase == Phase.TRICK_PLAY and iters < max_iters:
        iters += 1
        if st.current_trick is None:
            st = start_trick(st)
        while st.current_trick is not None and not st.current_trick.is_complete:
            p = st.current_player
            legal = st.legal_plays(p)
            if not legal:
                break
            card = rng.choice(legal)
            st = play_card(st, p, card)
        # If trick couldn't progress (empty legal), bail out
        if st.current_trick is not None and not st.current_trick.is_complete:
            break

    if st.phase in (Phase.SCORING, Phase.FINISHED):
        return score_game(st)

    # Shouldn't happen, but return zeros as safety
    return {p: 0 for p in range(st.num_players)}


def _evaluate_move(
    state: GameState,
    player_idx: int,
    card: Card,
    n_simulations: int,
    rng: random.Random,
) -> float:
    """Evaluate a single card play by averaging score over random worlds."""
    total_score = 0.0
    for _ in range(n_simulations):
        sim_rng = random.Random(rng.randint(0, 2**31))
        # Randomise unknown cards
        sim_state = _deal_unknown(state, player_idx, sim_rng)
        # Play our card
        sim_state = play_card(sim_state, player_idx, card)
        # Random playout to end
        scores = _simulate_random_playout(sim_state, sim_rng)
        total_score += scores.get(player_idx, 0)
    return total_score / max(n_simulations, 1)


def _evaluate_move_perfect(
    state: GameState,
    player_idx: int,
    card: Card,
    n_playouts: int,
    rng: random.Random,
) -> float:
    """Evaluate a card play with perfect info (known card distribution).

    Skips the expensive _deal_unknown sampling — uses the true hands
    directly.  Multiple playouts still average over opponent random play.
    """
    total_score = 0.0
    for _ in range(n_playouts):
        sim_rng = random.Random(rng.randint(0, 2**31))
        sim_state = copy.deepcopy(state)
        sim_state = play_card(sim_state, player_idx, card)
        scores = _simulate_random_playout(sim_state, sim_rng)
        total_score += scores.get(player_idx, 0)
    return total_score / max(n_playouts, 1)


# ---------------------------------------------------------------------------
# Bidding heuristic (hand strength evaluation)
# ---------------------------------------------------------------------------

def _hand_strength(hand: list[Card]) -> float:
    """Estimate hand strength as a rough score 0..100."""
    taroks = [c for c in hand if c.card_type == CardType.TAROK]
    kings = [c for c in hand if c.is_king]
    high_taroks = [c for c in taroks if c.value >= 15]

    strength = 0.0
    strength += len(taroks) * 3.5        # each tarok is worth ~3.5
    strength += len(high_taroks) * 2.0    # high taroks extra bonus
    strength += len(kings) * 5.0          # kings are very valuable
    strength += sum(c.points for c in hand) * 0.3  # raw point value
    return strength


def _choose_bid_heuristic(
    hand: list[Card],
    legal_bids: list[Contract | None],
) -> Contract | None:
    """Pick a bid based on hand strength."""
    strength = _hand_strength(hand)

    # Map strength thresholds to contracts
    bid_thresholds: list[tuple[float, Contract]] = [
        (25, Contract.THREE),
        (35, Contract.TWO),
        (45, Contract.ONE),
        (55, Contract.SOLO_THREE),
        (65, Contract.SOLO_TWO),
        (75, Contract.SOLO_ONE),
        (85, Contract.SOLO),
    ]

    best_bid: Contract | None = None
    for threshold, contract in bid_thresholds:
        if strength >= threshold and contract in legal_bids:
            best_bid = contract

    return best_bid


# ---------------------------------------------------------------------------
# LookaheadAgent
# ---------------------------------------------------------------------------

class LookaheadAgent:
    """Monte Carlo lookahead agent that evaluates all legal moves.

    Parameters
    ----------
    n_simulations : int
        Number of random worlds to sample per legal move.
        Higher = stronger but slower. Default 50.
    name : str
        Display name for the agent.
    rng : random.Random | None
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        n_simulations: int = 50,
        name: str = "Lookahead",
        rng: random.Random | None = None,
        perfect_information: bool = False,
    ):
        self._n_simulations = n_simulations
        self._name = name
        self._rng = rng or random.Random()
        self._perfect_information = perfect_information

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_simulations(self) -> int:
        return self._n_simulations

    @n_simulations.setter
    def n_simulations(self, value: int) -> None:
        self._n_simulations = max(1, value)

    # ---- PlayerPort implementation ----

    async def choose_bid(
        self,
        state: GameState,
        player_idx: int,
        legal_bids: list[Contract | None],
    ) -> Contract | None:
        return _choose_bid_heuristic(state.hands[player_idx], legal_bids)

    async def choose_king(
        self,
        state: GameState,
        player_idx: int,
        callable_kings: list[Card],
    ) -> Card:
        # Call the king of a suit where we have the most cards
        hand = state.hands[player_idx]
        suit_counts: dict[str, int] = {}
        for c in hand:
            if c.card_type == CardType.SUIT and c.suit is not None:
                suit_counts[c.suit.value] = suit_counts.get(c.suit.value, 0) + 1

        best_king = callable_kings[0]
        best_count = -1
        for king in callable_kings:
            if king.suit is not None:
                count = suit_counts.get(king.suit.value, 0)
                if count > best_count:
                    best_count = count
                    best_king = king
        return best_king

    async def choose_talon_group(
        self,
        state: GameState,
        player_idx: int,
        talon_groups: list[list[Card]],
    ) -> int:
        # Pick the group with the most total points
        best_idx = 0
        best_points = -1
        for i, group in enumerate(talon_groups):
            pts = sum(c.points for c in group)
            if pts > best_points:
                best_points = pts
                best_idx = i
        return best_idx

    async def choose_discard(
        self,
        state: GameState,
        player_idx: int,
        must_discard: int,
    ) -> list[Card]:
        hand = state.hands[player_idx]
        discardable = [
            c for c in hand
            if not c.is_king and c.card_type != CardType.TAROK
        ]
        if len(discardable) < must_discard:
            discardable = [c for c in hand if not c.is_king]
        # Discard lowest-point cards
        discardable.sort(key=lambda c: c.points)
        return discardable[:must_discard]

    async def choose_announcements(
        self,
        state: GameState,
        player_idx: int,
    ) -> list[Announcement]:
        return []

    async def choose_announce_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> int:
        return 0  # always pass on announcements

    async def choose_card(
        self,
        state: GameState,
        player_idx: int,
        legal_plays: list[Card],
    ) -> Card:
        """Evaluate every legal play and pick the best one."""
        if len(legal_plays) == 1:
            return legal_plays[0]

        best_card = legal_plays[0]
        best_score = float("-inf")

        eval_fn = _evaluate_move_perfect if self._perfect_information else _evaluate_move
        for card in legal_plays:
            avg_score = eval_fn(
                state, player_idx, card, self._n_simulations, self._rng
            )
            if avg_score > best_score:
                best_score = avg_score
                best_card = card

        return best_card

    # ---- Trainer-compatibility stubs ----

    def set_training(self, training: bool) -> None:
        """No-op — heuristic agent doesn't train."""
        pass

    def clear_experiences(self) -> None:
        """No-op — nothing to clear."""
        pass

    def finalize_game(self, reward: float) -> None:
        """No-op — no experiences to reward."""
        pass

    @property
    def experiences(self) -> list:
        """No experiences — always empty."""
        return []
