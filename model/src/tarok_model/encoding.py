"""State encoding — converts game state into tensor representation for the neural network.

Supports multiple decision types: bidding, king calling, talon selection, and card play.

v2 encoding adds:
  - 3×54 belief probability vectors (opponent card likelihoods)
  - 4 card-play-count features per opponent (cards played per suit category)
  - Trick lead suit & trick position features
"""

from __future__ import annotations

from enum import Enum

import torch

from tarok.entities import (
    Card, CardType, DECK, Suit,
    Announcement, Contract, GameState, KontraLevel, Phase, Team,
)

# Build a card-to-index mapping
CARD_TO_IDX: dict[Card, int] = {card: idx for idx, card in enumerate(DECK)}


class DecisionType(Enum):
    BID = 0
    KING_CALL = 1
    TALON_PICK = 2
    CARD_PLAY = 3
    ANNOUNCE = 4


class GameMode(Enum):
    SOLO = "solo"
    KLOP_BERAC = "klop_berac"
    PARTNER_PLAY = "partner_play"
    COLOR_VALAT = "color_valat"


def contract_to_game_mode(contract: Contract | None) -> GameMode:
    """Map contract to the card-play head family used by v4 models."""
    if contract in (Contract.SOLO_THREE, Contract.SOLO_TWO, Contract.SOLO_ONE, Contract.SOLO):
        return GameMode.SOLO
    if contract in (Contract.KLOP, Contract.BERAC):
        return GameMode.KLOP_BERAC
    if contract == Contract.BARVNI_VALAT:
        return GameMode.COLOR_VALAT
    return GameMode.PARTNER_PLAY


# Action space sizes per decision type
BID_ACTION_SIZE = 9       # pass + 8 biddable contracts (including berac)
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


def _encode_state_into(
    buf: torch.Tensor,
    state: GameState,
    player_idx: int,
    decision_type: DecisionType = DecisionType.CARD_PLAY,
) -> None:
    """Write state features into a pre-zeroed buffer (in-place)."""
    o = 0  # running offset

    # Cards in hand (54 binary)
    for card in state.hands[player_idx]:
        buf[o + CARD_TO_IDX[card]] = 1.0
    o += 54

    # Cards already played (54 binary)
    for trick in state.tricks:
        for _, card in trick.cards:
            buf[o + CARD_TO_IDX[card]] = 1.0
    o += 54

    # Cards in current trick (54 binary)
    if state.current_trick:
        for _, card in state.current_trick.cards:
            buf[o + CARD_TO_IDX[card]] = 1.0
    o += 54

    # Talon cards visible to this player (54 binary)
    if state.talon_revealed and player_idx == state.declarer:
        for group in state.talon_revealed:
            for card in group:
                buf[o + CARD_TO_IDX[card]] = 1.0
    o += 54

    # Player position relative to dealer (4 one-hot)
    relative_pos = (player_idx - state.dealer) % state.num_players
    buf[o + relative_pos] = 1.0
    o += 4

    # Contract (10 one-hot: none + KLOP + 8 biddable + BARVNI_VALAT)
    if state.contract:
        contract_list = list(Contract)
        idx = contract_list.index(state.contract)
        if idx < 10:
            buf[o + idx] = 1.0
    o += 10

    # Phase encoding (3 features: bidding, trick_play, other)
    if state.phase == Phase.BIDDING:
        buf[o] = 1.0
    elif state.phase == Phase.TRICK_PLAY:
        buf[o + 1] = 1.0
    else:
        buf[o + 2] = 1.0
    o += 3

    # Partner known
    if state.is_partner_revealed:
        buf[o] = 1.0
    o += 1

    # Tricks won by my team (normalized 0-1)
    my_team = state.get_team(player_idx)
    my_tricks = sum(
        1 for t in state.tricks if state.get_team(t.winner()) == my_team
    )
    buf[o] = my_tricks / 12.0
    o += 1

    # Tricks played (normalized 0-1)
    buf[o] = state.tricks_played / 12.0
    o += 1

    # Decision type (5 one-hot)
    buf[o + decision_type.value] = 1.0
    o += 5

    # Bidding context: highest bid so far (9 one-hot: no_bid + 8 contracts)
    bids_with_contract = [b.contract for b in state.bids if b.contract is not None]
    if bids_with_contract:
        highest = max(bids_with_contract, key=lambda c: c.strength)
        buf[o + BID_TO_IDX.get(highest, 0)] = 1.0
    else:
        buf[o] = 1.0  # No bid yet
    o += 9

    # Which players have passed (4 binary, relative to dealer)
    passed_set = {b.player for b in state.bids if b.contract is None}
    for i in range(state.num_players):
        p = (state.dealer + 1 + i) % state.num_players
        if p in passed_set:
            buf[o + i] = 1.0
    o += 4

    # Hand strength features (normalized)
    hand = state.hands[player_idx]
    tarok_count = sum(1 for c in hand if c.card_type == CardType.TAROK)
    high_taroks = sum(1 for c in hand if c.card_type == CardType.TAROK and c.value >= 15)
    king_count = sum(1 for c in hand if c.is_king)
    suits_in_hand = {c.suit for c in hand if c.suit is not None}
    void_count = 4 - len(suits_in_hand)
    buf[o] = tarok_count / 12.0
    buf[o + 1] = high_taroks / 7.0
    buf[o + 2] = king_count / 4.0
    buf[o + 3] = void_count / 4.0
    o += 4

    # Announcements made (4 binary: trula, kings, pagat, valat — by either team)
    ann_set: set[Announcement] = set()
    for anns in state.announcements.values():
        ann_set.update(anns)
    for i, ann in enumerate(
        [Announcement.TRULA, Announcement.KINGS, Announcement.PAGAT_ULTIMO, Announcement.VALAT]
    ):
        if ann in ann_set:
            buf[o + i] = 1.0
    o += 4

    # Kontra levels (5 features, normalized: game + 4 bonuses, each 0/0.33/0.67/1)
    _KONTRA_KEYS = ["game", Announcement.TRULA.value, Announcement.KINGS.value,
                    Announcement.PAGAT_ULTIMO.value, Announcement.VALAT.value]
    for i, key in enumerate(_KONTRA_KEYS):
        level = state.kontra_levels.get(key, KontraLevel.NONE)
        buf[o + i] = (level.value - 1) / 7.0  # 0→0, 1→0.14, 3→0.43, 7→1.0
    o += 5

    # Role one-hot (3 features: is_declarer, is_partner, is_opposition)
    if state.declarer is not None:
        if player_idx == state.declarer:
            buf[o] = 1.0      # is_declarer
        elif state.partner is not None and player_idx == state.partner:
            buf[o + 1] = 1.0  # is_partner
        elif state.get_team(player_idx) == Team.DECLARER_TEAM:
            # Partner not yet revealed but we ARE on declarer team (we're the hidden partner)
            buf[o + 1] = 1.0  # is_partner
        else:
            buf[o + 2] = 1.0  # is_opposition
    # During bidding (no declarer yet), all three stay 0 — role unknown
    o += 3

    # ===================================================================
    # v2 FEATURES: Belief tracking + card-play statistics
    # ===================================================================

    # --- 3×54 opponent belief probabilities ---
    # Compute the set of cards whose location is unknown to this player
    known_cards: set[int] = set()
    # Own hand
    for card in state.hands[player_idx]:
        known_cards.add(CARD_TO_IDX[card])
    # Played cards (completed tricks)
    for trick in state.tricks:
        for _, card in trick.cards:
            known_cards.add(CARD_TO_IDX[card])
    # Current trick
    if state.current_trick:
        for _, card in state.current_trick.cards:
            known_cards.add(CARD_TO_IDX[card])
    # Talon visible to declarer
    if state.talon_revealed and player_idx == state.declarer:
        for group in state.talon_revealed:
            for card in group:
                known_cards.add(CARD_TO_IDX[card])

    # Unknown cards = full deck minus known
    unknown_card_indices = [i for i in range(54) if i not in known_cards]
    num_unknown = len(unknown_card_indices)

    # Build per-opponent suit void constraints from trick history
    # If an opponent failed to follow suit when they could have, they are void
    opp_void_suits: dict[int, set[Suit]] = {i: set() for i in range(4) if i != player_idx}
    for trick in state.tricks:
        if not trick.cards:
            continue
        lead_player, lead_card = trick.cards[0]
        lead_suit = lead_card.suit  # None for taroks
        if lead_suit is None:
            continue  # tarok lead — no suit inference
        for p, card in trick.cards[1:]:
            if p == player_idx:
                continue
            if p in opp_void_suits and card.suit != lead_suit:
                # Player didn't follow suit → must be void in that suit
                opp_void_suits[p].add(lead_suit)

    # Compute uniform belief conditioned on void constraints
    # For each unknown card and each opponent, compute whether the opponent
    # could hold it (not void in that suit). Then distribute probability.
    for opp_offset in range(1, 4):
        opp_idx = (player_idx + opp_offset) % 4
        void_suits = opp_void_suits.get(opp_idx, set())
        for cidx in unknown_card_indices:
            card = DECK[cidx]
            # If opponent is void in this card's suit, probability = 0
            if card.suit is not None and card.suit in void_suits:
                buf[o + cidx] = 0.0
            else:
                # Uniform prior across possible holders (3 opponents + hidden talon/put_down)
                # Rough approximation: 1/3 for each opponent if no constraints
                buf[o + cidx] = 1.0 / 3.0
        o += 54

    # --- Per-opponent card-play counts (3×4 = 12 features) ---
    # For each opponent: [taroks_played, suit_cards_played, kings_played, total_played]
    # All normalized to 0-1
    for opp_offset in range(1, 4):
        opp_idx = (player_idx + opp_offset) % 4
        taroks_played = 0
        suit_played = 0
        kings_played = 0
        total_played = 0
        for trick in state.tricks:
            for p, card in trick.cards:
                if p == opp_idx:
                    total_played += 1
                    if card.card_type == CardType.TAROK:
                        taroks_played += 1
                    else:
                        suit_played += 1
                    if card.is_king:
                        kings_played += 1
        if state.current_trick:
            for p, card in state.current_trick.cards:
                if p == opp_idx:
                    total_played += 1
                    if card.card_type == CardType.TAROK:
                        taroks_played += 1
                    else:
                        suit_played += 1
                    if card.is_king:
                        kings_played += 1
        buf[o] = taroks_played / 12.0
        buf[o + 1] = suit_played / 12.0
        buf[o + 2] = kings_played / 4.0
        buf[o + 3] = total_played / 12.0
        o += 4

    # --- Trick context features (4 features) ---
    # Current trick position (0-3, normalized), lead suit one-hot (4+1 for tarok)
    if state.current_trick and state.current_trick.cards:
        trick_pos = len(state.current_trick.cards) / 4.0
        buf[o] = trick_pos
        lead_card = state.current_trick.cards[0][1]
        if lead_card.card_type == CardType.TAROK:
            buf[o + 1] = 1.0  # tarok lead
        elif lead_card.suit is not None:
            suit_idx = list(Suit).index(lead_card.suit)
            buf[o + 2 + suit_idx] = 1.0  # suit lead (4 positions)
    o += 6  # trick_pos(1) + tarok_lead(1) + suit_lead(4)


def encode_state(state: GameState, player_idx: int, decision_type: DecisionType = DecisionType.CARD_PLAY) -> torch.Tensor:
    """Encode visible game state for a specific player as a flat tensor.

    Uses a pre-allocated buffer to avoid per-call Python list allocation.
    """
    _state_buf.zero_()
    _encode_state_into(_state_buf, state, player_idx, decision_type)
    return _state_buf.clone()


# Compute STATE_SIZE from the feature layout
STATE_SIZE = (
    54 +  # hand
    54 +  # played
    54 +  # current_trick
    54 +  # talon_visible
    4 +   # player_position
    10 +  # contract (incl. berac + barvni_valat)
    3 +   # phase
    1 +   # partner_known
    1 +   # tricks_won_by_team
    1 +   # tricks_played
    5 +   # decision_type (now 5: bid, king, talon, card, announce)
    9 +   # highest_bid (incl. berac)
    4 +   # passed_players
    4 +   # hand_strength
    4 +   # announcements_made
    5 +   # kontra_levels
    3 +   # role (is_declarer, is_partner, is_opposition)
    # --- v2 features ---
    3 * 54 +  # opponent belief probabilities (3 opponents × 54 cards)
    3 * 4 +   # opponent card-play counts (3 opponents × 4 features)
    6         # trick context (position + lead suit)
)  # = 270 + 162 + 12 + 6 = 450
# Old state size for backward compat migration
_OLD_STATE_SIZE_V1 = 270
# Canonical offsets inside the flat state tensor.
CONTRACT_OFFSET = 220
CONTRACT_SIZE = 10
BELIEF_OFFSET = _OLD_STATE_SIZE_V1
# Oracle critic sees all opponent hands (Perfect Training, Imperfect Execution)
ORACLE_EXTRA_SIZE = 3 * 54  # 3 opponent hand vectors
ORACLE_STATE_SIZE = STATE_SIZE + ORACLE_EXTRA_SIZE  # 450 + 162 = 612

# Pre-allocated encoding buffers (one per process, safe for async single-threaded use)
_state_buf = torch.zeros(STATE_SIZE, dtype=torch.float32)
_oracle_buf = torch.zeros(ORACLE_STATE_SIZE, dtype=torch.float32)


def encode_oracle_state(
    state: GameState, player_idx: int, decision_type: DecisionType = DecisionType.CARD_PLAY
) -> torch.Tensor:
    """Encode perfect-information state for the oracle critic.

    Uses a pre-allocated buffer and the shared encoding helper.
    """
    _oracle_buf.zero_()
    _encode_state_into(_oracle_buf, state, player_idx, decision_type)

    o = STATE_SIZE
    for offset in range(1, state.num_players):
        opp_idx = (player_idx + offset) % state.num_players
        for card in state.hands[opp_idx]:
            _oracle_buf[o + CARD_TO_IDX[card]] = 1.0
        o += 54

    return _oracle_buf.clone()

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
