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

    # Contract (9 one-hot: none + KLOP + 8 biddable including berac)
    contract_vec = [0.0] * 9
    if state.contract:
        contract_list = list(Contract)
        idx = contract_list.index(state.contract)
        if idx < 9:
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

    # Bidding context: highest bid so far (9 one-hot: no_bid + 8 contracts)
    bid_vec = [0.0] * 9
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
    9 +   # contract (incl. berac)
    3 +   # phase
    1 +   # partner_known
    1 +   # tricks_won_by_team
    1 +   # tricks_played
    5 +   # decision_type (now 5: bid, king, talon, card, announce)
    9 +   # highest_bid (incl. berac)
    4 +   # passed_players
    4 +   # hand_strength
    4 +   # announcements_made
    5     # kontra_levels
)  # = 265
# Oracle critic sees all opponent hands (Perfect Training, Imperfect Execution)
ORACLE_EXTRA_SIZE = 3 * 54  # 3 opponent hand vectors
ORACLE_STATE_SIZE = STATE_SIZE + ORACLE_EXTRA_SIZE  # 263 + 162 = 425


def encode_oracle_state(
    state: GameState, player_idx: int, decision_type: DecisionType = DecisionType.CARD_PLAY
) -> torch.Tensor:
    """Encode perfect-information state for the oracle critic.

    Extends the regular imperfect-info state with all opponent hands,
    giving the critic access to hidden information during training.
    The actor never sees this — only the critic uses it for value estimation.
    """
    regular = encode_state(state, player_idx, decision_type)

    extra: list[float] = []
    for offset in range(1, state.num_players):
        opp_idx = (player_idx + offset) % state.num_players
        hand_vec = [0.0] * 54
        for card in state.hands[opp_idx]:
            hand_vec[CARD_TO_IDX[card]] = 1.0
        extra.extend(hand_vec)

    return torch.cat([regular, torch.tensor(extra, dtype=torch.float32)])

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
