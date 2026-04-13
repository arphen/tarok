"""Core RL kernel — shared between server and training-lab.

Contains the neural network, state encoding, compute backends, and
experience types. No game logic (that's in Rust).
"""

from tarok.core.encoding import (
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
    BID_ACTION_SIZE,
    KING_ACTIONS,
    KING_ACTION_SIZE,
    SUIT_TO_IDX,
    TALON_ACTION_SIZE,
    CARD_ACTION_SIZE,
    ANNOUNCE_ACTION_SIZE,
    ANNOUNCE_PASS,
    ANNOUNCE_IDX_TO_ANN,
    KONTRA_IDX_TO_KEY,
)
from tarok.core.network import TarokNet
from tarok.core.experience import Experience
from tarok.core.compute import ComputeBackend, create_backend
