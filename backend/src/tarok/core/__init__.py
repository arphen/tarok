"""Core RL kernel — shared between server and training-lab.

Contains the neural network, state encoding, compute backends, and
experience types. No game logic (that's in Rust).
"""

from tarok_model.encoding import (
    DecisionType,
    GameMode,
    contract_to_game_mode,
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
from tarok_model.network import TarokNet, TarokNetV4
from tarok.core.experience import Experience
from tarok_model.compute import ComputeBackend, create_backend
