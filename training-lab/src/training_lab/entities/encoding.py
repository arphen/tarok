"""Decision types and action space constants for Tarok RL.

This module contains ONLY raw indices, sizes, and enums — no dependency on
tarok.entities (Card, GameState, etc.).  The backend provides a bridge to
convert game-domain objects to these raw indices.

The Rust engine (tarok_engine) returns pre-encoded numpy arrays that map
directly to these constants, bypassing the game-domain layer entirely.
"""

from __future__ import annotations

from enum import IntEnum


class DecisionType(IntEnum):
    BID = 0
    KING_CALL = 1
    TALON_PICK = 2
    CARD_PLAY = 3
    ANNOUNCE = 4


# ---------------------------------------------------------------------------
# Action space sizes per decision type
# ---------------------------------------------------------------------------

BID_ACTION_SIZE = 9       # pass + 8 contracts
KING_ACTION_SIZE = 4      # one per suit
TALON_ACTION_SIZE = 6     # up to 6 talon groups
CARD_ACTION_SIZE = 54     # one per card in deck
ANNOUNCE_ACTION_SIZE = 10  # pass + 4 announcements + 5 kontras

ACTION_SIZES: dict[DecisionType, int] = {
    DecisionType.BID: BID_ACTION_SIZE,
    DecisionType.KING_CALL: KING_ACTION_SIZE,
    DecisionType.TALON_PICK: TALON_ACTION_SIZE,
    DecisionType.CARD_PLAY: CARD_ACTION_SIZE,
    DecisionType.ANNOUNCE: ANNOUNCE_ACTION_SIZE,
}

MAX_ACTION_SIZE = CARD_ACTION_SIZE  # 54 — largest head

# ---------------------------------------------------------------------------
# Bidding action maps (raw index ↔ contract strength)
# ---------------------------------------------------------------------------

# Index 0 = pass, 1..8 = contracts in ascending strength
BID_ACTIONS: list[str | None] = [
    None,       # 0: pass
    "KLOP",     # 1
    "THREE",    # 2
    "TWO",      # 3
    "ONE",      # 4
    "SOLO_THREE",  # 5
    "SOLO_TWO",    # 6
    "SOLO_ONE",    # 7
    "BERAC",       # 8
]

BID_TO_IDX: dict[str | None, int] = {name: i for i, name in enumerate(BID_ACTIONS)}

# ---------------------------------------------------------------------------
# King calling maps (raw index ↔ suit)
# ---------------------------------------------------------------------------

# Suit indices: HEARTS=0, DIAMONDS=1, SPADES=2, CLUBS=3
KING_ACTIONS: list[str] = ["HEARTS", "DIAMONDS", "SPADES", "CLUBS"]
SUIT_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(KING_ACTIONS)}

# ---------------------------------------------------------------------------
# Announcement action maps
# ---------------------------------------------------------------------------

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

ANNOUNCE_IDX_TO_NAME: dict[int, str] = {
    ANNOUNCE_TRULA: "TRULA",
    ANNOUNCE_KINGS: "KINGS",
    ANNOUNCE_PAGAT: "PAGAT_ULTIMO",
    ANNOUNCE_VALAT: "VALAT",
}

KONTRA_IDX_TO_KEY: dict[int, str] = {
    KONTRA_GAME: "game",
    KONTRA_TRULA: "trula",
    KONTRA_KINGS: "kings",
    KONTRA_PAGAT: "pagat_ultimo",
    KONTRA_VALAT: "valat",
}

# ---------------------------------------------------------------------------
# State encoding dimensions
# ---------------------------------------------------------------------------

STATE_SIZE = 450
ORACLE_EXTRA_SIZE = 3 * 54  # 3 opponent hand vectors
ORACLE_STATE_SIZE = STATE_SIZE + ORACLE_EXTRA_SIZE  # 612

# Old state size for backward compat migration
OLD_STATE_SIZE_V1 = 270

# ---------------------------------------------------------------------------
# Rust engine contract mappings (u8 repr from engine-rs game_state.rs)
# ---------------------------------------------------------------------------

RUST_U8_TO_CONTRACT_NAME: dict[int, str | None] = {
    0: "KLOP",
    1: "THREE",
    2: "TWO",
    3: "ONE",
    4: "SOLO_THREE",
    5: "SOLO_TWO",
    6: "SOLO_ONE",
    7: "SOLO",
    8: "BERAC",
    9: "BARVNI_VALAT",
}

RUST_U8_TO_BID_IDX: dict[int, int] = {
    # Rust u8 contract → BID_ACTIONS index (skipping KLOP and BARVNI_VALAT which aren't biddable)
    1: 2,  # THREE → idx 2
    2: 3,  # TWO → idx 3
    3: 4,  # ONE → idx 4
    4: 5,  # SOLO_THREE → idx 5
    5: 6,  # SOLO_TWO → idx 6
    6: 7,  # SOLO_ONE → idx 7
    7: 8,  # SOLO → idx 8 (actually BERAC in BID_ACTIONS)
    8: 8,  # BERAC → idx 8
}

# BID_ACTIONS index → Rust contract u8 (None = pass)
BID_IDX_TO_RUST: list[int | None] = [
    None,  # 0: pass
    0,     # 1: KLOP
    1,     # 2: THREE
    2,     # 3: TWO
    3,     # 4: ONE
    4,     # 5: SOLO_THREE
    5,     # 6: SOLO_TWO
    6,     # 7: SOLO_ONE
    8,     # 8: BERAC
]

# Rust decision type u8 → Python DecisionType
RUST_DT_MAP: dict[int, DecisionType] = {
    0: DecisionType.BID,
    1: DecisionType.KING_CALL,
    2: DecisionType.TALON_PICK,
    3: DecisionType.CARD_PLAY,
    4: DecisionType.ANNOUNCE,
}

# ---------------------------------------------------------------------------
# Deck layout helpers (raw index → suit, avoiding tarok.entities dependency)
# ---------------------------------------------------------------------------

NUM_TAROKS = 22
CARDS_PER_SUIT = 8
NUM_SUITS = 4

# Card index ranges: 0-21 taroks, 22-29 hearts, 30-37 diamonds, 38-45 spades, 46-53 clubs
KING_CARD_INDICES = [29, 37, 45, 53]  # highest card in each suit

def card_idx_to_suit(card_idx: int) -> int | None:
    """Map raw card index to suit index (0=hearts, 1=diamonds, 2=spades, 3=clubs).

    Returns None for taroks (indices 0-21).
    """
    if card_idx < NUM_TAROKS:
        return None
    return (card_idx - NUM_TAROKS) // CARDS_PER_SUIT

def is_tarok(card_idx: int) -> bool:
    return card_idx < NUM_TAROKS

def is_king(card_idx: int) -> bool:
    return card_idx in KING_CARD_INDICES

# ---------------------------------------------------------------------------
# Contract classification helpers
# ---------------------------------------------------------------------------

def is_klop(contract_u8: int | None) -> bool:
    return contract_u8 == 0

def is_solo(contract_u8: int | None) -> bool:
    return contract_u8 in (4, 5, 6, 7)  # SOLO_THREE through SOLO

def is_berac(contract_u8: int | None) -> bool:
    return contract_u8 == 8

def talon_cards_for_contract(contract_u8: int | None) -> int:
    """Number of talon cards the declarer picks for each contract."""
    if contract_u8 is None:
        return 0
    return {
        0: 0,  # KLOP
        1: 3,  # THREE
        2: 2,  # TWO
        3: 1,  # ONE
        4: 3,  # SOLO_THREE
        5: 2,  # SOLO_TWO
        6: 1,  # SOLO_ONE
        7: 0,  # SOLO (no talon)
        8: 0,  # BERAC
        9: 0,  # BARVNI_VALAT
    }.get(contract_u8, 0)
