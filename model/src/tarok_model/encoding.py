"""State encoding — converts game state into tensor representation for the neural network.

Supports multiple decision types: bidding, king calling, talon selection, and card play.

v7 encoding (531 dims) — blank slate, no backward-compat:
  - Header card planes 0..107 (54+54): hand, active trick.
    (The v6 forced-retention plane is subsumed by belief-block pinning
     on the declarer column.  The v5 global played plane is subsumed by
     per-opponent played planes at 343..504.)
  - Base scalars 108..163 (56 dims): seat, contract, phase,
    tricks_played, decision_type, highest_bid, passed_players,
    announcements split own/opp (4+4), kontra, role one-hot,
    partner_rel one-hot.
  - Centaur trick context 164..174 (11 dims): team card-points (3) +
    trick leader rel (4) + trick currently-winning rel (4).
  - Belief block 175..336 (3×54).
  - Trick context 337..342 (6).
  - Per-opponent played planes 343..504 (3×54).  Declarer's plane
    additionally includes publicly-retired unpicked talon cards.
  - Per-opp tarok-void (3) + suit-void (12) + live kings (4) + live
    trula (3) + called-king suit (4) = 26 dims.
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


def _compute_talon_visibility(
    state: GameState,
) -> tuple[set[int], list[Card] | None]:
    """Return (unpicked_talon_indices, picked_group_cards).

    In Slovenian Tarok the entire talon is revealed to all players before
    the declarer picks one group.  A revealed group whose cards are no
    longer in ``state.talon`` has been picked; the remaining groups are
    publicly retired (everyone can see those cards will never appear in
    a trick).  The picked group identifies the cards declarer just took
    into their hand — of which taroks and kings cannot be discarded and
    are therefore publicly known to still be held by the declarer.
    """
    unpicked: set[int] = set()
    picked: list[Card] | None = None
    if not state.talon_revealed:
        return unpicked, picked
    talon_set = {CARD_TO_IDX[c] for c in state.talon}
    for group in state.talon_revealed:
        if any(CARD_TO_IDX[c] in talon_set for c in group):
            for c in group:
                unpicked.add(CARD_TO_IDX[c])
        elif picked is None:
            picked = list(group)
    return unpicked, picked


def _encode_state_into(
    buf: torch.Tensor,
    state: GameState,
    player_idx: int,
    decision_type: DecisionType = DecisionType.CARD_PLAY,
) -> None:
    """Write state features into a pre-zeroed buffer (in-place)."""
    o = 0  # running offset

    # --- v5 talon-visibility pre-computation ---
    unpicked_talon_idx, picked_group = _compute_talon_visibility(state)
    forced_retention: set[int] = set()
    if picked_group is not None:
        for c in picked_group:
            if c.card_type == CardType.TAROK or c.is_king:
                forced_retention.add(CARD_TO_IDX[c])

    # Cards in hand (54 binary)
    for card in state.hands[player_idx]:
        buf[o + CARD_TO_IDX[card]] = 1.0
    o += 54

    # Active trick plane (54 binary) — cards currently on the table.
    if state.current_trick:
        for _, card in state.current_trick.cards:
            buf[o + CARD_TO_IDX[card]] = 1.0
    o += 54

    # (v7: the forced-retention plane is gone — those cards are pinned
    #  onto the declarer's column of the belief block below, which is
    #  all the information it ever carried.)

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

    # Running team card-point totals (used by centaur block).  Team
    # tricks-won count is omitted in v7 — it's derivable and doesn't
    # drive decisions.
    my_team = state.get_team(player_idx)
    my_team_points = 0
    opp_team_points = 0
    for trick in state.tricks:
        winner = trick.winner()
        trick_pts = sum(c.points for _, c in trick.cards)
        if state.get_team(winner) == my_team:
            my_team_points += trick_pts
        else:
            opp_team_points += trick_pts

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

    # Announcements split by team (4 + 4 = 8 binary).  Own-team first,
    # opposition second.  Each 4-bit slot is {trula, kings, pagat, valat}.
    # During bidding the team partition is not yet defined — both slots
    # stay zero until declarer is known.
    _ANNS = [Announcement.TRULA, Announcement.KINGS, Announcement.PAGAT_ULTIMO, Announcement.VALAT]
    if state.declarer is not None:
        own_ann: set[Announcement] = set()
        opp_ann: set[Announcement] = set()
        for seat, anns in state.announcements.items():
            if state.get_team(seat) == my_team:
                own_ann.update(anns)
            else:
                opp_ann.update(anns)
        for i, ann in enumerate(_ANNS):
            if ann in own_ann:
                buf[o + i] = 1.0
            if ann in opp_ann:
                buf[o + 4 + i] = 1.0
    o += 8

    # Kontra levels (5 features, normalized: game + 4 bonuses, each 0/0.33/0.67/1)
    _KONTRA_KEYS = ["game", Announcement.TRULA.value, Announcement.KINGS.value,
                    Announcement.PAGAT_ULTIMO.value, Announcement.VALAT.value]
    for i, key in enumerate(_KONTRA_KEYS):
        level = state.kontra_levels.get(key, KontraLevel.NONE)
        buf[o + i] = (level.value - 1) / 7.0  # 0→0, 1→0.14, 3→0.43, 7→1.0
    o += 5

    # Role one-hot (3 features: is_declarer, is_partner, is_opposition).
    # The acting player always knows their own role once bidding ends.
    # All-zero ⇒ bidding still in progress.
    if state.declarer is not None:
        if player_idx == state.declarer:
            buf[o] = 1.0      # is_declarer
        elif state.partner is not None and player_idx == state.partner:
            buf[o + 1] = 1.0  # is_partner
        elif state.get_team(player_idx) == Team.DECLARER_TEAM:
            buf[o + 1] = 1.0  # is_partner (hidden — we know our own role)
        else:
            buf[o + 2] = 1.0  # is_opposition
    o += 3

    # Partner relative seat (4 one-hot; 0 = self is partner).  This
    # identifies *which seat* holds the called king.  Set only for
    # players who actually know:
    #   * the partner themselves (always — they hold the king),
    #   * any seat once the king has been publicly played,
    #   * a self-called-king declarer (partner == declarer).
    if state.partner is not None:
        known = (
            player_idx == state.partner
            or state.is_partner_revealed
            or state.declarer == state.partner
        )
        if known:
            rel = (state.partner - player_idx) % state.num_players
            buf[o + rel] = 1.0
    o += 4

    # --- v6 Centaur trick context (11 dims) ---
    current_trick_points = 0
    current_leader: int | None = None
    current_winner: int | None = None
    if state.current_trick and state.current_trick.cards:
        cards = state.current_trick.cards
        current_leader = cards[0][0]
        # Compute currently-winning seat under standard trick rules
        # (pagat / mond / skis specials do not change who beats whom
        # for partial tricks; we use the same comparison as the full
        # trick winner logic).
        lead_card = cards[0][1]
        lead_suit = lead_card.suit if lead_card.card_type == CardType.SUIT else None
        best_p, best_c = cards[0]
        for p, c in cards[1:]:
            if c.beats(best_c, lead_suit):
                best_p, best_c = p, c
        current_winner = best_p
        current_trick_points = sum(c.points for _, c in cards)

    buf[o] = my_team_points / 70.0
    buf[o + 1] = opp_team_points / 70.0
    buf[o + 2] = current_trick_points / 20.0
    o += 3

    # Trick leader relative seat (4 one-hot, 0 = self).
    if current_leader is not None:
        rel = (current_leader - player_idx) % state.num_players
        buf[o + rel] = 1.0
    o += 4

    # Trick currently-winning seat (4 one-hot, relative).
    if current_winner is not None:
        rel = (current_winner - player_idx) % state.num_players
        buf[o + rel] = 1.0
    o += 4

    # ===================================================================
    # v5 BELIEF BLOCK (162 dims, offsets 270..431)
    # 3 opponents × 54 cards, with:
    #   - known (hand / tricks / current trick / unpicked talon) → 0
    #   - suit-void / tarok-void → 0
    #   - forced-retention (taroks+kings picked from talon) → 1.0 on
    #     declarer column, 0 on other opp columns
    #   - otherwise uniform 1/3 prior
    # ===================================================================

    # Cards known to be out of every opponent's hand.
    known_cards: set[int] = set()
    for card in state.hands[player_idx]:
        known_cards.add(CARD_TO_IDX[card])
    for trick in state.tricks:
        for _, card in trick.cards:
            known_cards.add(CARD_TO_IDX[card])
    if state.current_trick:
        for _, card in state.current_trick.cards:
            known_cards.add(CARD_TO_IDX[card])
    known_cards.update(unpicked_talon_idx)

    unknown_card_indices = [i for i in range(54) if i not in known_cards]

    # Build per-opponent suit-void and tarok-void constraints from trick
    # history.
    opp_void_suits: dict[int, set[Suit]] = {i: set() for i in range(4) if i != player_idx}
    opp_tarok_void: dict[int, bool] = {i: False for i in range(4) if i != player_idx}
    for trick in state.tricks:
        if not trick.cards:
            continue
        lead_player, lead_card = trick.cards[0]
        lead_suit = lead_card.suit
        if lead_suit is not None:
            for p, card in trick.cards[1:]:
                if p == player_idx or p not in opp_void_suits:
                    continue
                if card.suit != lead_suit:
                    opp_void_suits[p].add(lead_suit)
        elif lead_card.card_type == CardType.TAROK:
            for p, card in trick.cards[1:]:
                if p == player_idx or p not in opp_tarok_void:
                    continue
                if card.card_type != CardType.TAROK:
                    opp_tarok_void[p] = True

    declarer = state.declarer
    for opp_offset in range(1, 4):
        opp_idx = (player_idx + opp_offset) % 4
        is_declarer_col = declarer is not None and opp_idx == declarer
        void_suits = opp_void_suits.get(opp_idx, set())
        tarok_void = opp_tarok_void.get(opp_idx, False)
        for cidx in unknown_card_indices:
            # Forced retention pins the card onto the declarer column.
            if cidx in forced_retention:
                if is_declarer_col:
                    buf[o + cidx] = 1.0
                continue
            card = DECK[cidx]
            if card.card_type == CardType.TAROK:
                impossible = tarok_void
            else:
                impossible = card.suit is not None and card.suit in void_suits
            if not impossible:
                buf[o + cidx] = 1.0 / 3.0
        o += 54

    # --- Trick context features (6 dims) ---
    if state.current_trick and state.current_trick.cards:
        trick_pos = len(state.current_trick.cards) / 4.0
        buf[o] = trick_pos
        lead_card = state.current_trick.cards[0][1]
        if lead_card.card_type == CardType.TAROK:
            buf[o + 1] = 1.0
        elif lead_card.suit is not None:
            suit_idx = list(Suit).index(lead_card.suit)
            buf[o + 2 + suit_idx] = 1.0
    o += 6

    # ===================================================================
    # v6 PER-OPPONENT PLAYED PLANES (3 × 54 = 162 dims)
    # Binary plane per opponent: cards they have personally played.
    # The declarer's plane additionally includes publicly-retired
    # unpicked talon cards — this absorbs the old global played plane
    # into an identity-preserving representation.
    # ===================================================================
    opp_played_planes: dict[int, set[int]] = {i: set() for i in range(4)}
    for trick in state.tricks:
        for p, card in trick.cards:
            opp_played_planes[p].add(CARD_TO_IDX[card])
    if state.current_trick:
        for p, card in state.current_trick.cards:
            opp_played_planes[p].add(CARD_TO_IDX[card])
    if state.declarer is not None:
        opp_played_planes[state.declarer].update(unpicked_talon_idx)
    for opp_offset in range(1, 4):
        opp_idx = (player_idx + opp_offset) % 4
        for cidx in opp_played_planes[opp_idx]:
            buf[o + cidx] = 1.0
        o += 54

    # --- Per-opponent tarok-void flags (3 dims) ---
    for opp_offset in range(1, 4):
        opp_idx = (player_idx + opp_offset) % 4
        if opp_tarok_void.get(opp_idx, False):
            buf[o] = 1.0
        o += 1

    # --- Per-opponent suit-void flags (3 × 4 = 12 dims) ---
    _SUIT_ORDER = list(Suit)
    for opp_offset in range(1, 4):
        opp_idx = (player_idx + opp_offset) % 4
        void_suits = opp_void_suits.get(opp_idx, set())
        for s_i, s in enumerate(_SUIT_ORDER):
            if s in void_suits:
                buf[o + s_i] = 1.0
        o += 4

    # --- Live kings / live trula identity features ---
    # Scan tricks + current trick for which kings have fallen and which
    # of the trula taroks (pagat=0, mond=20, skis=21) have been played.
    king_played_suits: set[Suit] = set()
    trula_played_bits = [False, False, False]  # pagat, mond, skis

    def _check_card(card: Card) -> None:
        if card.is_king and card.suit is not None:
            king_played_suits.add(card.suit)
        if card.card_type == CardType.TAROK:
            cidx_local = CARD_TO_IDX[card]
            if cidx_local == 0:
                trula_played_bits[0] = True
            elif cidx_local == 20:
                trula_played_bits[1] = True
            elif cidx_local == 21:
                trula_played_bits[2] = True

    for trick in state.tricks:
        for _p, card in trick.cards:
            _check_card(card)
    if state.current_trick:
        for _p, card in state.current_trick.cards:
            _check_card(card)

    for s_i, s in enumerate(_SUIT_ORDER):
        if s not in king_played_suits:
            buf[o + s_i] = 1.0
    o += 4

    for i, played in enumerate(trula_played_bits):
        if not played:
            buf[o + i] = 1.0
    o += 3

    # --- Called-king suit one-hot (4 dims, public once king is called) ---
    if state.called_king is not None and state.called_king.suit is not None:
        suit_idx = _SUIT_ORDER.index(state.called_king.suit)
        buf[o + suit_idx] = 1.0
    o += 4


def encode_state(state: GameState, player_idx: int, decision_type: DecisionType = DecisionType.CARD_PLAY) -> torch.Tensor:
    """Encode visible game state for a specific player as a flat tensor.

    Uses a pre-allocated buffer to avoid per-call Python list allocation.
    """
    _state_buf.zero_()
    _encode_state_into(_state_buf, state, player_idx, decision_type)
    return _state_buf.clone()


# Compute STATE_SIZE from the feature layout (v7)
STATE_SIZE = (
    54 +  # hand
    54 +  # active trick
    4 +   # player_position
    10 +  # contract (incl. berac + barvni_valat)
    3 +   # phase
    1 +   # tricks_played
    5 +   # decision_type
    9 +   # highest_bid
    4 +   # passed_players
    8 +   # announcements_made (own-team + opp-team, 4 each)
    5 +   # kontra_levels
    3 +   # role
    4 +   # partner_rel (relative seat one-hot; all-zero when unknown)
    # --- v6 centaur trick context ---
    3 +   # team points (mine, opp, current_trick)
    4 +   # trick leader relative seat
    4 +   # trick currently-winning seat
    # --- belief + trick context ---
    3 * 54 +  # opponent belief probabilities
    6 +       # trick context
    # --- per-opponent played planes ---
    3 * 54 +  # per-opponent played identity planes (declarer plane ∪ unpicked talon)
    # --- public memory ---
    3 +       # per-opponent tarok-void flags
    3 * 4 +   # per-opponent suit-void flags
    4 +       # live kings (one-hot per suit)
    3 +       # live trula (pagat, mond, skis)
    4         # called-king suit one-hot
)  # = 531
# v1 base size: belief block always starts immediately after it.
_OLD_STATE_SIZE_V1 = 175
# Canonical offsets inside the flat state tensor.
CONTRACT_OFFSET = 112
CONTRACT_SIZE = 10
BELIEF_OFFSET = _OLD_STATE_SIZE_V1
# Oracle critic sees all opponent hands (Perfect Training, Imperfect Execution)
ORACLE_EXTRA_SIZE = 3 * 54  # 3 opponent hand vectors
ORACLE_STATE_SIZE = STATE_SIZE + ORACLE_EXTRA_SIZE  # 531 + 162 = 693

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
