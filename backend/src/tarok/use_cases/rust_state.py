"""Rust->Python state bridge owned by use-cases.

This module centralizes representational helpers used by orchestration code.
It intentionally contains no game mechanics; Rust remains the source of truth.
"""

from __future__ import annotations

from typing import Any

from tarok.entities import Contract, DECK, GameState, Phase, PlayerRole, Card


_RUST_U8_TO_PY_CONTRACT: dict[int, Contract] = {
    0: Contract.KLOP,
    1: Contract.THREE,
    2: Contract.TWO,
    3: Contract.ONE,
    4: Contract.SOLO_THREE,
    5: Contract.SOLO_TWO,
    6: Contract.SOLO_ONE,
    7: Contract.SOLO,
    8: Contract.BERAC,
    9: Contract.BARVNI_VALAT,
}

_PY_CONTRACT_TO_RUST_U8 = {v: k for k, v in _RUST_U8_TO_PY_CONTRACT.items()}

_BID_IDX_TO_RUST: list[int | None] = [None]
for _contract in (
    Contract.THREE,
    Contract.TWO,
    Contract.ONE,
    Contract.SOLO_THREE,
    Contract.SOLO_TWO,
    Contract.SOLO_ONE,
    Contract.SOLO,
    Contract.BERAC,
):
    _BID_IDX_TO_RUST.append(_PY_CONTRACT_TO_RUST_U8[_contract])

_RUST_PHASE_TO_PY = {
    0: Phase.DEALING,
    1: Phase.BIDDING,
    2: Phase.KING_CALLING,
    3: Phase.TALON_EXCHANGE,
    4: Phase.ANNOUNCEMENTS,
    5: Phase.TRICK_PLAY,
    6: Phase.SCORING,
    7: Phase.FINISHED,
}

_CURRENT_BIDDER_UNSET = object()


class _CurrentTrickSnapshot:
    """Current trick snapshot for observer/UI state."""

    def __init__(self, lead_player: int, cards: list[tuple[int, Card]]):
        self.lead_player = lead_player
        self.cards = cards


def _is_klop(contract_u8: int) -> bool:
    return contract_u8 == 0


def _is_solo(contract_u8: int) -> bool:
    return contract_u8 in (4, 5, 6, 7)


def _is_berac(contract_u8: int) -> bool:
    return contract_u8 == 8


def _talon_cards(contract_u8: int) -> int:
    return {1: 3, 2: 2, 3: 1, 4: 3, 5: 2, 6: 1}.get(contract_u8, 0)


def _build_talon_groups(talon_idxs: list[int], talon_cards: int) -> list[list[int]]:
    group_size = 6 // (6 // talon_cards) if talon_cards in (1, 2, 3) else talon_cards
    groups: list[list[int]] = []
    for i in range(0, len(talon_idxs), group_size):
        groups.append(talon_idxs[i : i + group_size])
    return groups


def _fallback_legal_bids_from_history(
    *,
    player_idx: int,
    dealer: int,
    bid_history: list[Any],
) -> list[int | None]:
    """Compute legal rust bid ids from snapshot history.

    Defensive fallback for websocket snapshots when Rust legal_bids is
    temporarily unavailable or returns an invalid pass-only result.
    """
    forehand = (dealer + 1) % 4
    is_forehand = player_idx == forehand

    strength = {
        1: 1,  # Three
        2: 2,  # Two
        3: 3,  # One
        4: 4,  # Solo Three
        5: 5,  # Solo Two
        6: 6,  # Solo One
        7: 8,  # Solo
        8: 7,  # Berac
    }

    highest: int | None = None
    highest_strength = -1
    for b in bid_history:
        contract = getattr(b, "contract", None)
        if contract is None:
            continue
        rust = _PY_CONTRACT_TO_RUST_U8.get(contract)
        if rust is None:
            continue
        s = strength.get(rust, -1)
        if s > highest_strength:
            highest_strength = s
            highest = rust

    result: list[int | None] = [None]
    for c in (1, 2, 3, 4, 5, 6, 7, 8):
        if c == 1 and not is_forehand:
            continue
        if highest is None:
            result.append(c)
            continue
        if is_forehand:
            legal = strength[c] >= strength.get(highest, -1)
        else:
            legal = strength[c] > strength.get(highest, -1)
        if legal:
            result.append(c)
    return result


def _build_py_state_from_rust(
    gs,
    completed_tricks: list[Any] | None = None,
    *,
    bids: list[Any] | None = None,
    current_bidder: int | None | object = _CURRENT_BIDDER_UNSET,
    current_trick: tuple[int, list[tuple[int, Card]]] | None = None,
    talon_revealed: list[list[int]] | None = None,
) -> GameState:
    """Build a lightweight Python GameState view from Rust state."""
    state = GameState.__new__(GameState)
    state.dealer = gs.dealer
    state.num_players = 4
    rust_phase = gs.phase if hasattr(gs, "phase") else 5
    state.phase = _RUST_PHASE_TO_PY.get(rust_phase, Phase.TRICK_PLAY)

    state.hands = [[DECK[idx] for idx in gs.hand(p)] for p in range(4)]

    contract_u8 = gs.contract if hasattr(gs, "contract") else None
    state.contract = _RUST_U8_TO_PY_CONTRACT.get(contract_u8) if contract_u8 is not None else None
    state.declarer = getattr(gs, "declarer", None)
    state.partner = getattr(gs, "partner", None)

    called_king_idx = getattr(gs, "called_king", None)
    state.called_king = DECK[called_king_idx] if called_king_idx is not None else None

    talon_idxs = gs.talon() if hasattr(gs, "talon") else []
    state.talon = [DECK[idx] for idx in talon_idxs]
    if talon_revealed is not None:
        state.talon_revealed = [[DECK[idx] for idx in g] for g in talon_revealed]
    else:
        state.talon_revealed = []
    state.put_down = []

    state.tricks = list(completed_tricks or [])
    if current_trick is not None:
        lead_player, cards = current_trick
        state.current_trick = _CurrentTrickSnapshot(lead_player=lead_player, cards=list(cards))
    else:
        state.current_trick = None
    state.bids = list(bids or [])
    state.announcements = {}
    state.kontra_levels = {}
    state.roles = {}
    role_map = {0: PlayerRole.DECLARER, 1: PlayerRole.PARTNER, 2: PlayerRole.OPPONENT}
    for p in range(4):
        try:
            rust_role = gs.get_role(p)
            state.roles[p] = role_map.get(rust_role, PlayerRole.OPPONENT)
        except Exception:
            state.roles[p] = PlayerRole.OPPONENT
    state.scores = {}
    state.current_player = getattr(gs, "current_player", 0)
    if current_bidder is _CURRENT_BIDDER_UNSET:
        state.current_bidder = state.current_player
    else:
        state.current_bidder = current_bidder

    if hasattr(gs, "legal_bids") and callable(getattr(gs, "legal_bids", None)):
        legal_bids_cache: dict[int, list[int | None]] = {}
        snapshot_bids = list(bids or [])
        snapshot_dealer = int(getattr(gs, "dealer", 0))

        def _legal_bids(player_idx: int, _cache=legal_bids_cache, _gs=gs) -> list[int | None]:
            if player_idx not in _cache:
                try:
                    computed = list(_gs.legal_bids(player_idx))
                    if computed == [None]:
                        computed = _fallback_legal_bids_from_history(
                            player_idx=player_idx,
                            dealer=snapshot_dealer,
                            bid_history=snapshot_bids,
                        )
                    _cache[player_idx] = computed
                except Exception:
                    _cache[player_idx] = _fallback_legal_bids_from_history(
                        player_idx=player_idx,
                        dealer=snapshot_dealer,
                        bid_history=snapshot_bids,
                    )
            return list(_cache[player_idx])

        state.legal_bids = _legal_bids
    if hasattr(gs, "legal_plays") and callable(getattr(gs, "legal_plays", None)):
        legal_plays_cache: dict[int, list[Card]] = {}

        def _legal_plays(player_idx: int, _cache=legal_plays_cache, _gs=gs) -> list[Card]:
            if player_idx not in _cache:
                try:
                    plays = [DECK[idx] for idx in _gs.legal_plays(player_idx)]
                    if not plays:
                        hand_cards = [DECK[idx] for idx in _gs.hand(player_idx)]
                        _cache[player_idx] = hand_cards if hand_cards else []
                    else:
                        _cache[player_idx] = plays
                except Exception:
                    try:
                        _cache[player_idx] = [DECK[idx] for idx in _gs.hand(player_idx)]
                    except Exception:
                        _cache[player_idx] = []
            return list(_cache[player_idx])

        state.legal_plays = _legal_plays
    if hasattr(gs, "callable_kings") and callable(getattr(gs, "callable_kings", None)):
        callable_kings_cache: list[Card] | None = None

        def _callable_kings(_gs=gs) -> list[Card]:
            nonlocal callable_kings_cache
            if callable_kings_cache is None:
                try:
                    callable_kings_cache = [DECK[idx] for idx in _gs.callable_kings()]
                except Exception:
                    callable_kings_cache = []
            return list(callable_kings_cache)

        state.callable_kings = _callable_kings

    state._rust_gs = gs

    return state


def _build_py_state_stub(
    dealer: int,
    contract: Contract | None,
    declarer: int | None,
    gs,
    initial_tarok_counts: dict,
    completed_tricks: list[Any],
    bid_history: list[Any],
) -> GameState:
    """Build a minimal Python GameState with fields the trainer needs."""
    state = GameState.__new__(GameState)
    state.dealer = dealer
    state.contract = contract
    state.declarer = declarer
    state.partner = getattr(gs, "partner", None)
    state.phase = Phase.FINISHED
    state.num_players = 4
    state.hands = [[DECK[idx] for idx in gs.hand(p)] for p in range(4)]
    state.tricks = list(completed_tricks)
    state.current_trick = None
    state.bids = list(bid_history)
    state.announcements = {}
    state.kontra_levels = {}
    state.talon = []
    state.talon_revealed = []
    state.put_down = []
    state.called_king = None
    state.scores = {}
    state.current_player = getattr(gs, "current_player", 0)
    state.initial_tarok_counts = initial_tarok_counts

    rust_role_map = {0: PlayerRole.DECLARER, 1: PlayerRole.PARTNER, 2: PlayerRole.OPPONENT}
    roles: dict[int, PlayerRole] = {}
    for p in range(4):
        try:
            rust_role = gs.get_role(p) if hasattr(gs, "get_role") else 2
            roles[p] = rust_role_map.get(rust_role, PlayerRole.OPPONENT)
        except Exception:
            roles[p] = PlayerRole.OPPONENT
    state.roles = roles

    return state


__all__ = [
    "_RUST_U8_TO_PY_CONTRACT",
    "_PY_CONTRACT_TO_RUST_U8",
    "_BID_IDX_TO_RUST",
    "_RUST_PHASE_TO_PY",
    "_CURRENT_BIDDER_UNSET",
    "_is_klop",
    "_is_solo",
    "_is_berac",
    "_talon_cards",
    "_build_talon_groups",
    "_build_py_state_from_rust",
    "_build_py_state_stub",
]