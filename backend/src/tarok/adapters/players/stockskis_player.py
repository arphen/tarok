"""PlayerPort adapter for Rust StockSkis heuristic bots."""

from __future__ import annotations

from tarok.ports.player_port import PlayerPort


class StockskisPlayer(PlayerPort):
    """PlayerPort adapter for Rust-side StockSkis bots (v5/m6)."""

    def __init__(self, *, variant: str = "v5", name: str = "StockSkis"):
        self._variant = variant
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _call(self, method_suffix: str, gs, player: int, *args):
        fn = getattr(gs, f"{self._variant}_{method_suffix}")
        return fn(player, *args)

    async def choose_bid(self, state, player, legal_bids):
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return legal_bids[0] if legal_bids else None

        raw = self._call("choose_bid", gs, player)
        if raw is None:
            return None

        from tarok.use_cases.rust_state import _RUST_U8_TO_PY_CONTRACT

        py_contract = _RUST_U8_TO_PY_CONTRACT.get(raw)
        if py_contract in legal_bids:
            return py_contract
        return None

    async def choose_card(self, state, player, legal_cards=None):
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return legal_cards[0] if legal_cards else None

        card_idx = self._call("choose_card", gs, player)
        from tarok.entities import DECK

        return DECK[card_idx]

    async def choose_king(self, state, player, callable_kings):
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return callable_kings[0] if callable_kings else None

        card_idx = self._call("choose_king", gs, player)
        from tarok.entities import DECK

        return DECK[card_idx]

    async def choose_talon_group(self, state, player, groups):
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return 0

        group_indices = [[card._idx for card in group] for group in groups]
        return self._call("choose_talon_group", gs, player, group_indices)

    async def choose_discard(self, state, player, num_cards):
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return []

        idxs = self._call("choose_discards", gs, player)
        from tarok.entities import DECK

        return [DECK[i] for i in idxs]
