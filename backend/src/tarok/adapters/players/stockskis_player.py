"""PlayerPort adapter for Rust StockSkis heuristic bots."""

from __future__ import annotations

from tarok.entities import Card, Contract
from tarok.entities.game_types import DECK
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

    async def choose_bid(
        self, state, player_idx: int, legal_bids: list[Contract | None]
    ) -> Contract | None:
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return legal_bids[0] if legal_bids else None

        raw = self._call("choose_bid", gs, player_idx)
        if raw is None:
            return None

        from tarok.use_cases.rust_state import _RUST_U8_TO_PY_CONTRACT

        py_contract = _RUST_U8_TO_PY_CONTRACT.get(raw)
        if py_contract in legal_bids:
            return py_contract
        return None

    async def choose_card(self, state, player_idx: int, legal_plays: list[Card]) -> Card:
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return legal_plays[0]

        card_idx = self._call("choose_card", gs, player_idx)

        return DECK[card_idx]

    async def choose_king(self, state, player_idx: int, callable_kings: list[Card]) -> Card:
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return callable_kings[0]

        card_idx = self._call("choose_king", gs, player_idx)

        return DECK[card_idx]

    async def choose_talon_group(
        self, state, player_idx: int, talon_groups: list[list[Card]]
    ) -> int:
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return 0

        group_indices = [[card._idx for card in group] for group in talon_groups]
        return self._call("choose_talon_group", gs, player_idx, group_indices)

    async def choose_discard(self, state, player_idx: int, must_discard: int) -> list[Card]:
        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return []

        idxs = self._call("choose_discards", gs, player_idx)
        if must_discard <= 0:
            return []

        return [DECK[i] for i in idxs[:must_discard]]
