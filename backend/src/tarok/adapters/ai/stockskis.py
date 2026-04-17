"""Thin adapter from Rust StockŠkis heuristic bots to the PlayerPort interface.

The Rust engine exposes ``v5_choose_*`` and ``m6_choose_*`` methods on
``RustGameState``.  This adapter wraps them so the RustGameLoop can call
``choose_card(state, player, legal)`` etc. like any other agent.
"""

from __future__ import annotations

from tarok.ports.player_port import PlayerPort


class RustStockskisPlayer(PlayerPort):
    """PlayerPort adapter for Rust-side StockŠkis heuristic bots.

    Parameters
    ----------
    variant : str
        ``"v5"`` or ``"m6"`` — selects which set of ``gs.{variant}_choose_*``
        methods to call.
    name : str
        Display name.
    """

    def __init__(self, *, variant: str = "v5", name: str = "StockŠkis"):
        self._variant = variant
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    # The RustGameLoop gives us the *RustGameState* directly via the
    # ``_rust_state`` attribute on the pystate object it constructs.
    # But currently the PlayerPort methods receive a *Python* GameState
    # representation.  The RustGameLoop stores the original RustGameState
    # in ``self._gs`` — but we don't have access to it from here.
    #
    # Instead we use the approach from the old wrappers: attach the Rust
    # game-state to the py_state as ``_rust_gs`` before calling the agent.
    # rust_game_loop already does this (``py_state._rust_gs = self._gs``).

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

        # Map Rust u8 contract → Python Contract enum
        from tarok.adapters.ai.rust_game_loop import _RUST_U8_TO_PY_CONTRACT
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
