"""Unit tests for websocket bidding state exposure to the UI."""
from unittest.mock import AsyncMock

import numpy as np
import pytest

from tarok.adapters.api.ws_observer import WebSocketObserver, _state_for_player
from tarok.adapters.ai import rust_game_loop as rgl
from tarok.adapters.ai.rust_game_loop import RustGameLoop, _build_py_state_from_rust
from tarok.entities import Contract, Phase


class DummyState:
    """Minimal state object for testing websocket serialization paths."""

    def __init__(self, *, phase: Phase, current_player: int, legal_bids: list[int | None]):
        self.phase = phase
        self.current_player = current_player
        self.hands = [[], [], [], []]
        self.talon_revealed = None
        self.bids = []
        self.contract = None
        self.declarer = None
        self.called_king = None
        self.is_partner_revealed = False
        self.partner = None
        self.current_trick = None
        self.tricks_played = 0
        self.scores = None
        self._legal_bids = legal_bids
        self.legal_bids_calls: list[int] = []

    def legal_bids(self, player_idx: int) -> list[int | None]:
        self.legal_bids_calls.append(player_idx)
        return list(self._legal_bids)


class FakeRustGS:
    """Minimal Rust-game-state stub for _build_py_state_from_rust tests."""

    def __init__(self) -> None:
        self.dealer = 0
        self.phase = 1  # bidding
        self.contract = None
        self.declarer = None
        self.partner = None
        self.called_king = None
        self.current_player = 0

    def hand(self, _player: int) -> list[int]:
        return []

    def talon(self) -> list[int]:
        return []

    def get_role(self, _player: int) -> int:
        return 2

    def legal_bids(self, player_idx: int) -> list[int | None]:
        assert player_idx == 0
        # Rust u8 ids: 7=SOLO, 3=ONE
        return [None, 7, 3]

    def legal_plays(self, player_idx: int) -> list[int]:
        assert player_idx == 0
        return [0, 10]


class FakeRustPassOnlyBidsGS(FakeRustGS):
    """Simulate a transient engine snapshot that reports pass-only bids."""

    def legal_bids(self, player_idx: int) -> list[int | None]:
        assert player_idx == 0
        return [None]


class _Bid:
    """Small helper for constructing state.bids entries in tests."""

    def __init__(self, player: int, contract):
        self.player = player
        self.contract = contract


def test_state_for_player_exposes_legal_bids_only_for_current_bidder() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]
    expected_legal_bids = [None, Contract.SOLO.value, Contract.ONE.value]
    state = DummyState(phase=Phase.BIDDING, current_player=0, legal_bids=expected_legal_bids)

    current_turn = _state_for_player(state, 0, names)
    other_turn = _state_for_player(state, 1, names)

    assert current_turn["legal_bids"] == expected_legal_bids
    assert other_turn["legal_bids"] is None
    assert state.legal_bids_calls == [0]


def test_state_for_player_does_not_compute_legal_bids_outside_bidding() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]
    state = DummyState(
        phase=Phase.KING_CALLING,
        current_player=0,
        legal_bids=[None, Contract.SOLO.value],
    )

    payload = _state_for_player(state, 0, names)

    assert payload["legal_bids"] is None
    assert state.legal_bids_calls == []


@pytest.mark.asyncio
async def test_observer_bid_event_contains_legal_bids_on_human_turn() -> None:
    ws = AsyncMock()
    observer = WebSocketObserver(ws, player_idx=0, player_names=["You", "AI-1", "AI-2", "AI-3"], ai_delay=0)
    expected_legal_bids = [None, Contract.SOLO.value, Contract.ONE.value, Contract.TWO.value]
    state = DummyState(phase=Phase.BIDDING, current_player=0, legal_bids=expected_legal_bids)

    await observer.on_bid(player=2, bid=None, state=state)

    ws.send_json.assert_awaited_once()
    message = ws.send_json.await_args.args[0]
    assert message["event"] == "bid"
    assert message["state"]["current_player"] == 0
    assert message["state"]["legal_bids"] == expected_legal_bids
    assert state.legal_bids_calls == [0]


@pytest.mark.asyncio
async def test_observer_bid_event_hides_legal_bids_when_not_human_turn() -> None:
    ws = AsyncMock()
    observer = WebSocketObserver(ws, player_idx=0, player_names=["You", "AI-1", "AI-2", "AI-3"], ai_delay=0)
    state = DummyState(
        phase=Phase.BIDDING,
        current_player=2,
        legal_bids=[None, Contract.SOLO.value, Contract.ONE.value],
    )

    await observer.on_bid(player=2, bid=None, state=state)

    message = ws.send_json.await_args.args[0]
    assert message["state"]["legal_bids"] is None
    assert state.legal_bids_calls == []


def test_live_rust_snapshot_should_offer_bids_on_human_turn() -> None:
    """Regression test for live websocket path:
    _build_py_state_from_rust -> _state_for_player should expose legal_bids.
    """
    names = ["You", "AI-1", "AI-2", "AI-3"]
    rust_gs = FakeRustGS()

    state = _build_py_state_from_rust(rust_gs)
    payload = _state_for_player(state, 0, names)

    # This is what UI needs to render bid buttons on your turn.
    assert payload["phase"] == "bidding"
    assert payload["current_player"] == 0
    assert payload["legal_bids"] is not None
    assert 7 in payload["legal_bids"]
    assert 3 in payload["legal_bids"]


def test_state_for_player_preserves_rust_bid_ids() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]
    state = DummyState(phase=Phase.BIDDING, current_player=0, legal_bids=[None, 4, 5, 6, 7, 8])

    payload = _state_for_player(state, 0, names)

    assert payload["legal_bids"] == [None, 4, 5, 6, 7, 8]


def test_bidding_turn_should_use_current_bidder_when_available() -> None:
    """Regression test: if current_bidder says it's our turn, UI should still get bids.

    Live snapshots can momentarily carry next/previous current_player while current_bidder
    remains authoritative for bidding turn.
    """
    names = ["You", "AI-1", "AI-2", "AI-3"]
    state = DummyState(
        phase=Phase.BIDDING,
        current_player=2,
        legal_bids=[None, Contract.SOLO.value, Contract.ONE.value],
    )
    state.current_bidder = 0

    payload = _state_for_player(state, 0, names)

    assert payload["legal_bids"] is not None


def test_bidding_turn_with_current_bidder_none_is_not_actionable() -> None:
    """Regression: terminal bidding snapshot must not expose a clickable pass button."""
    names = ["You", "AI-1", "AI-2", "AI-3"]
    state = DummyState(
        phase=Phase.BIDDING,
        current_player=0,
        legal_bids=[None, Contract.SOLO.value],
    )
    state.current_bidder = None

    payload = _state_for_player(state, 0, names)

    assert payload["current_player"] == 0
    assert payload["legal_bids"] is None


def test_rust_legal_bids_allow_only_solo_over_berac() -> None:
    """Regression: only Solo may overcall Berac."""
    import tarok_engine as te

    gs = te.RustGameState(0)
    gs.phase = te.PHASE_BIDDING

    # Player 1 (forehand when dealer=0) opens with Berac.
    gs.add_bid(1, 8)  # Rust contract code for BERAC

    # Player 0 must only be allowed to overcall with Solo.
    legal = gs.legal_bids(0)

    # Rust contract ids: 4=SOLO_THREE, 5=SOLO_TWO, 6=SOLO_ONE, 7=SOLO
    assert 4 not in legal
    assert 5 not in legal
    assert 6 not in legal
    assert 7 in legal


def test_rust_legal_bids_allow_solo_over_solo_one() -> None:
    """Regression: after Solo Two is raised to Solo One, Solo must remain legal."""
    import tarok_engine as te

    gs = te.RustGameState(0)
    gs.phase = te.PHASE_BIDDING

    gs.add_bid(1, 5)  # SOLO_TWO
    gs.add_bid(2, 6)  # SOLO_ONE

    legal = gs.legal_bids(0)

    assert 6 not in legal
    assert 7 in legal


def test_live_snapshot_after_solo_one_and_pass_keeps_solo_and_berac_for_next_player() -> None:
    import tarok_engine as te

    gs = te.RustGameState(0)
    gs.phase = te.PHASE_BIDDING

    # Bidding order for dealer=0 is 2 -> 3 -> 0 -> 1.
    gs.add_bid(2, 6)    # SOLO_ONE
    gs.add_bid(3, None) # PASS

    state = _build_py_state_from_rust(
        gs,
        bids=[rgl._PyBid(2, Contract.SOLO_ONE), rgl._PyBid(3, None)],
        current_bidder=0,
    )
    payload = _state_for_player(state, 0, ["You", "AI-1", "AI-2", "AI-3"])

    assert payload["current_player"] == 0
    assert payload["legal_bids"] == [None, 7, 8]


def test_live_snapshot_pass_only_legal_bids_uses_history_fallback() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]
    rust_gs = FakeRustPassOnlyBidsGS()

    # dealer=0 => order 2,3,0,1; after two passes, player 0 should have real bids.
    bids = [rgl._PyBid(2, None), rgl._PyBid(3, None)]
    state = _build_py_state_from_rust(rust_gs, bids=bids, current_bidder=0)
    payload = _state_for_player(state, 0, names)

    assert payload["phase"] == "bidding"
    assert payload["current_player"] == 0
    assert payload["legal_bids"] == [None, 2, 3, 4, 5, 6, 7, 8]


def test_live_rust_snapshot_should_offer_legal_plays_on_human_turn() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]
    rust_gs = FakeRustGS()
    rust_gs.phase = 5  # trick_play
    rust_gs.current_player = 0

    state = _build_py_state_from_rust(rust_gs)
    payload = _state_for_player(state, 0, names)

    assert payload["phase"] == "trick_play"
    assert payload["current_player"] == 0
    assert len(payload["legal_plays"]) == 2


def test_trick_start_snapshot_uses_lead_player_as_current_player() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]
    rust_gs = FakeRustGS()
    rust_gs.phase = 5  # trick_play
    rust_gs.current_player = 3  # stale previous value

    state = _build_py_state_from_rust(rust_gs, current_trick=(0, []))
    state.current_player = 0
    payload = _state_for_player(state, 0, names)

    assert payload["phase"] == "trick_play"
    assert payload["current_player"] == 0


def test_trick_play_legal_plays_falls_back_to_hand_when_empty() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]

    class _EmptyLegalRustGS(FakeRustGS):
        def hand(self, _player: int) -> list[int]:
            return [22, 23, 24]  # suit cards

        def legal_plays(self, _player_idx: int) -> list[int]:
            return []

    rust_gs = _EmptyLegalRustGS()
    rust_gs.phase = 5  # trick_play
    rust_gs.current_player = 0

    state = _build_py_state_from_rust(rust_gs)
    payload = _state_for_player(state, 0, names)

    assert payload["phase"] == "trick_play"
    assert payload["current_player"] == 0
    assert len(payload["hand"]) == 3
    assert len(payload["legal_plays"]) == 3


@pytest.mark.asyncio
async def test_bidding_does_not_reprompt_highest_bidder_after_all_others_pass(monkeypatch) -> None:
    """If bidding returns to highest bidder with everyone else passed, contract is won immediately."""

    class _ScriptedPlayer:
        def __init__(self, script: list[Contract | None]):
            self._script = list(script)

        async def choose_bid(self, _state, _player_idx: int, _legal_bids):
            return self._script.pop(0) if self._script else None

    class _Observer:
        def __init__(self):
            self.events: list[dict] = []

        async def on_bid(self, player, bid, state):
            self.events.append({"player": player, "bid": bid, "state": state})

    class _FakeGS:
        def __init__(self):
            self.dealer = 0
            self.current_player = 0
            self.declarer = None
            self.contract = None
            self.phase = 1

        def legal_bids(self, _player):
            return [1, 2, 3, 4, 5, 6, 7, 8]

        def add_bid(self, _player, _contract):
            return None

        def set_role(self, _player, _role):
            return None

        def encode_state(self, _player, _decision_type):
            return np.zeros((1143,), dtype=np.float32)

    # Bidding order for dealer=0 starts at player 2:
    # P2 Solo Three, P3 pass, P0 Solo Two, P1 pass, P2 pass -> should end.
    players = [
        _ScriptedPlayer([Contract.SOLO_TWO]),
        _ScriptedPlayer([None]),
        _ScriptedPlayer([Contract.SOLO_THREE, None]),
        _ScriptedPlayer([None]),
    ]
    obs = _Observer()
    loop = RustGameLoop(players, observer=obs)
    gs = _FakeGS()

    monkeypatch.setattr(
        rgl,
        "_build_py_state_from_rust",
        lambda _gs, _completed_tricks, **kwargs: {"current_bidder": kwargs.get("current_bidder")},
    )

    highest, winner = await loop._run_bidding(gs, completed_tricks=[], bid_history=[])

    assert highest == 5  # SOLO_TWO rust id
    assert winner == 0
    assert gs.phase == 3  # talon_exchange
    assert gs.current_player == 0
    assert obs.events[-1]["player"] == 2
    assert obs.events[-1]["bid"] is None
    # Final pass event should not advertise a next bidder turn.
    assert obs.events[-1]["state"]["current_bidder"] is None


def test_state_for_player_raises_on_pass_only_without_active_solo_bid() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]
    state = DummyState(
        phase=Phase.BIDDING,
        current_player=0,
        legal_bids=[None],
    )
    # Active non-pass bid exists, but it's not Solo.
    state.bids = [_Bid(player=2, contract=Contract.ONE)]

    with pytest.raises(RuntimeError, match="pass-only legal_bids without active solo bid"):
        _state_for_player(state, 0, names)


def test_state_for_player_allows_pass_only_when_active_solo_bid_exists() -> None:
    names = ["You", "AI-1", "AI-2", "AI-3"]
    state = DummyState(
        phase=Phase.BIDDING,
        current_player=0,
        legal_bids=[None],
    )
    state.bids = [_Bid(player=1, contract=Contract.SOLO)]
    assert _state_for_player(state, 0, names)["legal_bids"] == [None]