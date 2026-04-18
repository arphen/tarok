"""Play one full game of Tarok.

Phase transitions are driven by the Rust engine — Python never checks
what contract type was bid or whether a declarer exists.  It just asks
"what phase are we in?" and dispatches to the right handler.
"""

from __future__ import annotations

from dataclasses import dataclass

import tarok_engine as te

from tarok.entities import (
    Card,
    DECK,
    Contract,
    GameState,
    Phase,
    PlayerRole,
)
from tarok.ports.observer_port import GameObserverPort
from tarok.ports.player_port import PlayerPort
from tarok.ports.score_breakdown_parser_port import ScoreBreakdownParserPort


# ---------------------------------------------------------------------------
# Rust ↔ Python translation tables (static, built once)
# ---------------------------------------------------------------------------

_U8_TO_CONTRACT: dict[int, Contract] = {
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
_CONTRACT_TO_U8: dict[Contract, int] = {v: k for k, v in _U8_TO_CONTRACT.items()}

_U8_TO_PHASE: dict[int, Phase] = {
    0: Phase.DEALING,
    1: Phase.BIDDING,
    2: Phase.KING_CALLING,
    3: Phase.TALON_EXCHANGE,
    4: Phase.ANNOUNCEMENTS,
    5: Phase.TRICK_PLAY,
    6: Phase.SCORING,
    7: Phase.FINISHED,
}

_U8_TO_ROLE = {0: PlayerRole.DECLARER, 1: PlayerRole.PARTNER, 2: PlayerRole.OPPONENT}

# ---------------------------------------------------------------------------
# Snapshot types (read-only views for observers / PlayerPort)
# ---------------------------------------------------------------------------


@dataclass
class TrickResult:
    lead_player: int
    cards: list[tuple[int, Card]]
    winner_player: int
    points: int

    def winner(self) -> int:
        return self.winner_player


@dataclass
class _Bid:
    player: int
    contract: Contract | None


# ---------------------------------------------------------------------------
# Session — all accumulated state lives here
# ---------------------------------------------------------------------------


class _Session:
    """Mutable game session.  Holds Rust state + Python bookkeeping for
    observers.  Every method that touches the Rust engine lives here so
    the top-level loop stays trivially readable.
    """

    def __init__(
        self,
        gs: te.RustGameState,
        players: list[PlayerPort],
        observer: GameObserverPort,
        allow_berac: bool,
        score_breakdown_parser: ScoreBreakdownParserPort | None,
    ):
        self.gs = gs
        self.players = players
        self.observer = observer
        self.allow_berac = allow_berac
        self.score_breakdown_parser = score_breakdown_parser

        self.tricks: list[TrickResult] = []
        self.bids: list[_Bid] = []
        self.initial_tarok_counts: dict[int, int] = {}

    # -- snapshot -----------------------------------------------------------

    def snap(
        self,
        *,
        current_trick: tuple[int, list[tuple[int, Card]]] | None = None,
        talon_revealed: list[list[int]] | None = None,
    ) -> GameState:
        """Build a read-only Python GameState from the current Rust state."""
        gs = self.gs
        s = GameState.__new__(GameState)
        s.dealer = gs.dealer
        s.num_players = 4
        s.phase = _U8_TO_PHASE.get(gs.phase, Phase.TRICK_PLAY)
        s.hands = [[DECK[i] for i in gs.hand(p)] for p in range(4)]
        s.contract = _U8_TO_CONTRACT.get(gs.contract) if gs.contract is not None else None
        s.declarer = getattr(gs, "declarer", None)
        s.partner = getattr(gs, "partner", None)
        ck = getattr(gs, "called_king", None)
        s.called_king = DECK[ck] if ck is not None else None
        s.talon = [DECK[i] for i in gs.talon()]
        s.talon_revealed = [[DECK[i] for i in g] for g in talon_revealed] if talon_revealed else []
        s.put_down = []
        s.tricks = list(self.tricks)
        if current_trick is not None:
            lead, cards = current_trick
            s.current_trick = type("Trick", (), {"lead_player": lead, "cards": list(cards)})()
        else:
            s.current_trick = None
        s.bids = list(self.bids)
        s.announcements = {}
        s.kontra_levels = {}
        s.roles = {}
        for p in range(4):
            try:
                s.roles[p] = _U8_TO_ROLE.get(gs.get_role(p), PlayerRole.OPPONENT)
            except Exception:
                s.roles[p] = PlayerRole.OPPONENT
        s.scores = {}
        s.current_player = getattr(gs, "current_player", 0)
        s._rust_gs = gs
        return s

    # -- phases -------------------------------------------------------------

    async def deal(self) -> None:
        await self.observer.on_game_start(self.snap())
        await self.observer.on_deal(self.snap())

    async def bidding(self) -> None:
        gs = self.gs
        passed = [False] * 4
        highest: int | None = None
        winner: int | None = None
        bidder = (gs.dealer + 2) % 4

        for _ in range(20):  # upper bound on bidding rounds
            active = [i for i in range(4) if not passed[i]]
            if (len(active) <= 1 and winner is not None) or len(active) == 0:
                break

            gs.current_player = bidder
            agent = self.players[bidder]

            # Legal bids (Rust u8 → Python Contract)
            rust_legal = gs.legal_bids(bidder)
            if not self.allow_berac:
                berac_u8 = _CONTRACT_TO_U8[Contract.BERAC]
                rust_legal = [b for b in rust_legal if b != berac_u8]
            py_legal: list[Contract | None] = [None]  # pass always legal
            for b in rust_legal:
                if b is not None:
                    c = _U8_TO_CONTRACT.get(b)
                    if c is not None:
                        py_legal.append(c)

            bid = await agent.choose_bid(self.snap(), bidder, py_legal)

            if bid is None:
                passed[bidder] = True
                gs.add_bid(bidder, None)
                self.bids.append(_Bid(player=bidder, contract=None))
            else:
                rust_u8 = _CONTRACT_TO_U8.get(bid)
                if rust_u8 not in rust_legal:
                    # illegal bid → treat as pass
                    passed[bidder] = True
                    gs.add_bid(bidder, None)
                    self.bids.append(_Bid(player=bidder, contract=None))
                else:
                    gs.add_bid(bidder, rust_u8)
                    self.bids.append(_Bid(player=bidder, contract=bid))
                    highest = rust_u8
                    winner = bidder

            # advance to next non-passed bidder
            nxt = bidder
            for _ in range(4):
                nxt = (nxt + 1) % 4
                if not passed[nxt]:
                    break
            bidder = nxt
            gs.current_player = bidder

            await self.observer.on_bid(
                self.bids[-1].player,
                self.bids[-1].contract,
                self.snap(),
            )

        gs.resolve_bidding(winner=winner, contract=highest)

        # Drain any stale queued actions from the bidding phase.
        # The last on_bid sends state with current_player=next_bidder;
        # if bidding ends before that player bids, they may have already
        # sent a premature response that must be discarded.
        for p in self.players:
            drain_queue = getattr(p, "drain_queue", None)
            if callable(drain_queue):
                drain_queue()

        py_contract = _U8_TO_CONTRACT.get(gs.contract) if gs.contract is not None else None
        decl = getattr(gs, "declarer", None) or 0
        await self.observer.on_contract_won(decl, py_contract, self.snap())

        # stash initial tarok counts for post-game metrics
        for p in range(4):
            self.initial_tarok_counts[p] = sum(1 for c in gs.hand(p) if c < 22)

    async def king_calling(self) -> None:
        gs = self.gs
        declarer = gs.declarer
        if declarer is None:
            return
        callable_idxs = gs.callable_kings()
        if not callable_idxs:
            return

        gs.current_player = declarer
        py_callable = [DECK[i] for i in callable_idxs]
        card = await self.players[declarer].choose_king(
            self.snap(),
            declarer,
            py_callable,
        )
        card_idx = DECK.index(card) if card in DECK else callable_idxs[0]
        if card_idx not in callable_idxs:
            card_idx = callable_idxs[0]

        gs.apply_king_call(card_idx)
        await self.observer.on_king_called(declarer, DECK[card_idx], self.snap())

    async def talon_exchange(self) -> None:
        gs = self.gs
        declarer = gs.declarer
        if declarer is None:
            return
        groups = gs.build_talon_groups()
        gs.set_talon_revealed(groups)

        py_groups = [[DECK[i] for i in g] for g in groups]
        await self.observer.on_talon_revealed(py_groups, self.snap(talon_revealed=groups))

        group_idx = await self.players[declarer].choose_talon_group(
            self.snap(talon_revealed=groups),
            declarer,
            py_groups,
        )
        if group_idx >= len(groups):
            group_idx = 0
        picked = groups[group_idx]

        gs.apply_talon_pick(group_idx)
        await self.observer.on_talon_group_picked(self.snap())

        talon_cards = len(picked)
        discards = await self.players[declarer].choose_discard(
            self.snap(),
            declarer,
            talon_cards,
        )
        discarded_idxs = [DECK.index(c) for c in discards]
        gs.apply_discards(discarded_idxs)

        await self.observer.on_talon_exchanged(
            self.snap(),
            picked=[DECK[i] for i in picked],
            discarded=discards,
        )

    async def trick_play(self) -> None:
        gs = self.gs
        lead = gs.current_player

        while gs.phase == te.PHASE_TRICK_PLAY:
            trick_cards: list[tuple[int, Card]] = []
            gs.start_trick(lead)
            gs.current_player = lead
            await self.observer.on_trick_start(
                self.snap(current_trick=(lead, trick_cards)),
            )

            for offset in range(4):
                player = (lead + offset) % 4
                gs.current_player = player
                legal = gs.legal_plays(player)
                py_legal = [DECK[i] for i in legal]

                card = await self.players[player].choose_card(
                    self.snap(current_trick=(lead, trick_cards)),
                    player,
                    py_legal,
                )
                card_idx = DECK.index(card) if card in DECK else legal[0]
                if card_idx not in legal:
                    card_idx = legal[0]

                gs.play_card(player, card_idx)
                trick_cards.append((player, DECK[card_idx]))
                gs.current_player = (player + 1) % 4

                await self.observer.on_card_played(
                    player,
                    DECK[card_idx],
                    self.snap(current_trick=(lead, trick_cards)),
                )

            winner, points = gs.finish_trick()
            result = TrickResult(
                lead_player=lead,
                cards=trick_cards,
                winner_player=winner,
                points=points,
            )
            self.tricks.append(result)

            await self.observer.on_trick_won(result, winner, self.snap())
            lead = winner

    def score(self) -> tuple[GameState, dict[int, int]]:
        gs = self.gs
        gs.phase = te.PHASE_SCORING
        scores_arr = gs.score_game()
        scores = {i: int(scores_arr[i]) for i in range(4)}

        py_state = self.snap()
        py_state.phase = Phase.FINISHED
        py_state.scores = scores
        py_state.initial_tarok_counts = self.initial_tarok_counts

        return py_state, scores

    def breakdown(self) -> dict | None:
        if self.score_breakdown_parser is None:
            return None
        try:
            raw = self.score_breakdown_parser.parse(self.gs.score_game_breakdown_json())
            return {
                "breakdown": {
                    "contract": raw.get("contract"),
                    "mode": raw.get("mode"),
                    "declarer_won": raw.get("declarer_won"),
                    "declarer_points": raw.get("declarer_points"),
                    "opponent_points": raw.get("opponent_points"),
                    "lines": raw.get("lines", []),
                },
                "trick_summary": raw.get("trick_summary", []),
            }
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class NullObserver:
    """Observer that does nothing — used when no UI is attached."""

    async def on_game_start(self, state):
        pass

    async def on_deal(self, state):
        pass

    async def on_bid(self, player, bid, state):
        pass

    async def on_contract_won(self, player, contract, state):
        pass

    async def on_king_called(self, player, king, state):
        pass

    async def on_talon_revealed(self, groups, state):
        pass

    async def on_talon_group_picked(self, state):
        pass

    async def on_talon_exchanged(self, state, picked=None, discarded=None):
        pass

    async def on_trick_start(self, state):
        pass

    async def on_card_played(self, player, card, state):
        pass

    async def on_rule_verified(self, player, rule, state):
        pass

    async def on_trick_won(self, trick, winner, state):
        pass

    async def on_game_end(self, scores, state, breakdown=None):
        pass


_PHASE_HANDLERS = {
    te.PHASE_BIDDING: "bidding",
    te.PHASE_KING_CALLING: "king_calling",
    te.PHASE_TALON_EXCHANGE: "talon_exchange",
    te.PHASE_TRICK_PLAY: "trick_play",
}


async def play_game(
    players: list[PlayerPort],
    *,
    dealer: int = 0,
    observer: GameObserverPort | None = None,
    allow_berac: bool = True,
    score_breakdown_parser: ScoreBreakdownParserPort | None = None,
    preset_hands: list[list[int]] | None = None,
    preset_talon: list[int] | None = None,
) -> tuple[GameState, dict[int, int]]:
    """Play one full game of Tarok.

    Returns (final_state, {player_idx: score}).
    """
    gs = te.RustGameState(dealer)
    if preset_hands is not None and preset_talon is not None:
        gs.deal_hands(preset_hands, preset_talon)
    else:
        gs.deal()

    session = _Session(
        gs,
        players,
        observer or NullObserver(),
        allow_berac,
        score_breakdown_parser,
    )

    await session.deal()

    while gs.phase not in (te.PHASE_SCORING, te.PHASE_FINISHED):
        handler_name = _PHASE_HANDLERS.get(gs.phase)
        if handler_name is None:
            break
        await getattr(session, handler_name)()

    py_state, scores = session.score()
    bd = session.breakdown()

    try:
        await session.observer.on_game_end(scores, py_state, breakdown=bd)
    except TypeError:
        await session.observer.on_game_end(scores, py_state)

    return py_state, scores
