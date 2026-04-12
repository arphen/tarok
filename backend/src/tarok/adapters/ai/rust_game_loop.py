"""Canonical game loop — all game mechanics run in Rust.

State management, dealing, legal moves, trick evaluation, and scoring all
happen in the Rust engine (tarok_engine).  Python is only responsible for
player decisions (neural-network inference, human input, heuristic bots).

Players with ``_decide_from_tensors`` use the fast tensor path (pre-encoded
state from Rust → action index).  All other PlayerPort implementations
(HumanPlayer, RandomPlayer, etc.) receive a lightweight Python state view
built from the Rust game state and return Python-domain objects that are
translated back to Rust indices.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tarok.adapters.ai.agent import RLAgent

try:
    import tarok_engine as te
except ImportError:
    te = None  # type: ignore[assignment]

from tarok.adapters.ai.encoding import (
    DecisionType,
    BID_ACTIONS,
    BID_TO_IDX,
    KING_ACTIONS,
    SUIT_TO_IDX,
    CARD_TO_IDX,
    ANNOUNCE_PASS,
    ANNOUNCE_IDX_TO_ANN,
    KONTRA_IDX_TO_KEY,
    encode_bid_mask,
    encode_king_mask,
    encode_talon_mask,
    encode_announce_mask,
    card_idx_to_card,
)
from tarok.entities.card import Card, CardType, Suit, DECK
from tarok.entities.game_state import (
    Announcement,
    Contract,
    GameState,
    KontraLevel,
    Phase,
    PlayerRole,
)
from tarok.ports.observer_port import GameObserverPort


# Map Rust contract u8 → Python Contract enum
_RUST_CONTRACT = {c.value if hasattr(c, 'value') else c: c for c in Contract}
# Build a fast lookup: Rust contract index → Python Contract
_U8_TO_CONTRACT: dict[int, Contract] = {}
for c in Contract:
    # Contract enum values: KLOP=-99, THREE=3, TWO=2, ONE=1, ...
    # Rust Contract enum: Klop=0, Three=1, Two=2, One=3, ...
    pass

# Rust Contract as u8 (from game_state.rs repr(u8)):
_RUST_U8_TO_PY_CONTRACT = {
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

# BID_ACTIONS index → Rust contract u8 or None (pass)
_BID_IDX_TO_RUST: list[int | None] = [None]  # index 0 = pass
for c in [Contract.THREE, Contract.TWO, Contract.ONE,
          Contract.SOLO_THREE, Contract.SOLO_TWO, Contract.SOLO_ONE,
          Contract.SOLO, Contract.BERAC]:
    _BID_IDX_TO_RUST.append(_PY_CONTRACT_TO_RUST_U8[c])


class NullObserver:
    """Default observer that does nothing."""

    async def on_game_start(self, state): pass
    async def on_deal(self, state): pass
    async def on_bid(self, player, bid, state): pass
    async def on_contract_won(self, player, contract, state): pass
    async def on_king_called(self, player, king, state): pass
    async def on_talon_revealed(self, groups, state): pass
    async def on_talon_group_picked(self, state): pass
    async def on_talon_exchanged(self, state, picked=None, discarded=None): pass
    async def on_trick_start(self, state): pass
    async def on_card_played(self, player, card, state): pass
    async def on_rule_verified(self, player, rule, state): pass
    async def on_trick_won(self, trick, winner, state): pass
    async def on_game_end(self, scores, state): pass


class _RustTrickSnapshot:
    """Minimal trick object for observers/tests backed by Rust outcomes."""

    def __init__(
        self,
        lead_player: int,
        cards: list[tuple[int, Card]],
        winner_player: int,
        points: int,
    ):
        self.lead_player = lead_player
        self.cards = cards
        self._winner_player = winner_player
        self._points = points

    def winner(self) -> int:
        return self._winner_player

    @property
    def points(self) -> int:
        return self._points


class _CurrentTrickSnapshot:
    """Current trick snapshot for observer/UI state."""

    def __init__(self, lead_player: int, cards: list[tuple[int, Card]]):
        self.lead_player = lead_player
        self.cards = cards


class _PyBid:
    """Simple bid record compatible with observer/state serializers."""

    def __init__(self, player: int, contract: Contract | None):
        self.player = player
        self.contract = contract


class RustGameLoop:
    """Plays a full game of Tarok using the Rust engine.

    Agents are still Python RLAgent objects — they receive pre-encoded
    tensors from Rust and return action indices.  All game-state mutation
    and encoding happens in Rust for maximum throughput.
    """

    def __init__(
        self,
        players: list,
        observer: GameObserverPort | None = None,
        rng: random.Random | None = None,
        allow_berac: bool = True,
    ):
        assert te is not None, "tarok_engine Rust extension not installed"
        assert len(players) == 4
        self._players = players
        self._observer: GameObserverPort = observer or NullObserver()  # type: ignore
        self._rng = rng or random.Random()
        self._allow_berac = allow_berac

    async def run(self, dealer: int = 0) -> tuple[GameState, dict[int, int]]:
        """Play one full game, returning (final_state, {player: score})."""
        gs = te.RustGameState(dealer)
        gs.deal()
        completed_tricks: list[_RustTrickSnapshot] = []
        bid_history: list[_PyBid] = []

        # Build a thin Python GameState for observer callbacks (optional)
        # Only populated when we actually have observer callbacks
        has_observer = not isinstance(self._observer, NullObserver)

        await self._observer.on_game_start(_build_py_state_from_rust(gs, completed_tricks, bids=bid_history))
        await self._observer.on_deal(_build_py_state_from_rust(gs, completed_tricks, bids=bid_history))

        # === BIDDING ===
        contract, declarer = await self._run_bidding(gs, completed_tricks, bid_history)

        if contract is None:
            # Re-deal on all pass (shouldn't happen with klop)
            return await self.run(dealer=(dealer + 1) % 4)

        py_contract = _RUST_U8_TO_PY_CONTRACT.get(contract)

        if py_contract is not None:
            contract_player = declarer if declarer is not None else -1
            await self._observer.on_contract_won(
                contract_player,
                py_contract,
                _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
            )

        # Store initial tarok counts for metrics
        initial_tarok_counts = {}
        for p in range(4):
            hand = gs.hand(p)
            initial_tarok_counts[p] = sum(1 for c in hand if c < 22)

        # === KING CALLING ===
        if declarer is not None and not _is_klop(contract) and not _is_solo(contract) and not _is_berac(contract):
            chosen_king_idx = await self._run_king_call(gs, declarer)
            if chosen_king_idx is not None:
                await self._observer.on_king_called(
                    declarer,
                    DECK[chosen_king_idx],
                    _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
                )

        # === TALON EXCHANGE ===
        talon_cards = _talon_cards(contract)
        if declarer is not None and talon_cards > 0 and not _is_klop(contract) and not _is_berac(contract):
            talon_groups = _build_talon_groups(gs.talon(), talon_cards)
            await self._observer.on_talon_revealed(
                [[DECK[idx] for idx in g] for g in talon_groups],
                _build_py_state_from_rust(gs, completed_tricks, bids=bid_history, talon_revealed=talon_groups),
            )
            picked_idxs, discarded_idxs = await self._run_talon_exchange(gs, declarer, talon_cards, talon_groups)
            await self._observer.on_talon_group_picked(
                _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
            )
            await self._observer.on_talon_exchanged(
                _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
                picked=[DECK[idx] for idx in picked_idxs],
                discarded=[DECK[idx] for idx in discarded_idxs],
            )

        # === ANNOUNCEMENTS ===
        if declarer is not None and not _is_klop(contract) and not _is_berac(contract):
            gs.phase = te.PHASE_ANNOUNCEMENTS
            # Simplified: skip announcements in Rust loop for now
            # (random play / early training rarely uses them)
            gs.phase = te.PHASE_TRICK_PLAY

        # === TRICK PLAY ===
        gs.phase = te.PHASE_TRICK_PLAY
        lead_player = (dealer + 1) % 4
        berac_early = False
        for trick_num in range(12):
            lead = lead_player
            trick_cards: list[tuple[int, Card]] = []
            gs.start_trick(lead_player)
            await self._observer.on_trick_start(
                _build_py_state_from_rust(gs, completed_tricks, bids=bid_history, current_trick=(lead, trick_cards)),
            )

            for offset in range(4):
                player = (lead_player + offset) % 4
                gs.current_player = player

                agent = self._players[player]
                legal_cards = gs.legal_plays(player)  # list of u8 card indices

                if hasattr(agent, '_decide_from_tensors'):
                    # Fast tensor path
                    state_t = torch.from_numpy(
                        gs.encode_state(player, te.DT_CARD_PLAY)
                    ).float()
                    legal_mask = torch.from_numpy(
                        gs.legal_plays_mask(player)
                    ).float()

                    oracle_t = None
                    if (
                        hasattr(agent, 'network')
                        and agent._training
                        and agent.network.oracle_critic_enabled
                    ):
                        oracle_t = torch.from_numpy(
                            gs.encode_oracle_state(player, te.DT_CARD_PLAY)
                        ).float()

                    action_idx = agent._decide_from_tensors(
                        state_t, legal_mask, DecisionType.CARD_PLAY, oracle_t,
                    )

                    if action_idx not in legal_cards:
                        action_idx = legal_cards[0]
                else:
                    # PlayerPort path
                    py_state = _build_py_state_from_rust(
                        gs,
                        completed_tricks,
                        bids=bid_history,
                        current_trick=(lead, trick_cards),
                    )
                    py_legal = [DECK[idx] for idx in legal_cards]
                    card = await agent.choose_card(py_state, player, py_legal)
                    action_idx = DECK.index(card) if card in DECK else legal_cards[0]
                    if action_idx not in legal_cards:
                        action_idx = legal_cards[0]

                gs.play_card(player, action_idx)
                trick_cards.append((player, DECK[action_idx]))
                # Broadcast next player so UI can surface legal actions immediately.
                next_player = (player + 1) % 4
                gs.current_player = next_player
                await self._observer.on_card_played(
                    player,
                    DECK[action_idx],
                    _build_py_state_from_rust(gs, completed_tricks, bids=bid_history, current_trick=(lead, trick_cards)),
                )

            # Finish trick
            winner, points = gs.finish_trick()
            trick_snapshot = _RustTrickSnapshot(
                lead_player=lead,
                cards=trick_cards,
                winner_player=winner,
                points=points,
            )
            completed_tricks.append(trick_snapshot)
            await self._observer.on_trick_won(
                trick_snapshot,
                winner,
                _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
            )
            lead_player = winner

            # Berač early termination: declarer wins a trick → instant loss
            if _is_berac(contract) and declarer is not None and winner == declarer:
                berac_early = True
                break

        # === SCORING ===
        gs.phase = te.PHASE_SCORING
        scores_arr = gs.score_game()
        scores = {i: int(scores_arr[i]) for i in range(4)}

        # Build a minimal Python GameState for compatibility with trainer
        py_state = _build_py_state_stub(
            dealer, py_contract, declarer,
            gs, initial_tarok_counts, completed_tricks, bid_history,
        )
        py_state.scores = scores

        await self._observer.on_game_end(scores, py_state)

        return py_state, scores

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    async def _run_bidding(
        self,
        gs,
        completed_tricks: list[_RustTrickSnapshot],
        bid_history: list[_PyBid],
    ) -> tuple[int | None, int | None]:
        """Run bidding, returns (rust_contract_u8, declarer_player)."""
        passed = [False] * 4
        highest: int | None = None  # Rust contract u8
        winning_player: int | None = None
        bidder = (gs.dealer + 1) % 4

        for _round in range(20):
            active = [i for i in range(4) if not passed[i]]
            if len(active) <= 1 and winning_player is not None:
                break
            if len(active) == 0:
                break

            gs.current_player = bidder
            agent = self._players[bidder]

            # Build legal bid list (Python Contract objects)
            rust_legal = gs.legal_bids(bidder)  # list of Option<u8>
            if not self._allow_berac:
                berac_u8 = _PY_CONTRACT_TO_RUST_U8[Contract.BERAC]
                rust_legal = [lb for lb in rust_legal if lb != berac_u8]
            py_legal_bids = [None]  # can always pass
            for lb in rust_legal:
                if lb is not None:
                    py_c = _RUST_U8_TO_PY_CONTRACT.get(lb)
                    if py_c is not None:
                        py_legal_bids.append(py_c)

            if hasattr(agent, '_decide_from_tensors'):
                # Fast tensor path (RLAgent)
                state_t = torch.from_numpy(
                    gs.encode_state(bidder, te.DT_BID)
                ).float()
                mask = encode_bid_mask(py_legal_bids)

                oracle_t = None
                if (
                    hasattr(agent, 'network')
                    and agent._training
                    and agent.network.oracle_critic_enabled
                ):
                    oracle_t = torch.from_numpy(
                        gs.encode_oracle_state(bidder, te.DT_BID)
                    ).float()

                action_idx = agent._decide_from_tensors(
                    state_t, mask, DecisionType.BID, oracle_t,
                )
                bid_contract = BID_ACTIONS[action_idx]  # Contract | None
            else:
                # PlayerPort path (HumanPlayer, RandomPlayer, etc.)
                py_state = _build_py_state_from_rust(gs, completed_tricks, bids=bid_history)
                bid_contract = await agent.choose_bid(py_state, bidder, py_legal_bids)

            if bid_contract is None:
                passed[bidder] = True
                gs.add_bid(bidder, None)
                bid_history.append(_PyBid(bidder, None))
            else:
                rust_u8 = _PY_CONTRACT_TO_RUST_U8.get(bid_contract)
                if rust_u8 not in rust_legal:
                    passed[bidder] = True
                    gs.add_bid(bidder, None)
                    bid_history.append(_PyBid(bidder, None))
                else:
                    gs.add_bid(bidder, rust_u8)
                    bid_history.append(_PyBid(bidder, bid_contract))
                    highest = rust_u8
                    winning_player = bidder

            # Next bidder
            next_bidder = bidder
            for _ in range(4):
                next_bidder = (next_bidder + 1) % 4
                if not passed[next_bidder]:
                    break
            bidder = next_bidder
            gs.current_player = bidder

            await self._observer.on_bid(
                bid_history[-1].player,
                bid_history[-1].contract,
                _build_py_state_from_rust(gs, completed_tricks, bids=bid_history),
            )

        # Resolve bidding
        if winning_player is not None and highest is not None:
            gs.declarer = winning_player
            gs.contract = highest
            gs.set_role(winning_player, 0)  # Declarer
            for i in range(4):
                if i != winning_player:
                    gs.set_role(i, 2)  # Opponent

            if _is_berac(highest):
                gs.phase = te.PHASE_TRICK_PLAY
                gs.current_player = (gs.dealer + 1) % 4
            elif _is_solo(highest):
                gs.phase = te.PHASE_TALON_EXCHANGE
            else:
                gs.phase = te.PHASE_KING_CALLING

            return highest, winning_player
        else:
            # All passed → Klop
            gs.contract = 0  # Klop = 0
            for i in range(4):
                gs.set_role(i, 2)  # Opponent
            gs.phase = te.PHASE_TRICK_PLAY
            gs.current_player = (gs.dealer + 1) % 4
            return 0, None

    async def _run_king_call(self, gs, declarer: int) -> int | None:
        """Declarer calls a king."""
        callable_idxs = gs.callable_kings()  # list of u8 card indices
        if not callable_idxs:
            return None

        gs.current_player = declarer

        py_callable = [DECK[idx] for idx in callable_idxs]
        agent = self._players[declarer]

        if hasattr(agent, '_decide_from_tensors'):
            # Fast tensor path
            state_t = torch.from_numpy(
                gs.encode_state(declarer, te.DT_KING_CALL)
            ).float()
            mask = encode_king_mask(py_callable)

            oracle_t = None
            if (
                hasattr(agent, 'network')
                and agent._training
                and agent.network.oracle_critic_enabled
            ):
                oracle_t = torch.from_numpy(
                    gs.encode_oracle_state(declarer, te.DT_KING_CALL)
                ).float()

            action_idx = agent._decide_from_tensors(
                state_t, mask, DecisionType.KING_CALL, oracle_t,
            )

            chosen_suit = KING_ACTIONS[action_idx]
            chosen_card_idx = None
            for idx in callable_idxs:
                card = DECK[idx]
                if card.suit == chosen_suit:
                    chosen_card_idx = idx
                    break
            if chosen_card_idx is None:
                chosen_card_idx = callable_idxs[0]
        else:
            # PlayerPort path
            py_state = _build_py_state_from_rust(gs)
            chosen_card = await agent.choose_king(py_state, declarer, py_callable)
            chosen_card_idx = DECK.index(chosen_card) if chosen_card in DECK else callable_idxs[0]
            if chosen_card_idx not in callable_idxs:
                chosen_card_idx = callable_idxs[0]

        gs.set_called_king(chosen_card_idx)

        # Identify partner (who has the called king)
        for p in range(4):
            if p != declarer:
                hand = gs.hand(p)
                if chosen_card_idx in hand:
                    gs.partner = p
                    gs.set_role(p, 1)  # Partner
                    break

        return chosen_card_idx

    async def _run_talon_exchange(
        self,
        gs,
        declarer: int,
        talon_cards: int,
        groups: list[list[int]],
    ) -> tuple[list[int], list[int]]:
        """Reveal talon, let declarer pick a group and discard."""
        num_groups = len(groups)

        gs.set_talon_revealed(groups)

        agent = self._players[declarer]

        if hasattr(agent, '_decide_from_tensors'):
            # Fast tensor path
            state_t = torch.from_numpy(
                gs.encode_state(declarer, te.DT_TALON_PICK)
            ).float()
            mask = encode_talon_mask(num_groups)

            oracle_t = None
            if (
                hasattr(agent, 'network')
                and agent._training
                and agent.network.oracle_critic_enabled
            ):
                oracle_t = torch.from_numpy(
                    gs.encode_oracle_state(declarer, te.DT_TALON_PICK)
                ).float()

            action_idx = agent._decide_from_tensors(
                state_t, mask, DecisionType.TALON_PICK, oracle_t,
            )
            if action_idx >= num_groups:
                action_idx = 0

            # Pick up the chosen group
            picked = groups[action_idx]
            for card_idx in picked:
                gs.add_to_hand(declarer, card_idx)
                gs.remove_from_talon(card_idx)

            # Heuristic discard (cheapest non-king non-tarok)
            hand = gs.hand(declarer)
            hand_cards = [(idx, DECK[idx]) for idx in hand]
            discardable = [
                (idx, c) for idx, c in hand_cards
                if not c.is_king and c.card_type != CardType.TAROK
            ]
            if len(discardable) < talon_cards:
                discardable = [
                    (idx, c) for idx, c in hand_cards if not c.is_king
                ]
            discardable.sort(key=lambda x: x[1].points)
            discarded_idxs = [idx for idx, _ in discardable[:talon_cards]]
            for idx in discarded_idxs:
                gs.remove_card(declarer, idx)
                gs.add_put_down(idx)
        else:
            # PlayerPort path
            py_groups = [[DECK[idx] for idx in g] for g in groups]
            py_state = _build_py_state_from_rust(gs)
            group_idx = await agent.choose_talon_group(py_state, declarer, py_groups)
            if group_idx >= num_groups:
                group_idx = 0

            picked = groups[group_idx]
            for card_idx in picked:
                gs.add_to_hand(declarer, card_idx)
                gs.remove_from_talon(card_idx)

            # Let player choose discards
            py_state = _build_py_state_from_rust(gs)
            discards = await agent.choose_discard(py_state, declarer, talon_cards)
            discarded_idxs = []
            for card in discards:
                card_idx = DECK.index(card)
                gs.remove_card(declarer, card_idx)
                gs.add_put_down(card_idx)
                discarded_idxs.append(card_idx)

        gs.phase = te.PHASE_TRICK_PLAY
        gs.current_player = (gs.dealer + 1) % 4
        return picked, discarded_idxs


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


def _build_py_state_from_rust(
    gs,
    completed_tricks: list[_RustTrickSnapshot] | None = None,
    *,
    bids: list[_PyBid] | None = None,
    current_trick: tuple[int, list[tuple[int, Card]]] | None = None,
    talon_revealed: list[list[int]] | None = None,
) -> GameState:
    """Build a lightweight Python GameState view from Rust state.

    Used to provide PlayerPort implementations (HumanPlayer, RandomPlayer,
    etc.) with the state they need to make decisions.  All game mechanics
    still run in Rust — this is a read-only snapshot.
    """
    state = GameState.__new__(GameState)
    state.dealer = gs.dealer
    state.num_players = 4
    rust_phase = gs.phase if hasattr(gs, "phase") else 5
    state.phase = _RUST_PHASE_TO_PY.get(rust_phase, Phase.TRICK_PLAY)

    # Hands as Python Card objects
    state.hands = [[DECK[idx] for idx in gs.hand(p)] for p in range(4)]

    # Contract / declarer / partner
    contract_u8 = gs.contract if hasattr(gs, 'contract') else None
    state.contract = _RUST_U8_TO_PY_CONTRACT.get(contract_u8) if contract_u8 is not None else None
    state.declarer = getattr(gs, 'declarer', None)
    state.partner = getattr(gs, 'partner', None)

    # Called king
    called_king_idx = getattr(gs, 'called_king', None)
    state.called_king = DECK[called_king_idx] if called_king_idx is not None else None

    # Talon
    talon_idxs = gs.talon() if hasattr(gs, 'talon') else []
    state.talon = [DECK[idx] for idx in talon_idxs]
    if talon_revealed is not None:
        state.talon_revealed = [[DECK[idx] for idx in g] for g in talon_revealed]
    else:
        state.talon_revealed = []
    state.put_down = []

    # Bids / roles / trick state for observers and UI legal-action display
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
    state.current_player = getattr(gs, 'current_player', 0)

    return state


def _build_py_state_stub(
    dealer: int,
    contract: Contract | None,
    declarer: int | None,
    gs,
    initial_tarok_counts: dict,
    completed_tricks: list[_RustTrickSnapshot],
    bid_history: list[_PyBid],
) -> GameState:
    """Build a minimal Python GameState with fields the trainer needs."""
    state = GameState.__new__(GameState)
    state.dealer = dealer
    state.contract = contract
    state.declarer = declarer
    state.partner = getattr(gs, 'partner', None)
    state.phase = Phase.FINISHED
    state.num_players = 4
    state.hands = [[DECK[idx] for idx in gs.hand(p)] for p in range(4)]
    # Populate tricks as a list of correct length so tricks_played works
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
    state.initial_tarok_counts = initial_tarok_counts

    # Build roles dict from Rust state
    _RUST_ROLE_MAP = {0: PlayerRole.DECLARER, 1: PlayerRole.PARTNER, 2: PlayerRole.OPPONENT}
    roles: dict[int, PlayerRole] = {}
    for p in range(4):
        try:
            rust_role = gs.get_role(p) if hasattr(gs, 'get_role') else 2
            roles[p] = _RUST_ROLE_MAP.get(rust_role, PlayerRole.OPPONENT)
        except Exception:
            roles[p] = PlayerRole.OPPONENT
    state.roles = roles

    return state


# Backward-compatibility alias — all code should use RustGameLoop directly
GameLoop = RustGameLoop
