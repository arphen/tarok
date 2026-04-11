"""Fast game loop using the Rust engine — replaces the pure-Python GameLoop
for training, spectating, and human play.

All game mechanics (dealing, legal moves, trick evaluation, scoring, state
encoding) happen in Rust.  Only neural-network inference stays in Python.
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
)
from tarok.ports.observer_port import GameObserverPort
from tarok.use_cases.game_loop import GameLoop


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

# BID_ACTIONS index → Rust contract u8 or None (pass)
_BID_IDX_TO_RUST: list[int | None] = [None]  # index 0 = pass
for c in [Contract.THREE, Contract.TWO, Contract.ONE,
          Contract.SOLO_THREE, Contract.SOLO_TWO, Contract.SOLO_ONE,
          Contract.SOLO, Contract.BERAC]:
    _BID_IDX_TO_RUST.append(_PY_CONTRACT_TO_RUST_U8[c])


class _NullObserver:
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
    ):
        assert te is not None, "tarok_engine Rust extension not installed"
        assert len(players) == 4
        self._players = players
        self._observer: GameObserverPort = observer or _NullObserver()  # type: ignore
        self._rng = rng or random.Random()

    async def run(self, dealer: int = 0) -> tuple[GameState, dict[int, int]]:
        """Play one full game, returning (final_state, {player: score})."""
        # Universal PlayerPort compatibility:
        # If any player lacks the tensor fast-path API, delegate to the
        # canonical Python GameLoop which uses choose_* methods for all players.
        if not all(hasattr(player, '_decide_from_tensors') for player in self._players):
            loop = GameLoop(self._players, observer=self._observer, rng=self._rng)
            return await loop.run(dealer=dealer)

        gs = te.RustGameState(dealer)
        gs.deal()

        # Build a thin Python GameState for observer callbacks (optional)
        # Only populated when we actually have observer callbacks
        has_observer = not isinstance(self._observer, _NullObserver)

        await self._observer.on_game_start(None)
        await self._observer.on_deal(None)

        # === BIDDING ===
        contract, declarer = await self._run_bidding(gs)

        if contract is None:
            # Re-deal on all pass (shouldn't happen with klop)
            return await self.run(dealer=(dealer + 1) % 4)

        py_contract = _RUST_U8_TO_PY_CONTRACT.get(contract)

        # Store initial tarok counts for metrics
        initial_tarok_counts = {}
        for p in range(4):
            hand = gs.hand(p)
            initial_tarok_counts[p] = sum(1 for c in hand if c < 22)

        # === KING CALLING ===
        if declarer is not None and not _is_klop(contract) and not _is_solo(contract) and not _is_berac(contract):
            await self._run_king_call(gs, declarer)

        # === TALON EXCHANGE ===
        talon_cards = _talon_cards(contract)
        if declarer is not None and talon_cards > 0 and not _is_klop(contract) and not _is_berac(contract):
            await self._run_talon_exchange(gs, declarer, talon_cards)

        # === ANNOUNCEMENTS ===
        if declarer is not None and not _is_klop(contract) and not _is_berac(contract):
            gs.phase = te.PHASE_ANNOUNCEMENTS
            # Simplified: skip announcements in Rust loop for now
            # (random play / early training rarely uses them)
            gs.phase = te.PHASE_TRICK_PLAY

        # === TRICK PLAY ===
        gs.phase = te.PHASE_TRICK_PLAY
        lead_player = (dealer + 1) % 4
        for trick_num in range(12):
            gs.start_trick(lead_player)

            for offset in range(4):
                player = (lead_player + offset) % 4
                gs.current_player = player

                # Get legal moves and state encoding from Rust
                state_t = torch.from_numpy(
                    gs.encode_state(player, te.DT_CARD_PLAY)
                ).float()
                legal_mask = torch.from_numpy(
                    gs.legal_plays_mask(player)
                ).float()

                oracle_t = None
                agent = self._players[player]
                if (
                    hasattr(agent, 'network')
                    and agent._training
                    and agent.network.oracle_critic_enabled
                ):
                    oracle_t = torch.from_numpy(
                        gs.encode_oracle_state(player, te.DT_CARD_PLAY)
                    ).float()

                # Agent picks action using pre-encoded tensors
                action_idx = agent._decide_from_tensors(
                    state_t, legal_mask, DecisionType.CARD_PLAY, oracle_t,
                )

                # Validate and play
                legal_cards = gs.legal_plays(player)
                if action_idx not in legal_cards:
                    action_idx = legal_cards[0]

                gs.play_card(player, action_idx)

            # Finish trick
            winner, points = gs.finish_trick()
            lead_player = winner

        # === SCORING ===
        gs.phase = te.PHASE_SCORING
        scores_arr = gs.score_game()
        scores = {i: int(scores_arr[i]) for i in range(4)}

        # Build a minimal Python GameState for compatibility with trainer
        py_state = _build_py_state_stub(
            dealer, py_contract, declarer,
            gs, initial_tarok_counts,
        )

        await self._observer.on_game_end(scores, py_state)

        return py_state, scores

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    async def _run_bidding(self, gs) -> tuple[int | None, int | None]:
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

            # Encode state + mask for bidding
            state_t = torch.from_numpy(
                gs.encode_state(bidder, te.DT_BID)
            ).float()

            # Build legal bid mask
            rust_legal = gs.legal_bids(bidder)  # list of Option<u8>
            py_legal_bids = [None]  # can always pass
            for lb in rust_legal:
                if lb is not None:
                    py_c = _RUST_U8_TO_PY_CONTRACT.get(lb)
                    if py_c is not None:
                        py_legal_bids.append(py_c)
            mask = encode_bid_mask(py_legal_bids)

            oracle_t = None
            agent = self._players[bidder]
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

            # Map action to bid
            bid_contract = BID_ACTIONS[action_idx]  # Contract | None
            if bid_contract is None:
                # Pass
                passed[bidder] = True
                gs.add_bid(bidder, None)
            else:
                rust_u8 = _PY_CONTRACT_TO_RUST_U8.get(bid_contract)
                # Validate it's actually legal
                if rust_u8 not in rust_legal:
                    passed[bidder] = True
                    gs.add_bid(bidder, None)
                else:
                    gs.add_bid(bidder, rust_u8)
                    highest = rust_u8
                    winning_player = bidder

            # Next bidder
            for _ in range(4):
                bidder = (bidder + 1) % 4
                if not passed[bidder]:
                    break

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

    async def _run_king_call(self, gs, declarer: int):
        """Declarer calls a king."""
        callable_idxs = gs.callable_kings()  # list of u8 card indices
        if not callable_idxs:
            return

        # Map Rust card indices to Python Card objects for mask encoding
        py_callable = [DECK[idx] for idx in callable_idxs]

        state_t = torch.from_numpy(
            gs.encode_state(declarer, te.DT_KING_CALL)
        ).float()
        mask = encode_king_mask(py_callable)

        oracle_t = None
        agent = self._players[declarer]
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

        # Map suit index to card index
        chosen_suit = KING_ACTIONS[action_idx]
        chosen_card_idx = None
        for idx in callable_idxs:
            card = DECK[idx]
            if card.suit == chosen_suit:
                chosen_card_idx = idx
                break
        if chosen_card_idx is None:
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

    async def _run_talon_exchange(self, gs, declarer: int, talon_cards: int):
        """Reveal talon, let declarer pick a group and discard."""
        # Get talon contents from Rust and build groups
        talon_idxs = gs.talon()  # list of u8 card indices
        group_size = 6 // (6 // talon_cards) if talon_cards in (1, 2, 3) else talon_cards
        groups: list[list[int]] = []
        for i in range(0, len(talon_idxs), group_size):
            groups.append(talon_idxs[i : i + group_size])
        num_groups = len(groups)

        # Tell Rust about revealed talon
        gs.set_talon_revealed(groups)

        # Agent picks a group
        state_t = torch.from_numpy(
            gs.encode_state(declarer, te.DT_TALON_PICK)
        ).float()
        mask = encode_talon_mask(num_groups)

        oracle_t = None
        agent = self._players[declarer]
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

        # Heuristic discard (same as Python agent: cheapest non-king non-tarok)
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
        for idx, _ in discardable[:talon_cards]:
            gs.remove_card(declarer, idx)
            gs.add_put_down(idx)

        gs.phase = te.PHASE_TRICK_PLAY
        gs.current_player = (gs.dealer + 1) % 4


def _is_klop(contract_u8: int) -> bool:
    return contract_u8 == 0

def _is_solo(contract_u8: int) -> bool:
    return contract_u8 in (4, 5, 6, 7)

def _is_berac(contract_u8: int) -> bool:
    return contract_u8 == 8

def _talon_cards(contract_u8: int) -> int:
    return {1: 3, 2: 2, 3: 1, 4: 3, 5: 2, 6: 1}.get(contract_u8, 0)


def _build_py_state_stub(
    dealer: int,
    contract: Contract | None,
    declarer: int | None,
    gs,
    initial_tarok_counts: dict,
) -> GameState:
    """Build a minimal Python GameState with fields the trainer needs."""
    state = GameState.__new__(GameState)
    state.dealer = dealer
    state.contract = contract
    state.declarer = declarer
    state.partner = gs.partner
    state.phase = Phase.FINISHED
    state.num_players = 4
    state.hands = [[] for _ in range(4)]
    state.tricks = []
    state.current_trick = None
    state.bids = []
    state.announcements = {}
    state.kontra_levels = {}
    state.talon = []
    state.talon_revealed = []
    state.put_down = []
    state.called_king = None
    state.roles = [None] * 4
    state.initial_tarok_counts = initial_tarok_counts

    # Reconstruct bids from Rust state (minimal — just for metrics)
    # The trainer only checks: state.bids for player 0 bids, state.contract, state.declarer
    class _FakeBid:
        def __init__(self, player, contract):
            self.player = player
            self.contract = contract

    # We don't have full bid history from Rust, but the trainer only needs:
    # - whether player 0 placed a real bid (agent0_bids)
    # - the final contract and declarer
    # So we reconstruct a minimal bid list
    if declarer is not None and contract is not None:
        state.bids = [_FakeBid(declarer, contract)]
        if declarer == 0:
            state.bids.append(_FakeBid(0, contract))
    
    return state
