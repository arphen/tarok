"""Batched self-play game runner — runs M games concurrently with batched NN inference.

Instead of playing games one-at-a-time (each requiring ~60-80 individual
forward passes), this module manages M in-flight games simultaneously.
All pending NN decisions are collected and processed in a single batched
forward pass, dramatically improving throughput on multi-core CPUs and GPUs.

Each game is modelled as a state machine that advances until it needs an
NN decision, yields its tensors, receives the action, and continues.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field

import torch

try:
    import tarok_engine as te
except ImportError:
    te = None  # type: ignore[assignment]

from tarok.adapters.ai.encoding import (
    DecisionType,
    CARD_ACTION_SIZE,
    ORACLE_STATE_SIZE,
    BID_ACTIONS,
    KING_ACTIONS,
    encode_bid_mask,
    encode_king_mask,
    encode_talon_mask,
)
from tarok.adapters.ai.agent import Experience
from tarok.adapters.ai.compute import ComputeBackend, create_backend
from tarok.adapters.ai.network import TarokNet
from tarok.entities.card import Card, CardType, DECK
from tarok.entities.game_state import Contract, Phase, GameState


# Reuse rust_game_loop helpers
from tarok.adapters.ai.rust_game_loop import (
    _RUST_U8_TO_PY_CONTRACT,
    _PY_CONTRACT_TO_RUST_U8,
    _BID_IDX_TO_RUST,
    _is_klop,
    _is_solo,
    _is_berac,
    _talon_cards,
    _build_py_state_stub,
)


class GamePhase(enum.Enum):
    """State machine phases for a single in-flight game."""
    DEAL = "deal"
    BID = "bid"
    KING_CALL = "king_call"
    TALON_PICK = "talon_pick"
    TRICK_PLAY = "trick_play"
    DONE = "done"


@dataclass
class PendingDecision:
    """A single NN decision request from one in-flight game."""
    game_idx: int
    player: int
    state_tensor: torch.Tensor
    legal_mask: torch.Tensor
    decision_type: DecisionType
    oracle_tensor: torch.Tensor | None = None


@dataclass
class GameExperience:
    """Experience entry collected during batched play (before reward assignment)."""
    state: torch.Tensor
    action: int
    log_prob: torch.Tensor
    value: torch.Tensor
    decision_type: DecisionType
    oracle_state: torch.Tensor | None
    legal_mask: torch.Tensor
    game_id: int
    step_in_game: int


@dataclass
class InFlightGame:
    """State for a single in-flight game managed by the batch runner."""
    gs: object  # te.RustGameState
    dealer: int
    game_id: int  # global game counter for experience tracking
    phase: GamePhase = GamePhase.DEAL
    # Bidding state
    passed: list[bool] = field(default_factory=lambda: [False] * 4)
    highest_bid: int | None = None
    winning_bidder: int | None = None
    current_bidder: int = 0
    bid_round: int = 0
    # Trick play state
    trick_num: int = 0
    trick_offset: int = 0  # 0..3 within a trick
    lead_player: int = 0
    # Metadata
    contract_u8: int | None = None
    declarer: int | None = None
    py_contract: Contract | None = None
    initial_tarok_counts: dict = field(default_factory=dict)
    # Experiences collected during this game
    experiences: list[GameExperience] = field(default_factory=list)
    step_counter: int = 0
    # Whether agents should record experiences (False for external opponents)
    record_all_players: bool = True
    # Epsilon-greedy: pre-decided random actions for pending decisions
    explore_action: int | None = None


@dataclass
class GameResult:
    """Result of a completed game."""
    game_id: int
    scores: dict[int, int]
    py_state: GameState
    experiences: list[GameExperience]
    contract_name: str
    is_klop: bool
    is_solo: bool
    declarer_p0: bool
    initial_tarok_counts: dict


class BatchGameRunner:
    """Runs M games concurrently with batched NN inference.

    Usage::

        runner = BatchGameRunner(network, concurrency=32, oracle=False)
        results = runner.run(
            total_games=20,
            explore_rate=0.1,
            dealer_offset=0,
        )
    """

    def __init__(
        self,
        network: TarokNet,
        concurrency: int = 32,
        oracle: bool = False,
        device: str = "cpu",
        *,
        compute: ComputeBackend | None = None,
    ):
        assert te is not None, "tarok_engine Rust extension not installed"
        self.network = network
        self.concurrency = concurrency
        self.oracle = oracle
        # Prefer explicit backend; fall back to factory for backward compat
        self.compute = compute if compute is not None else create_backend(device)
        self.device = self.compute.device
        self._rng = random.Random()

    def run(
        self,
        total_games: int,
        explore_rate: float = 0.1,
        dealer_offset: int = 0,
        game_id_offset: int = 0,
    ) -> list[GameResult]:
        """Play *total_games* games with batched NN inference. Returns results."""
        results: list[GameResult] = []
        games: list[InFlightGame | None] = [None] * self.concurrency
        game_id_to_slot: dict[int, int] = {}  # game_id → slot index for O(1) lookup
        next_game_idx = 0  # next game to start (0..total_games-1)
        active_count = 0

        def _start_game(slot: int) -> InFlightGame:
            nonlocal next_game_idx
            dealer = (dealer_offset + next_game_idx) % 4
            gs = te.RustGameState(dealer)
            gs.deal()
            game = InFlightGame(
                gs=gs,
                dealer=dealer,
                game_id=game_id_offset + next_game_idx,
                current_bidder=(dealer + 1) % 4,
                lead_player=(dealer + 1) % 4,
            )
            # Store initial tarok counts
            for p in range(4):
                hand = gs.hand(p)
                game.initial_tarok_counts[p] = sum(1 for c in hand if c < 22)
            game.phase = GamePhase.BID
            game_id_to_slot[game.game_id] = slot
            next_game_idx += 1
            return game

        # Seed initial games
        n_initial = min(self.concurrency, total_games)
        for i in range(n_initial):
            games[i] = _start_game(i)
            active_count += 1

        # Main loop: advance games until all are done
        while active_count > 0:
            # Step 1: Advance each game until it needs an NN decision or finishes
            pending: list[PendingDecision] = []
            for slot_idx, game in enumerate(games):
                if game is None or game.phase == GamePhase.DONE:
                    continue
                decision = self._advance_until_decision(game)
                if decision is not None:
                    pending.append(decision)
                elif game.phase == GamePhase.DONE:
                    # Game finished — collect result
                    result = self._finalize_game(game)
                    results.append(result)
                    game_id_to_slot.pop(game.game_id, None)
                    # Replace with new game if quota remains
                    if next_game_idx < total_games:
                        games[slot_idx] = _start_game(slot_idx)
                    else:
                        games[slot_idx] = None
                        active_count -= 1

            if not pending:
                continue

            # Step 2: Handle epsilon-greedy exploration
            # For explore decisions, we still need the value estimate from the network
            explore_decisions: list[int] = []  # indices into pending
            for i, p in enumerate(pending):
                if self._rng.random() < explore_rate:
                    # Pick random legal action
                    legal_indices = p.legal_mask.nonzero(as_tuple=True)[0].tolist()
                    games_for_slot = self._game_for_pending(games, p, game_id_to_slot)
                    if games_for_slot is not None:
                        games_for_slot.explore_action = self._rng.choice(legal_indices)
                    explore_decisions.append(i)

            # Step 3: Batch forward pass (delegated to compute backend)
            states = torch.stack([p.state_tensor for p in pending])
            masks_list = [p.legal_mask for p in pending]
            dtypes = [p.decision_type for p in pending]
            oracle_states = None
            if self.oracle and any(p.oracle_tensor is not None for p in pending):
                oracle_states = torch.stack([
                    p.oracle_tensor if p.oracle_tensor is not None
                    else torch.zeros(ORACLE_STATE_SIZE)
                    for p in pending
                ])

            all_logits, all_values = self.compute.forward_batch(
                self.network, states, dtypes, oracle_states,
            )

            # Step 4: Sample actions and record experiences
            for i, p in enumerate(pending):
                game = self._game_for_pending(games, p, game_id_to_slot)
                if game is None:
                    continue

                logits_i = all_logits[i]
                value_i = all_values[i]
                mask_i = masks_list[i]

                # Pad mask to match logits size (logits are padded to CARD_ACTION_SIZE)
                if mask_i.shape[0] < CARD_ACTION_SIZE:
                    padded = torch.zeros(CARD_ACTION_SIZE)
                    padded[:mask_i.shape[0]] = mask_i
                    mask_i = padded

                # Check if this was an explore decision
                if game.explore_action is not None:
                    action_idx = game.explore_action
                    log_prob = torch.tensor(0.0)
                    game.explore_action = None
                else:
                    # Mask illegal actions
                    masked_logits = logits_i.clone()
                    masked_logits[mask_i == 0] = float("-inf")
                    probs = torch.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    action_idx = action.item()
                    log_prob = dist.log_prob(action)

                # Record experience
                should_record = game.record_all_players or p.player == 0
                if should_record:
                    game.experiences.append(GameExperience(
                        state=p.state_tensor,
                        action=action_idx,
                        log_prob=log_prob.detach().cpu(),
                        value=value_i.detach().cpu(),
                        decision_type=p.decision_type,
                        oracle_state=p.oracle_tensor,
                        legal_mask=p.legal_mask.detach().cpu(),
                        game_id=game.game_id,
                        step_in_game=game.step_counter,
                    ))
                    game.step_counter += 1

                # Apply action to game state
                self._apply_action(game, p, action_idx)

            # Step 5: Check for newly finished games after applying actions
            for slot_idx, game in enumerate(games):
                if game is None or game.phase != GamePhase.DONE:
                    continue
                result = self._finalize_game(game)
                results.append(result)
                game_id_to_slot.pop(game.game_id, None)
                if next_game_idx < total_games:
                    games[slot_idx] = _start_game(slot_idx)
                else:
                    games[slot_idx] = None
                    active_count -= 1

        return results

    # ------------------------------------------------------------------
    # Game advancement — returns PendingDecision or None (game finished/advanced)
    # ------------------------------------------------------------------

    def _advance_until_decision(self, game: InFlightGame) -> PendingDecision | None:
        """Advance the game state machine until it needs an NN decision.

        Returns a PendingDecision if the game needs input, or None if the
        game completed or transitioned to a new phase autonomously.
        """
        gs = game.gs

        if game.phase == GamePhase.BID:
            return self._bid_step(game)

        if game.phase == GamePhase.KING_CALL:
            return self._king_call_step(game)

        if game.phase == GamePhase.TALON_PICK:
            return self._talon_pick_step(game)

        if game.phase == GamePhase.TRICK_PLAY:
            return self._trick_step(game)

        return None

    # ------------------------------------------------------------------
    # Phase: Bidding
    # ------------------------------------------------------------------

    def _bid_step(self, game: InFlightGame) -> PendingDecision | None:
        gs = game.gs

        # Check if bidding is over
        active = [i for i in range(4) if not game.passed[i]]
        if (len(active) <= 1 and game.winning_bidder is not None) or len(active) == 0:
            self._resolve_bidding(game)
            return self._advance_until_decision(game)

        if game.bid_round >= 20:
            self._resolve_bidding(game)
            return self._advance_until_decision(game)

        bidder = game.current_bidder
        if game.passed[bidder]:
            # Skip passed players
            self._next_bidder(game)
            return self._advance_until_decision(game)

        # Encode state + mask
        state_t = torch.from_numpy(
            gs.encode_state(bidder, te.DT_BID)
        ).float()

        rust_legal = gs.legal_bids(bidder)
        py_legal_bids = [None]
        for lb in rust_legal:
            if lb is not None:
                py_c = _RUST_U8_TO_PY_CONTRACT.get(lb)
                if py_c is not None:
                    py_legal_bids.append(py_c)
        mask = encode_bid_mask(py_legal_bids)

        oracle_t = None
        if self.oracle:
            oracle_t = torch.from_numpy(
                gs.encode_oracle_state(bidder, te.DT_BID)
            ).float()

        return PendingDecision(
            game_idx=game.game_id,
            player=bidder,
            state_tensor=state_t,
            legal_mask=mask,
            decision_type=DecisionType.BID,
            oracle_tensor=oracle_t,
        )

    def _apply_bid(self, game: InFlightGame, action_idx: int) -> None:
        gs = game.gs
        bidder = game.current_bidder
        bid_contract = BID_ACTIONS[action_idx]

        if bid_contract is None:
            game.passed[bidder] = True
            gs.add_bid(bidder, None)
        else:
            rust_u8 = _PY_CONTRACT_TO_RUST_U8.get(bid_contract)
            rust_legal = gs.legal_bids(bidder)
            if rust_u8 not in rust_legal:
                game.passed[bidder] = True
                gs.add_bid(bidder, None)
            else:
                gs.add_bid(bidder, rust_u8)
                game.highest_bid = rust_u8
                game.winning_bidder = bidder

        game.bid_round += 1
        self._next_bidder(game)

    def _next_bidder(self, game: InFlightGame) -> None:
        for _ in range(4):
            game.current_bidder = (game.current_bidder + 1) % 4
            if not game.passed[game.current_bidder]:
                break

    def _resolve_bidding(self, game: InFlightGame) -> None:
        gs = game.gs

        if game.winning_bidder is not None and game.highest_bid is not None:
            gs.declarer = game.winning_bidder
            gs.contract = game.highest_bid
            gs.set_role(game.winning_bidder, 0)  # Declarer
            for i in range(4):
                if i != game.winning_bidder:
                    gs.set_role(i, 2)  # Opponent

            game.contract_u8 = game.highest_bid
            game.declarer = game.winning_bidder
            game.py_contract = _RUST_U8_TO_PY_CONTRACT.get(game.highest_bid)

            if _is_berac(game.highest_bid):
                gs.phase = te.PHASE_TRICK_PLAY
                game.phase = GamePhase.TRICK_PLAY
                game.lead_player = (game.dealer + 1) % 4
            elif _is_klop(game.highest_bid):
                gs.phase = te.PHASE_TRICK_PLAY
                game.phase = GamePhase.TRICK_PLAY
                game.lead_player = (game.dealer + 1) % 4
            elif _is_solo(game.highest_bid):
                # Solo: skip king calling, go to talon
                gs.phase = te.PHASE_TALON_EXCHANGE
                talon_cards = _talon_cards(game.highest_bid)
                if talon_cards > 0:
                    game.phase = GamePhase.TALON_PICK
                else:
                    gs.phase = te.PHASE_TRICK_PLAY
                    game.phase = GamePhase.TRICK_PLAY
                    game.lead_player = (game.dealer + 1) % 4
            else:
                gs.phase = te.PHASE_KING_CALLING
                game.phase = GamePhase.KING_CALL
        else:
            # All passed → Klop
            gs.contract = 0
            for i in range(4):
                gs.set_role(i, 2)
            gs.phase = te.PHASE_TRICK_PLAY
            game.contract_u8 = 0
            game.declarer = None
            game.py_contract = Contract.KLOP
            game.phase = GamePhase.TRICK_PLAY
            game.lead_player = (game.dealer + 1) % 4

    # ------------------------------------------------------------------
    # Phase: King Call
    # ------------------------------------------------------------------

    def _king_call_step(self, game: InFlightGame) -> PendingDecision | None:
        gs = game.gs
        declarer = game.declarer
        callable_idxs = gs.callable_kings()
        if not callable_idxs:
            # No callable kings — skip to talon
            self._transition_after_king_call(game)
            return self._advance_until_decision(game)

        py_callable = [DECK[idx] for idx in callable_idxs]
        state_t = torch.from_numpy(
            gs.encode_state(declarer, te.DT_KING_CALL)
        ).float()
        mask = encode_king_mask(py_callable)

        oracle_t = None
        if self.oracle:
            oracle_t = torch.from_numpy(
                gs.encode_oracle_state(declarer, te.DT_KING_CALL)
            ).float()

        return PendingDecision(
            game_idx=game.game_id,
            player=declarer,
            state_tensor=state_t,
            legal_mask=mask,
            decision_type=DecisionType.KING_CALL,
            oracle_tensor=oracle_t,
        )

    def _apply_king_call(self, game: InFlightGame, action_idx: int) -> None:
        gs = game.gs
        declarer = game.declarer
        callable_idxs = gs.callable_kings()

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

        # Identify partner
        for p in range(4):
            if p != declarer:
                hand = gs.hand(p)
                if chosen_card_idx in hand:
                    gs.partner = p
                    gs.set_role(p, 1)  # Partner
                    break

        self._transition_after_king_call(game)

    def _transition_after_king_call(self, game: InFlightGame) -> None:
        gs = game.gs
        talon_cards = _talon_cards(game.contract_u8)
        if talon_cards > 0:
            gs.phase = te.PHASE_TALON_EXCHANGE
            game.phase = GamePhase.TALON_PICK
        else:
            gs.phase = te.PHASE_TRICK_PLAY
            game.phase = GamePhase.TRICK_PLAY
            game.lead_player = (game.dealer + 1) % 4

    # ------------------------------------------------------------------
    # Phase: Talon Exchange
    # ------------------------------------------------------------------

    def _talon_pick_step(self, game: InFlightGame) -> PendingDecision | None:
        gs = game.gs
        declarer = game.declarer
        talon_cards = _talon_cards(game.contract_u8)

        talon_idxs = gs.talon()
        group_size = 6 // (6 // talon_cards) if talon_cards in (1, 2, 3) else talon_cards
        groups: list[list[int]] = []
        for i in range(0, len(talon_idxs), group_size):
            groups.append(talon_idxs[i : i + group_size])
        num_groups = len(groups)

        gs.set_talon_revealed(groups)

        state_t = torch.from_numpy(
            gs.encode_state(declarer, te.DT_TALON_PICK)
        ).float()
        mask = encode_talon_mask(num_groups)

        oracle_t = None
        if self.oracle:
            oracle_t = torch.from_numpy(
                gs.encode_oracle_state(declarer, te.DT_TALON_PICK)
            ).float()

        # Store groups on the game for use in _apply_talon_pick
        game._talon_groups = groups  # type: ignore[attr-defined]
        game._talon_cards = talon_cards  # type: ignore[attr-defined]

        return PendingDecision(
            game_idx=game.game_id,
            player=declarer,
            state_tensor=state_t,
            legal_mask=mask,
            decision_type=DecisionType.TALON_PICK,
            oracle_tensor=oracle_t,
        )

    def _apply_talon_pick(self, game: InFlightGame, action_idx: int) -> None:
        gs = game.gs
        declarer = game.declarer
        groups = game._talon_groups  # type: ignore[attr-defined]
        talon_cards = game._talon_cards  # type: ignore[attr-defined]
        num_groups = len(groups)

        if action_idx >= num_groups:
            action_idx = 0

        picked = groups[action_idx]
        for card_idx in picked:
            gs.add_to_hand(declarer, card_idx)
            gs.remove_from_talon(card_idx)

        # Heuristic discard
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
        game.phase = GamePhase.TRICK_PLAY
        game.lead_player = (game.dealer + 1) % 4

    # ------------------------------------------------------------------
    # Phase: Trick Play
    # ------------------------------------------------------------------

    def _trick_step(self, game: InFlightGame) -> PendingDecision | None:
        gs = game.gs

        if game.trick_num >= 12:
            game.phase = GamePhase.DONE
            return None

        # Start new trick if at offset 0
        if game.trick_offset == 0:
            gs.start_trick(game.lead_player)

        player = (game.lead_player + game.trick_offset) % 4
        gs.current_player = player

        state_t = torch.from_numpy(
            gs.encode_state(player, te.DT_CARD_PLAY)
        ).float()
        legal_mask = torch.from_numpy(
            gs.legal_plays_mask(player)
        ).float()

        oracle_t = None
        if self.oracle:
            oracle_t = torch.from_numpy(
                gs.encode_oracle_state(player, te.DT_CARD_PLAY)
            ).float()

        return PendingDecision(
            game_idx=game.game_id,
            player=player,
            state_tensor=state_t,
            legal_mask=legal_mask,
            decision_type=DecisionType.CARD_PLAY,
            oracle_tensor=oracle_t,
        )

    def _apply_trick_card(self, game: InFlightGame, p: PendingDecision, action_idx: int) -> None:
        gs = game.gs
        player = p.player

        # Validate
        legal_cards = gs.legal_plays(player)
        if action_idx not in legal_cards:
            action_idx = legal_cards[0]

        gs.play_card(player, action_idx)
        game.trick_offset += 1

        if game.trick_offset >= 4:
            # Trick complete
            winner, points = gs.finish_trick()
            game.lead_player = winner
            game.trick_num += 1
            game.trick_offset = 0

            if game.trick_num >= 12:
                game.phase = GamePhase.DONE

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, game: InFlightGame, pending: PendingDecision, action_idx: int) -> None:
        """Apply the NN's chosen action to the game state."""
        if pending.decision_type == DecisionType.BID:
            self._apply_bid(game, action_idx)
        elif pending.decision_type == DecisionType.KING_CALL:
            self._apply_king_call(game, action_idx)
        elif pending.decision_type == DecisionType.TALON_PICK:
            self._apply_talon_pick(game, action_idx)
        elif pending.decision_type == DecisionType.CARD_PLAY:
            self._apply_trick_card(game, pending, action_idx)

    # ------------------------------------------------------------------
    # Game finalization
    # ------------------------------------------------------------------

    def _finalize_game(self, game: InFlightGame) -> GameResult:
        gs = game.gs
        gs.phase = te.PHASE_SCORING
        scores_arr = gs.score_game()
        scores = {i: int(scores_arr[i]) for i in range(4)}

        py_state = _build_py_state_stub(
            game.dealer, game.py_contract, game.declarer,
            gs, game.initial_tarok_counts,
        )

        contract_name = game.py_contract.name.lower() if game.py_contract else "klop"

        return GameResult(
            game_id=game.game_id,
            scores=scores,
            py_state=py_state,
            experiences=game.experiences,
            contract_name=contract_name,
            is_klop=_is_klop(game.contract_u8) if game.contract_u8 is not None else True,
            is_solo=_is_solo(game.contract_u8) if game.contract_u8 is not None else False,
            declarer_p0=(game.declarer == 0),
            initial_tarok_counts=game.initial_tarok_counts,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _game_for_pending(
        self, games: list[InFlightGame | None], p: PendingDecision,
        game_id_to_slot: dict[int, int],
    ) -> InFlightGame | None:
        """Find the InFlightGame that owns this pending decision."""
        slot = game_id_to_slot.get(p.game_idx)
        if slot is not None:
            return games[slot]
        return None
