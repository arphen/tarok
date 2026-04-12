"""Batched self-play game runner — runs M games concurrently with batched NN inference.

All game mechanics happen in Rust (tarok_engine). Only NN inference stays in Python.
Each game is a state machine that advances until it needs an NN decision, yields
tensors, receives the action, and continues.

This is the ONLY game simulator for training — no Python fallback.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field

import torch

import tarok_engine as te

from training_lab.entities.encoding import (
    DecisionType,
    CARD_ACTION_SIZE,
    BID_ACTION_SIZE,
    KING_ACTION_SIZE,
    TALON_ACTION_SIZE,
    ORACLE_STATE_SIZE,
    BID_IDX_TO_RUST,
    KING_ACTIONS,
    RUST_U8_TO_CONTRACT_NAME,
    is_klop,
    is_solo,
    is_berac,
    is_king,
    is_tarok,
    card_idx_to_suit,
    talon_cards_for_contract,
)
from training_lab.entities.experience import Experience
from training_lab.entities.network import TarokNet
from training_lab.ports.compute_backend import ComputeBackendPort
from training_lab.ports.game_simulator import GameResult, GameSimulatorPort


class GamePhase(enum.Enum):
    DEAL = "deal"
    BID = "bid"
    KING_CALL = "king_call"
    TALON_PICK = "talon_pick"
    TRICK_PLAY = "trick_play"
    DONE = "done"


@dataclass
class PendingDecision:
    game_idx: int
    player: int
    state_tensor: torch.Tensor
    legal_mask: torch.Tensor
    decision_type: DecisionType
    oracle_tensor: torch.Tensor | None = None


@dataclass
class InFlightGame:
    gs: object  # te.RustGameState
    dealer: int
    game_id: int
    phase: GamePhase = GamePhase.DEAL
    passed: list[bool] = field(default_factory=lambda: [False] * 4)
    highest_bid: int | None = None
    winning_bidder: int | None = None
    current_bidder: int = 0
    bid_round: int = 0
    trick_num: int = 0
    trick_offset: int = 0
    lead_player: int = 0
    contract_u8: int | None = None
    declarer: int | None = None
    experiences: list[dict] = field(default_factory=list)
    step_counter: int = 0
    explore_action: int | None = None


class RustBatchGameRunner(GameSimulatorPort):
    """Runs M games concurrently with batched NN inference via tarok_engine.

    Usage::

        runner = RustBatchGameRunner(compute, concurrency=128)
        results = runner.play_batch(network, n_games=20, explore_rate=0.1)
    """

    def __init__(
        self,
        compute: ComputeBackendPort,
        concurrency: int = 128,
        oracle: bool = False,
    ):
        self.compute = compute
        self.concurrency = concurrency
        self.oracle = oracle
        self._rng = random.Random()

    def play_batch(
        self,
        network: TarokNet,
        n_games: int,
        explore_rate: float = 0.1,
    ) -> list[GameResult]:
        results: list[GameResult] = []
        games: list[InFlightGame | None] = [None] * self.concurrency
        game_id_to_slot: dict[int, int] = {}
        next_game_idx = 0
        active_count = 0

        def _start_game(slot: int) -> InFlightGame:
            nonlocal next_game_idx
            dealer = next_game_idx % 4
            gs = te.RustGameState(dealer)
            gs.deal()
            game = InFlightGame(
                gs=gs,
                dealer=dealer,
                game_id=next_game_idx,
                current_bidder=(dealer + 1) % 4,
                lead_player=(dealer + 1) % 4,
            )
            game.phase = GamePhase.BID
            game_id_to_slot[game.game_id] = slot
            next_game_idx += 1
            return game

        n_initial = min(self.concurrency, n_games)
        for i in range(n_initial):
            games[i] = _start_game(i)
            active_count += 1

        while active_count > 0:
            pending: list[PendingDecision] = []
            for slot_idx, game in enumerate(games):
                if game is None or game.phase == GamePhase.DONE:
                    continue
                decision = self._advance_until_decision(game)
                if decision is not None:
                    pending.append(decision)
                elif game.phase == GamePhase.DONE:
                    result = self._finalize_game(game)
                    results.append(result)
                    game_id_to_slot.pop(game.game_id, None)
                    if next_game_idx < n_games:
                        games[slot_idx] = _start_game(slot_idx)
                    else:
                        games[slot_idx] = None
                        active_count -= 1

            if not pending:
                continue

            # Epsilon-greedy exploration
            for p in pending:
                if self._rng.random() < explore_rate:
                    legal_indices = p.legal_mask.nonzero(as_tuple=True)[0].tolist()
                    game = self._game_for_pending(games, p, game_id_to_slot)
                    if game is not None:
                        game.explore_action = self._rng.choice(legal_indices)

            # Batch forward pass
            states = torch.stack([p.state_tensor for p in pending])
            dtypes = [p.decision_type for p in pending]
            oracle_states = None
            if self.oracle and any(p.oracle_tensor is not None for p in pending):
                oracle_states = torch.stack([
                    p.oracle_tensor if p.oracle_tensor is not None
                    else torch.zeros(ORACLE_STATE_SIZE)
                    for p in pending
                ])

            all_logits, all_values = self.compute.forward_batch(
                network, states, dtypes, oracle_states,
            )

            # Sample actions and record experiences
            for i, p in enumerate(pending):
                game = self._game_for_pending(games, p, game_id_to_slot)
                if game is None:
                    continue

                logits_i = all_logits[i]
                value_i = all_values[i]
                mask_i = p.legal_mask

                if mask_i.shape[0] < CARD_ACTION_SIZE:
                    padded = torch.zeros(CARD_ACTION_SIZE)
                    padded[:mask_i.shape[0]] = mask_i
                    mask_i = padded

                if game.explore_action is not None:
                    action_idx = game.explore_action
                    log_prob = torch.tensor(0.0)
                    game.explore_action = None
                else:
                    masked_logits = logits_i.clone()
                    masked_logits[mask_i == 0] = float("-inf")
                    probs = torch.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    action_idx = action.item()
                    log_prob = dist.log_prob(action)

                game.experiences.append({
                    "state": p.state_tensor,
                    "action": action_idx,
                    "log_prob": log_prob.detach().cpu(),
                    "value": value_i.detach().cpu(),
                    "decision_type": p.decision_type,
                    "oracle_state": p.oracle_tensor,
                    "legal_mask": p.legal_mask.detach().cpu(),
                    "game_id": game.game_id,
                    "step_in_game": game.step_counter,
                })
                game.step_counter += 1

                self._apply_action(game, p, action_idx)

            # Check for newly finished games
            for slot_idx, game in enumerate(games):
                if game is None or game.phase != GamePhase.DONE:
                    continue
                result = self._finalize_game(game)
                results.append(result)
                game_id_to_slot.pop(game.game_id, None)
                if next_game_idx < n_games:
                    games[slot_idx] = _start_game(slot_idx)
                else:
                    games[slot_idx] = None
                    active_count -= 1

        return results

    # ------------------------------------------------------------------
    # Game advancement
    # ------------------------------------------------------------------

    def _advance_until_decision(self, game: InFlightGame) -> PendingDecision | None:
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
    # Bidding
    # ------------------------------------------------------------------

    def _bid_step(self, game: InFlightGame) -> PendingDecision | None:
        active = [i for i in range(4) if not game.passed[i]]
        if (len(active) <= 1 and game.winning_bidder is not None) or len(active) == 0:
            self._resolve_bidding(game)
            return self._advance_until_decision(game)

        if game.bid_round >= 20:
            self._resolve_bidding(game)
            return self._advance_until_decision(game)

        bidder = game.current_bidder
        if game.passed[bidder]:
            self._next_bidder(game)
            return self._advance_until_decision(game)

        state_t = torch.from_numpy(game.gs.encode_state(bidder, te.DT_BID)).float()
        rust_legal = game.gs.legal_bids(bidder)

        # Build bid mask from raw Rust contract u8 values
        mask = torch.zeros(BID_ACTION_SIZE, dtype=torch.float32)
        mask[0] = 1.0  # pass is always legal
        for lb in rust_legal:
            if lb is not None:
                # Map Rust u8 contract → BID_ACTIONS index
                for bid_idx, rust_u8 in enumerate(BID_IDX_TO_RUST):
                    if rust_u8 == lb:
                        mask[bid_idx] = 1.0
                        break

        oracle_t = None
        if self.oracle:
            oracle_t = torch.from_numpy(game.gs.encode_oracle_state(bidder, te.DT_BID)).float()

        return PendingDecision(
            game_idx=game.game_id,
            player=bidder,
            state_tensor=state_t,
            legal_mask=mask,
            decision_type=DecisionType.BID,
            oracle_tensor=oracle_t,
        )

    def _apply_bid(self, game: InFlightGame, action_idx: int) -> None:
        bidder = game.current_bidder
        rust_u8 = BID_IDX_TO_RUST[action_idx] if action_idx < len(BID_IDX_TO_RUST) else None

        if rust_u8 is None:
            game.passed[bidder] = True
            game.gs.add_bid(bidder, None)
        else:
            rust_legal = game.gs.legal_bids(bidder)
            if rust_u8 not in rust_legal:
                game.passed[bidder] = True
                game.gs.add_bid(bidder, None)
            else:
                game.gs.add_bid(bidder, rust_u8)
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
            gs.set_role(game.winning_bidder, 0)
            for i in range(4):
                if i != game.winning_bidder:
                    gs.set_role(i, 2)

            game.contract_u8 = game.highest_bid
            game.declarer = game.winning_bidder

            if is_berac(game.highest_bid):
                gs.phase = te.PHASE_TRICK_PLAY
                game.phase = GamePhase.TRICK_PLAY
                game.lead_player = (game.dealer + 1) % 4
            elif is_klop(game.highest_bid):
                gs.phase = te.PHASE_TRICK_PLAY
                game.phase = GamePhase.TRICK_PLAY
                game.lead_player = (game.dealer + 1) % 4
            elif is_solo(game.highest_bid):
                gs.phase = te.PHASE_TALON_EXCHANGE
                tc = talon_cards_for_contract(game.highest_bid)
                if tc > 0:
                    game.phase = GamePhase.TALON_PICK
                else:
                    gs.phase = te.PHASE_TRICK_PLAY
                    game.phase = GamePhase.TRICK_PLAY
                    game.lead_player = (game.dealer + 1) % 4
            else:
                gs.phase = te.PHASE_KING_CALLING
                game.phase = GamePhase.KING_CALL
        else:
            gs.contract = 0
            for i in range(4):
                gs.set_role(i, 2)
            gs.phase = te.PHASE_TRICK_PLAY
            game.contract_u8 = 0
            game.declarer = None
            game.phase = GamePhase.TRICK_PLAY
            game.lead_player = (game.dealer + 1) % 4

    # ------------------------------------------------------------------
    # King Call
    # ------------------------------------------------------------------

    def _king_call_step(self, game: InFlightGame) -> PendingDecision | None:
        declarer = game.declarer
        callable_idxs = game.gs.callable_kings()
        if not callable_idxs:
            self._transition_after_king_call(game)
            return self._advance_until_decision(game)

        state_t = torch.from_numpy(
            game.gs.encode_state(declarer, te.DT_KING_CALL)
        ).float()

        # Build king mask from raw card indices
        mask = torch.zeros(KING_ACTION_SIZE, dtype=torch.float32)
        for card_idx in callable_idxs:
            suit = card_idx_to_suit(card_idx)
            if suit is not None:
                mask[suit] = 1.0

        oracle_t = None
        if self.oracle:
            oracle_t = torch.from_numpy(
                game.gs.encode_oracle_state(declarer, te.DT_KING_CALL)
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
        declarer = game.declarer
        callable_idxs = game.gs.callable_kings()

        # action_idx = suit index (0-3), find the king card with that suit
        chosen_card_idx = None
        for idx in callable_idxs:
            if card_idx_to_suit(idx) == action_idx:
                chosen_card_idx = idx
                break
        if chosen_card_idx is None:
            chosen_card_idx = callable_idxs[0]

        game.gs.set_called_king(chosen_card_idx)

        for p in range(4):
            if p != declarer:
                hand = game.gs.hand(p)
                if chosen_card_idx in hand:
                    game.gs.partner = p
                    game.gs.set_role(p, 1)
                    break

        self._transition_after_king_call(game)

    def _transition_after_king_call(self, game: InFlightGame) -> None:
        tc = talon_cards_for_contract(game.contract_u8)
        if tc > 0:
            game.gs.phase = te.PHASE_TALON_EXCHANGE
            game.phase = GamePhase.TALON_PICK
        else:
            game.gs.phase = te.PHASE_TRICK_PLAY
            game.phase = GamePhase.TRICK_PLAY
            game.lead_player = (game.dealer + 1) % 4

    # ------------------------------------------------------------------
    # Talon Exchange
    # ------------------------------------------------------------------

    def _talon_pick_step(self, game: InFlightGame) -> PendingDecision | None:
        declarer = game.declarer
        tc = talon_cards_for_contract(game.contract_u8)
        talon_idxs = game.gs.talon()
        group_size = 6 // (6 // tc) if tc in (1, 2, 3) else tc
        groups: list[list[int]] = []
        for i in range(0, len(talon_idxs), group_size):
            groups.append(talon_idxs[i : i + group_size])
        num_groups = len(groups)

        game.gs.set_talon_revealed(groups)

        state_t = torch.from_numpy(
            game.gs.encode_state(declarer, te.DT_TALON_PICK)
        ).float()

        mask = torch.zeros(TALON_ACTION_SIZE, dtype=torch.float32)
        for i in range(num_groups):
            mask[i] = 1.0

        oracle_t = None
        if self.oracle:
            oracle_t = torch.from_numpy(
                game.gs.encode_oracle_state(declarer, te.DT_TALON_PICK)
            ).float()

        game._talon_groups = groups
        game._talon_cards = tc

        return PendingDecision(
            game_idx=game.game_id,
            player=declarer,
            state_tensor=state_t,
            legal_mask=mask,
            decision_type=DecisionType.TALON_PICK,
            oracle_tensor=oracle_t,
        )

    def _apply_talon_pick(self, game: InFlightGame, action_idx: int) -> None:
        declarer = game.declarer
        groups = game._talon_groups
        tc = game._talon_cards
        num_groups = len(groups)

        if action_idx >= num_groups:
            action_idx = 0

        picked = groups[action_idx]
        for card_idx in picked:
            game.gs.add_to_hand(declarer, card_idx)
            game.gs.remove_from_talon(card_idx)

        # Heuristic discard: prefer low-point non-king suit cards
        hand = game.gs.hand(declarer)
        discardable = [idx for idx in hand if not is_king(idx) and not is_tarok(idx)]
        if len(discardable) < tc:
            discardable = [idx for idx in hand if not is_king(idx)]
        # Sort by card index as a proxy for point value (lower = less valuable)
        discardable.sort()
        for idx in discardable[:tc]:
            game.gs.remove_card(declarer, idx)
            game.gs.add_put_down(idx)

        game.gs.phase = te.PHASE_TRICK_PLAY
        game.phase = GamePhase.TRICK_PLAY
        game.lead_player = (game.dealer + 1) % 4

    # ------------------------------------------------------------------
    # Trick Play
    # ------------------------------------------------------------------

    def _trick_step(self, game: InFlightGame) -> PendingDecision | None:
        if game.trick_num >= 12:
            game.phase = GamePhase.DONE
            return None

        if game.trick_offset == 0:
            game.gs.start_trick(game.lead_player)

        player = (game.lead_player + game.trick_offset) % 4
        game.gs.current_player = player

        state_t = torch.from_numpy(
            game.gs.encode_state(player, te.DT_CARD_PLAY)
        ).float()
        legal_mask = torch.from_numpy(
            game.gs.legal_plays_mask(player)
        ).float()

        oracle_t = None
        if self.oracle:
            oracle_t = torch.from_numpy(
                game.gs.encode_oracle_state(player, te.DT_CARD_PLAY)
            ).float()

        return PendingDecision(
            game_idx=game.game_id,
            player=player,
            state_tensor=state_t,
            legal_mask=legal_mask,
            decision_type=DecisionType.CARD_PLAY,
            oracle_tensor=oracle_t,
        )

    def _apply_trick_card(self, game: InFlightGame, action_idx: int) -> None:
        player = (game.lead_player + game.trick_offset) % 4
        legal_cards = game.gs.legal_plays(player)
        if action_idx not in legal_cards:
            action_idx = legal_cards[0]

        game.gs.play_card(player, action_idx)
        game.trick_offset += 1

        if game.trick_offset >= 4:
            winner, points = game.gs.finish_trick()
            game.lead_player = winner
            game.trick_num += 1
            game.trick_offset = 0

            if game.trick_num >= 12:
                game.phase = GamePhase.DONE

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, game: InFlightGame, pending: PendingDecision, action_idx: int) -> None:
        if pending.decision_type == DecisionType.BID:
            self._apply_bid(game, action_idx)
        elif pending.decision_type == DecisionType.KING_CALL:
            self._apply_king_call(game, action_idx)
        elif pending.decision_type == DecisionType.TALON_PICK:
            self._apply_talon_pick(game, action_idx)
        elif pending.decision_type == DecisionType.CARD_PLAY:
            self._apply_trick_card(game, action_idx)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize_game(self, game: InFlightGame) -> GameResult:
        game.gs.phase = te.PHASE_SCORING
        scores_arr = game.gs.score_game()
        scores = [float(scores_arr[i]) for i in range(4)]

        # Convert raw experience dicts to Experience objects
        experiences = [
            Experience(
                state=e["state"],
                action=e["action"],
                log_prob=e["log_prob"],
                value=e["value"],
                decision_type=e["decision_type"],
                oracle_state=e["oracle_state"],
                legal_mask=e["legal_mask"],
                game_id=e["game_id"],
                step_in_game=e["step_in_game"],
            )
            for e in game.experiences
        ]

        winner = max(range(4), key=lambda i: scores[i])

        return GameResult(
            experiences=experiences,
            scores=scores,
            winner=winner,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _game_for_pending(
        self, games: list[InFlightGame | None], p: PendingDecision,
        game_id_to_slot: dict[int, int],
    ) -> InFlightGame | None:
        slot = game_id_to_slot.get(p.game_idx)
        if slot is not None:
            return games[slot]
        return None
