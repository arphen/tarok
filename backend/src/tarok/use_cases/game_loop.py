"""Main game loop use case — orchestrates a full game from deal to score.

This is the core orchestrator that depends on ports (PlayerPort, GameObserverPort)
but knows nothing about adapters (WebSocket, RL agents, etc.).
"""

from __future__ import annotations

import random

from tarok.entities.game_state import GameState, KontraLevel, Phase, Announcement
from tarok.entities.scoring import score_game
from tarok.ports.player_port import PlayerPort
from tarok.ports.observer_port import GameObserverPort
from tarok.use_cases.deal import deal
from tarok.use_cases.bid import place_bid
from tarok.use_cases.call_king import call_king
from tarok.use_cases.exchange_talon import reveal_talon, pick_talon_group, discard_cards
from tarok.use_cases.play_trick import start_trick, play_card


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


class GameLoop:
    """Runs a single game of Tarok end-to-end."""

    def __init__(
        self,
        players: list[PlayerPort],
        observer: GameObserverPort | None = None,
        rng: random.Random | None = None,
    ):
        assert len(players) == 4
        self._players = players
        self._observer: GameObserverPort = observer or NullObserver()  # type: ignore
        self._rng = rng or random.Random()

    async def run(self, dealer: int = 0) -> tuple[GameState, dict[int, int]]:
        """Play one full game. Returns (final_state, scores)."""
        state = GameState(dealer=dealer)
        await self._observer.on_game_start(state)

        # === DEAL ===
        state.phase = Phase.DEALING
        state = deal(state, self._rng)
        await self._observer.on_deal(state)

        # === BIDDING ===
        max_rounds = 20  # Safety limit
        rounds = 0
        while state.phase == Phase.BIDDING and rounds < max_rounds:
            player_idx = state.current_bidder
            legal = state.legal_bids(player_idx)
            bid = await self._players[player_idx].choose_bid(state, player_idx, legal)
            if bid is not None and bid not in legal:
                bid = None  # fall back to pass if bid is illegal
            state = place_bid(state, player_idx, bid)
            await self._observer.on_bid(player_idx, bid, state)
            rounds += 1

        if state.phase == Phase.DEALING:
            # Everyone passed — re-deal
            # (This shouldn't happen now that klop exists, but keep as safety)
            return await self.run(dealer=(dealer + 1) % 4)

        assert state.contract is not None

        # Klop: skip king/talon/announcements, go straight to tricks
        if state.contract.is_klop:
            await self._observer.on_contract_won(-1, state.contract, state)
        else:
            assert state.declarer is not None
            await self._observer.on_contract_won(state.declarer, state.contract, state)

        # === KING CALLING (if not solo and not klop) ===
        if state.phase == Phase.KING_CALLING:
            callable = state.callable_kings()
            if callable:
                king = await self._players[state.declarer].choose_king(
                    state, state.declarer, callable
                )
                state = call_king(state, king)
                await self._observer.on_king_called(state.declarer, king, state)
            else:
                state.phase = Phase.TALON_EXCHANGE

        # === TALON EXCHANGE ===
        if state.phase == Phase.TALON_EXCHANGE and state.contract.talon_cards > 0:
            groups = reveal_talon(state)
            await self._observer.on_talon_revealed(groups, state)

            group_idx = await self._players[state.declarer].choose_talon_group(
                state, state.declarer, groups
            )
            picked = list(state.talon_revealed[group_idx])  # copy before mutation
            state = pick_talon_group(state, group_idx)
            await self._observer.on_talon_group_picked(state)

            discards = await self._players[state.declarer].choose_discard(
                state, state.declarer, state.contract.talon_cards
            )
            state = discard_cards(state, discards)
            await self._observer.on_talon_exchanged(state, picked=picked, discarded=discards)
        elif state.phase == Phase.TALON_EXCHANGE:
            state.phase = Phase.ANNOUNCEMENTS

        # === ANNOUNCEMENTS + KONTRA/RE/SUB ===
        if state.phase == Phase.ANNOUNCEMENTS:
            # Round-robin: each player can announce or kontra, repeat until all pass.
            # Max rounds to prevent infinite loops (5 announcement types × 4 kontra levels = bounded).
            MAX_ROUNDS = 20
            for _round in range(MAX_ROUNDS):
                any_action = False
                for p in range(state.num_players):
                    if not hasattr(self._players[p], 'choose_announce_action'):
                        # Legacy path: player doesn't support the new protocol
                        continue
                    action = await self._players[p].choose_announce_action(state, p)
                    if action == 0:  # PASS
                        continue

                    any_action = True

                    # Import action constants (avoid circular)
                    from tarok.adapters.ai.encoding import (
                        ANNOUNCE_IDX_TO_ANN,
                        KONTRA_IDX_TO_KEY,
                    )

                    if action in ANNOUNCE_IDX_TO_ANN:
                        ann = ANNOUNCE_IDX_TO_ANN[action]
                        if p not in state.announcements:
                            state.announcements[p] = []
                        if ann not in state.announcements[p]:
                            state.announcements[p].append(ann)
                    elif action in KONTRA_IDX_TO_KEY:
                        key = KONTRA_IDX_TO_KEY[action]
                        current = state.kontra_levels.get(key, KontraLevel.NONE)
                        next_level = current.next_level
                        if next_level is not None:
                            state.kontra_levels[key] = next_level

                if not any_action:
                    break

            state.phase = Phase.TRICK_PLAY
            # Forehand leads first trick
            state.current_player = (state.dealer + 1) % state.num_players

        # === TRICK PLAY ===
        from tarok.entities.card import CardType
        while state.phase == Phase.TRICK_PLAY:
            state = start_trick(state)
            await self._observer.on_trick_start(state)
            for card_num in range(state.num_players):
                player_idx = state.current_player
                legal = state.legal_plays(player_idx)
                card = await self._players[player_idx].choose_card(
                    state, player_idx, legal
                )
                
                # Rule Verification: Check if playing tarok on suit lead
                if state.current_trick.cards:
                    led_card = state.current_trick.cards[0][1]
                    if led_card.card_type == CardType.SUIT and card.card_type == CardType.TAROK:
                        led_suit = led_card.suit
                        await self._observer.on_rule_verified(
                            player_idx, 
                            f"Played a Tarok because they have no {led_suit.value.title() if hasattr(led_suit, 'value') else led_suit}s left", 
                            state
                        )

                is_last_card = card_num == state.num_players - 1
                if is_last_card:
                    # Before resolving the trick, add the card to the trick
                    # so we can broadcast the full 4-card trick
                    state.hands[player_idx].remove(card)
                    state.current_trick.cards.append((player_idx, card))
                    # Broadcast with all 4 cards visible on the table
                    await self._observer.on_card_played(player_idx, card, state)
                    # Now resolve the trick (moves to state.tricks, clears current_trick)
                    from tarok.use_cases.play_trick import _resolve_trick
                    _resolve_trick(state)
                else:
                    state = play_card(state, player_idx, card)
                    await self._observer.on_card_played(player_idx, card, state)

            if state.tricks:
                last_trick = state.tricks[-1]
                winner = last_trick.winner()
                await self._observer.on_trick_won(last_trick, winner, state)

                # Berač early termination: declarer wins a trick → instant loss
                if (
                    state.contract
                    and state.contract.is_berac
                    and state.declarer is not None
                    and winner == state.declarer
                ):
                    state.phase = Phase.SCORING
                    break

        # === SCORING ===
        scores = score_game(state)
        state.scores = scores
        state.phase = Phase.FINISHED
        await self._observer.on_game_end(scores, state)

        return state, scores
