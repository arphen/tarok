#!/usr/bin/env python3
"""Estimate average full move-tree size for late Tarok trick play.

Method:
1. Deal random hands with the Rust engine.
2. Skip bidding/talon and play random legal cards until each player has N cards.
3. Exhaustively enumerate every legal continuation from that position.

The exhaustive count returns:
- leaf_count: number of complete legal play sequences to game end
- node_count: number of decision nodes explored (includes root)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

# macOS/PyO3+libtorch workaround: ensure torch dylibs are loaded first.
import torch  # noqa: F401
import tarok_engine as te

if TYPE_CHECKING:
    from collections.abc import Sequence

NUM_PLAYERS = 4
TRICKS_PER_GAME = 12


@dataclass(frozen=True)
class EndgameState:
    hands: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    lead_player: int
    trick_index: int
    trick_cards: tuple[tuple[int, int], ...]


def _remove_card(hand: tuple[int, ...], card: int) -> tuple[int, ...]:
    return tuple(c for c in hand if c != card)


def _random_roll_to_n(
    *,
    n_cards_left: int,
    dealer: int,
    rng: random.Random,
    contract: int | None,
) -> EndgameState:
    gs = te.RustGameState(dealer)
    gs.deal()

    hands = [tuple(sorted(gs.hand(p))) for p in range(NUM_PLAYERS)]
    lead_player = (dealer + 1) % NUM_PLAYERS
    trick_index = 0

    while trick_index < TRICKS_PER_GAME - n_cards_left:
        trick_cards: list[tuple[int, int]] = []

        for offset in range(NUM_PLAYERS):
            player = (lead_player + offset) % NUM_PLAYERS
            legal = te.compute_legal_plays(
                list(hands[player]),
                trick_cards,
                contract,
                trick_index == TRICKS_PER_GAME - 1,
            )
            chosen = rng.choice(legal)
            hands[player] = _remove_card(hands[player], chosen)
            trick_cards.append((player, chosen))

        winner = te.evaluate_trick_winner(
            trick_cards,
            trick_index == TRICKS_PER_GAME - 1,
            contract,
        )
        lead_player = winner
        trick_index += 1

    return EndgameState(
        hands=(hands[0], hands[1], hands[2], hands[3]),
        lead_player=lead_player,
        trick_index=trick_index,
        trick_cards=(),
    )


@lru_cache(maxsize=2_000_000)
def _count_full_tree(state: EndgameState, contract: int | None) -> tuple[int, int]:
    """Return (node_count, leaf_count) from the given state to game end."""
    if state.trick_index >= TRICKS_PER_GAME:
        return (1, 1)

    current_player = (state.lead_player + len(state.trick_cards)) % NUM_PLAYERS
    legal = te.compute_legal_plays(
        list(state.hands[current_player]),
        list(state.trick_cards),
        contract,
        state.trick_index == TRICKS_PER_GAME - 1,
    )

    node_total = 1
    leaf_total = 0

    for card in legal:
        hands_mut = list(state.hands)
        hands_mut[current_player] = _remove_card(hands_mut[current_player], card)
        next_cards = state.trick_cards + ((current_player, card),)

        if len(next_cards) == NUM_PLAYERS:
            winner = te.evaluate_trick_winner(
                list(next_cards),
                state.trick_index == TRICKS_PER_GAME - 1,
                contract,
            )
            child = EndgameState(
                hands=(
                    hands_mut[0],
                    hands_mut[1],
                    hands_mut[2],
                    hands_mut[3],
                ),
                lead_player=winner,
                trick_index=state.trick_index + 1,
                trick_cards=(),
            )
        else:
            child = EndgameState(
                hands=(
                    hands_mut[0],
                    hands_mut[1],
                    hands_mut[2],
                    hands_mut[3],
                ),
                lead_player=state.lead_player,
                trick_index=state.trick_index,
                trick_cards=next_cards,
            )

        child_nodes, child_leaves = _count_full_tree(child, contract)
        node_total += child_nodes
        leaf_total += child_leaves

    return (node_total, leaf_total)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average endgame full-tree size for last N cards per player",
    )
    parser.add_argument("--n", type=int, default=2, help="Cards left per player in sampled states")
    parser.add_argument("--samples", type=int, default=200, help="Number of random sampled states")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--contract",
        type=int,
        default=-1,
        help=(
            "Rust contract id (0=Klop,1=Three,...,9=BarvniValat). "
            "Use -1 for None (default)."
        ),
    )
    parser.add_argument(
        "--random-dealer",
        action="store_true",
        help="Sample a random dealer per hand (default dealer=0)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (1 <= args.n <= 12):
        raise SystemExit("--n must be in [1, 12]")
    if args.samples <= 0:
        raise SystemExit("--samples must be > 0")

    contract: int | None = None if args.contract < 0 else args.contract
    rng = random.Random(args.seed)

    total_nodes = 0
    total_leaves = 0

    for i in range(args.samples):
        dealer = rng.randrange(NUM_PLAYERS) if args.random_dealer else 0
        state = _random_roll_to_n(
            n_cards_left=args.n,
            dealer=dealer,
            rng=rng,
            contract=contract,
        )
        nodes, leaves = _count_full_tree(state, contract)
        total_nodes += nodes
        total_leaves += leaves

        if (i + 1) % max(1, args.samples // 10) == 0:
            print(f"progress: {i + 1}/{args.samples}")

    avg_nodes = total_nodes / args.samples
    avg_leaves = total_leaves / args.samples

    print("\n=== Endgame Full-Tree Estimate ===")
    print(f"samples: {args.samples}")
    print(f"N cards left/player: {args.n}")
    print(f"contract: {contract}")
    print(f"avg decision nodes: {avg_nodes:.2f}")
    print(f"avg complete sequences (leaves): {avg_leaves:.2f}")
    print(f"log10(avg leaves): {math.log10(avg_leaves):.4f}" if avg_leaves > 0 else "log10(avg leaves): -inf")


if __name__ == "__main__":
    main()
