#!/usr/bin/env python3
"""Benchmark PIMC on Berač at very low world counts, with a configurable
handoff trick.

Berač is a negative contract: the declarer must NOT win any trick. The
Berač-specialised PIMC in the engine runs a Boolean α-β survival search
per world ("can declarer reach empty hands with 0 tricks won?") that
aborts the instant declarer would be forced to win a trick. That makes
very low `num_worlds` viable, and makes the cost a strong function of
how many tricks remain.

This script uses the specialised PIMC only from a configurable handoff
trick onwards, and delegates earlier tricks to a cheap heuristic, so you
can get a feel for the cost before unleashing PIMC on trick 1.

Output is streamed live (stdout is unbuffered) so you can watch each
decision as it happens — including the per-card survival vote
distribution — and decide when to kill the run if it's too slow.

Usage
-----
    # quick first feel: 1 game, 2 worlds, PIMC only for last 4 tricks
    python backend/scripts/bench_pimc_berac.py

    # PIMC from trick 5 onwards (8 tricks of PIMC), 2 worlds, 1 game
    python backend/scripts/bench_pimc_berac.py --handoff-trick 4

    # full-hand PIMC from trick 1, 2 worlds, 1 game (may be SLOW)
    python backend/scripts/bench_pimc_berac.py --handoff-trick 0

Notes
-----
* `--handoff-trick N` means "start using PIMC from the (N+1)-th trick".
  Default 8 → last 4 tricks (tricks 9..12).
* Before the handoff, the declarer plays with ``stockskis-v5`` like the
  opponents. It's a heuristic that knows about Berač and tends to shed
  dangerous cards.
* The game ends early the moment the declarer wins a trick, so in many
  random hands the declarer is dead well before the handoff trick and
  PIMC never runs. Use ``--seed`` to try different deals.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Make the backend package importable when running from the repo root.
_BACKEND_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

# Import torch before tarok_engine so the PyO3 extension can resolve
# @rpath/libtorch_cpu.dylib on macOS.
import torch  # noqa: E402, F401

import tarok_engine as te  # noqa: E402

from tarok.adapters.players.stockskis_player import StockskisPlayer  # noqa: E402
from tarok.entities import Card, Contract  # noqa: E402
from tarok.entities.game_types import DECK, CardType, Suit, SuitRank  # noqa: E402
from tarok.ports.player_port import PlayerPort  # noqa: E402
from tarok.use_cases.game_loop import NullObserver, RustGameLoop  # noqa: E402


# Force unbuffered stdout so progress prints stream live.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# First-move heuristic (skip PIMC on trick 1)
# ---------------------------------------------------------------------------


# "High cover" in a suit = holding a Jack/Knight/Queen/King — any of those
# tends to win a trick, so they are the cards we're afraid of being forced
# to play. If we hold one, we want to dump the LOW cards of the same suit
# first, while the suit is still "live" and we can be underbid.
_HIGH_RANK_VALUES = {
    SuitRank.JACK.value,
    SuitRank.KNIGHT.value,
    SuitRank.QUEEN.value,
    SuitRank.KING.value,
}

_KING_VALUE = SuitRank.KING.value
_SKIS_TAROK_VALUE = 22

# "Low pip" per user's spec: 3 or 4 in hearts/diamonds, 7 or 8 in clubs/spades.
# In our engine, hearts/diamonds pips are labelled by value (1..4), while
# clubs/spades pips are labelled "7..10" for values 1..4 — so the "7/8" labels
# the user named correspond to values {1, 2}.
_LOW_PIP_BY_SUIT: dict[Suit, set[int]] = {
    Suit.HEARTS: {3, 4},
    Suit.DIAMONDS: {3, 4},
    Suit.CLUBS: {1, 2},
    Suit.SPADES: {1, 2},
}


def _berac_bid_veto_reason(hand: list[Card]) -> str | None:
    """Return a reason to refuse bidding Berač, or None if the hand is eligible.

    Currently: holding Škis (tarok 22) — it can never be ducked, so any trick
    it follows into is automatically won.
    """
    for c in hand:
        if c.card_type == CardType.TAROK and c.value == _SKIS_TAROK_VALUE:
            return "hand contains Škis"
    return None


def _pick_highest_legal(legal: list[Card]) -> Card:
    """Highest-valued legal card — aims to win the trick when following."""
    return max(legal, key=lambda c: (c.value, c._idx))


def _pick_lowest_legal(legal: list[Card]) -> Card:
    """Lowest-valued legal card — safe dump when following."""
    return min(legal, key=lambda c: (c.value, c._idx))


def _pick_opp_trick2_lead(hand: list[Card], legal: list[Card]) -> tuple[Card, str]:
    """Opponent's aggressive lead on trick 2 after winning trick 1.

    Three heuristic options (first applicable wins):
      A. Lowest tarok, if it is lower than VIII — forces declarer to keep
         dumping taroks and eventually get stuck on a high one.
      B. Otherwise, if the hand has a suit where we hold BOTH a king and a
         "low pip" (3/4 in H/D, 7/8 in C/S), lead that low pip. The must-
         overplay rule then forces declarer to burn a higher card in that
         suit, often winning the trick they didn't want.
      C. Fallback: lead our lowest suit card.
    """
    legal_set = {c._idx for c in legal}
    taroks = sorted(
        (c for c in hand if c.card_type == CardType.TAROK),
        key=lambda c: c.value,
    )

    # Option A
    if taroks and taroks[0].value < 8 and taroks[0]._idx in legal_set:
        return taroks[0], f"opt-A lowest-tarok ({taroks[0].label})"

    # Option B
    suit_cards: dict[Suit, list[Card]] = {}
    for c in hand:
        if c.card_type == CardType.SUIT and c.suit is not None:
            suit_cards.setdefault(c.suit, []).append(c)
    for suit, cards in suit_cards.items():
        has_king = any(c.value == _KING_VALUE for c in cards)
        if not has_king:
            continue
        low_pips = sorted(
            (c for c in cards if c.value in _LOW_PIP_BY_SUIT.get(suit, set())),
            key=lambda c: c.value,
        )
        for lp in low_pips:
            if lp._idx in legal_set:
                return lp, f"opt-B low-pip+king ({lp.label}, king in {suit.name})"

    # Option C
    suit_sorted = sorted(
        (c for c in hand if c.card_type == CardType.SUIT),
        key=lambda c: c.value,
    )
    for c in suit_sorted:
        if c._idx in legal_set:
            return c, f"opt-C lowest-suit ({c.label})"

    fb = _pick_lowest_legal(legal)
    return fb, f"opt-C fallback ({fb.label})"


def pick_first_move(hand: list[Card], legal: list[Card]) -> tuple[Card, str]:
    """Pick a Berač opening lead without running PIMC.

    Rationale
    ---------
    * The **highest tarok** forces every other player to follow with tarok
      and tends to pull their highest taroks too. A trump lead we cannot
      be beaten on.
    * A **low suit card in a suit where we also hold a Jack / Knight /
      Queen / King** lets opponents play their high cards of that suit
      while we shed a safe low one. Later the suit is "thinned" and our
      high holding is less dangerous.

    Preference order:
        1. If we hold tarok 21 (Mond) or 22 (Škis), lead it.
        2. Else if we hold any suit with cover (J/N/Q/K), lead the
           LOWEST such suit's LOWEST card.
        3. Else if we hold any tarok, lead our highest.
        4. Fallback: the first legal play.
    """

    legal_set = {c._idx for c in legal}

    # -- category 1: Mond (21) or Škis (22) if legal -----------------------
    taroks = sorted(
        (c for c in hand if c.card_type == CardType.TAROK),
        key=lambda c: c.value,
        reverse=True,
    )
    for c in taroks:
        if c.value >= 21 and c._idx in legal_set:
            return c, f"highest-tarok ({c.label})"

    # -- category 2: low card of a suit where we hold a J/N/Q/K -----------
    suits_covered: dict = {}
    for c in hand:
        if c.card_type != CardType.TAROK and c.value in _HIGH_RANK_VALUES:
            suits_covered[c.suit] = True
    if suits_covered:
        # Lowest card of the lowest-indexed covered suit, that is legal.
        for suit in suits_covered:
            suit_cards_low_first = sorted(
                (c for c in hand if c.suit == suit),
                key=lambda c: c.value,
            )
            for c in suit_cards_low_first:
                if c._idx in legal_set and c.value not in _HIGH_RANK_VALUES:
                    return c, f"low-of-covered-suit ({c.label}, cover in {suit.name})"

    # -- category 3: highest tarok fallback -------------------------------
    for c in taroks:
        if c._idx in legal_set:
            return c, f"highest-tarok-fallback ({c.label})"

    # -- category 4: first legal play -------------------------------------
    return legal[0], f"fallback ({legal[0].label})"


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


class PimcBeracDeclarer(PlayerPort):
    """Declarer that bids Berač.

    Before ``handoff_trick``, delegates card play to a StockSkis heuristic
    bot (fast). From ``handoff_trick`` onwards, calls
    ``te.pimc_berac_votes`` with the requested ``num_worlds``.
    """

    def __init__(
        self,
        *,
        num_worlds: int,
        handoff_trick: int,
        seed_base: int = 0xB3C_0DE,
        pre_handoff_variant: str = "v5",
    ):
        self._num_worlds = num_worlds
        self._handoff_trick = handoff_trick
        self._seed_base = seed_base
        self._fallback = StockskisPlayer(variant=pre_handoff_variant, name="fallback")
        self._pimc_decision_count = 0
        self.decision_log: list[dict] = []
        self.opening_hand_labels: list[str] = []
        self.first_lead_label: str | None = None
        self.first_lead_reason: str | None = None

    @property
    def name(self) -> str:
        return f"PIMC-Berač(W={self._num_worlds}, handoff={self._handoff_trick})"

    async def choose_bid(self, state, player_idx, legal_bids):
        if Contract.BERAC not in legal_bids:
            return None
        gs = getattr(state, "_rust_gs", None)
        if gs is not None:
            hand = [DECK[i] for i in gs.hand(player_idx)]
            veto = _berac_bid_veto_reason(hand)
            if veto is not None:
                log(f"  [BID-VETO] declarer passes: {veto}")
                return None
        return Contract.BERAC

    async def choose_king(self, state, player_idx, callable_kings):
        return callable_kings[0]

    async def choose_talon_group(self, state, player_idx, talon_groups):
        return 0

    async def choose_discard(self, state, player_idx, must_discard):
        return []

    async def choose_announcements(self, state, player_idx):
        return []

    async def choose_card(self, state, player_idx, legal_plays):
        if len(legal_plays) == 1:
            return legal_plays[0]

        gs = getattr(state, "_rust_gs", None)
        tricks_played = getattr(state, "tricks_played", 0)

        if gs is not None and not self.opening_hand_labels:
            self.opening_hand_labels = [DECK[i].label for i in gs.hand(player_idx)]

        # ---- First-move heuristic -----------------------------------------
        # On the very first trick, when we are leading to an empty trick,
        # PIMC wastes a huge amount of work exploring obviously-losing
        # opening leads. Use a cheap heuristic instead.
        if (
            tricks_played == 0
            and gs is not None
            and len(gs.current_trick_cards()) == 0
        ):
            hand = [DECK[i] for i in gs.hand(player_idx)]
            chosen, reason = pick_first_move(hand, legal_plays)
            self.first_lead_label = chosen.label
            self.first_lead_reason = reason
            log(
                f"  [HEURISTIC] trick  1 | hand size {len(hand):>2} | "
                f"{len(legal_plays)} legal move(s) — "
                f"picked {chosen.label}  [{reason}]"
            )
            return chosen

        # Pre-handoff (but not trick 1 lead): cheap heuristic.
        if gs is None or tricks_played < self._handoff_trick:
            return await self._fallback.choose_card(state, player_idx, legal_plays)

        # PIMC decision.
        seed = (
            self._seed_base ^ (self._pimc_decision_count * 0x9E37_79B9_7F4A_7C15)
        ) & 0xFFFF_FFFF_FFFF_FFFF
        self._pimc_decision_count += 1

        trick_no = tricks_played + 1  # 1-indexed for humans
        hand_size = len(gs.hand(player_idx))
        log(
            f"  [PIMC] trick {trick_no:>2} | hand size {hand_size:>2} | "
            f"{len(legal_plays)} legal move(s) — running "
            f"{self._num_worlds} world(s)…"
        )

        t0 = time.perf_counter()
        votes = te.pimc_berac_votes(gs, player_idx, self._num_worlds, seed)
        dt = time.perf_counter() - t0

        # votes: list[(card_idx, survival_count, world_count)]
        best_idx = -1
        best_frac = -1.0
        vote_strs = []
        for card_idx, sur, cnt in votes:
            frac = (sur / cnt) if cnt > 0 else -1.0
            vote_strs.append(f"{DECK[card_idx].label}:{sur}/{cnt}")
            if frac > best_frac:
                best_frac = frac
                best_idx = card_idx

        chosen = DECK[best_idx] if best_idx >= 0 else legal_plays[0]
        chosen_sur = 0
        chosen_cnt = 0
        for card_idx, sur, cnt in votes:
            if card_idx == best_idx:
                chosen_sur = sur
                chosen_cnt = cnt
                break
        log(
            f"    → chose {chosen.label} "
            f"(survival fraction {best_frac:.2f})"
            f" in {dt * 1000:.1f} ms   votes: {', '.join(vote_strs)}"
        )
        self.decision_log.append(
            {
                "trick": trick_no,
                "hand_size": hand_size,
                "num_legal": len(legal_plays),
                "wall_time": dt,
                "chosen": chosen.label,
                "best_frac": best_frac,
                "chosen_survival": chosen_sur,
                "chosen_worlds": chosen_cnt,
                "all_votes": vote_strs,
            }
        )
        return chosen


class ScriptedBeracOpponent(PlayerPort):
    """Opponent that passes on bidding and plays scripted trick-1/2 moves.

    Play policy:
      * Trick 1 (declarer leads): play the **highest legal card** to win the
        trick and bring the next seat into play.
      * Trick 2, if THIS opponent won trick 1: lead via the 3-option
        heuristic (see ``_pick_opp_trick2_lead``).
      * Trick 2, following another opponent's lead: play **lowest legal**
        (partner mirror — forces declarer to keep burning through cards).
      * Trick 3 onward: delegate to StockSkis ``v5``.
    """

    def __init__(self, *, variant: str = "v5", name: str = "Opp"):
        self._stockskis = StockskisPlayer(variant=variant, name=name)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def choose_bid(self, state, player_idx, legal_bids):
        return None

    async def choose_king(self, state, player_idx, callable_kings):
        return callable_kings[0]

    async def choose_talon_group(self, state, player_idx, talon_groups):
        return 0

    async def choose_discard(self, state, player_idx, must_discard):
        return []

    async def choose_announcements(self, state, player_idx):
        return []

    async def choose_card(self, state, player_idx, legal_plays):
        if len(legal_plays) == 1:
            return legal_plays[0]

        gs = getattr(state, "_rust_gs", None)
        if gs is None:
            return await self._stockskis.choose_card(state, player_idx, legal_plays)

        tricks_played = getattr(state, "tricks_played", 0)
        current_trick = gs.current_trick_cards()

        # Trick 1 — we are following declarer's lead: play the HIGHEST legal
        # card to win the trick.
        if tricks_played == 0 and len(current_trick) > 0:
            chosen = _pick_highest_legal(legal_plays)
            log(
                f"  [OPP-SCRIPT] seat {player_idx} trick 1 follow — "
                f"highest legal {chosen.label}"
            )
            return chosen

        # Trick 2 — split on lead vs follow.
        if tricks_played == 1:
            prev_winner = (
                state.tricks[0].winner()
                if getattr(state, "tricks", None)
                else None
            )
            if len(current_trick) == 0 and prev_winner == player_idx:
                hand = [DECK[i] for i in gs.hand(player_idx)]
                chosen, reason = _pick_opp_trick2_lead(hand, legal_plays)
                log(
                    f"  [OPP-SCRIPT] seat {player_idx} trick 2 LEAD — "
                    f"{chosen.label}  [{reason}]"
                )
                return chosen
            if len(current_trick) > 0:
                chosen = _pick_lowest_legal(legal_plays)
                log(
                    f"  [OPP-SCRIPT] seat {player_idx} trick 2 follow — "
                    f"lowest legal {chosen.label}"
                )
                return chosen

        # Trick 3+ or we are leading trick 2 but did not win trick 1 (shouldn't
        # happen in berač since the game would have ended) → StockSkis.
        return await self._stockskis.choose_card(state, player_idx, legal_plays)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_one_game(
    seed: int,
    num_worlds: int,
    handoff_trick: int,
    opponent_variant: str,
) -> dict[str, Any] | None:
    """Play a single game; return True if Berač was actually bid."""

    log(f"\n=== game seed={seed} | worlds={num_worlds} | handoff_trick={handoff_trick} ===")

    declarer = PimcBeracDeclarer(
        num_worlds=num_worlds,
        handoff_trick=handoff_trick,
        seed_base=seed,
    )
    players: list[PlayerPort] = [
        declarer,
        ScriptedBeracOpponent(variant=opponent_variant, name="Opp-1"),
        ScriptedBeracOpponent(variant=opponent_variant, name="Opp-2"),
        ScriptedBeracOpponent(variant=opponent_variant, name="Opp-3"),
    ]

    loop = RustGameLoop(
        players,
        observer=NullObserver(),
        rng=random.Random(seed),
    )

    t0 = time.perf_counter()
    # dealer=3 → seat 0 (declarer) bids first.
    state, scores = await loop.run(dealer=3)
    wall = time.perf_counter() - t0

    if state.contract is None or not state.contract.is_berac or state.declarer != 0:
        log(f"  skipped: contract={state.contract} declarer={state.declarer} "
            f"(opening hand didn't support Berač)")
        return None

    declarer_tricks = sum(1 for t in state.tricks if t.winner() == 0)
    won = scores[0] == 70

    log(
        f"  result: declarer_score={scores[0]:+d}  "
        f"declarer_tricks={declarer_tricks}  "
        f"tricks_played={state.tricks_played}/12  "
        f"won={won}  total_wall={wall * 1000:.1f} ms"
    )

    if declarer.decision_log:
        total_pimc = sum(d["wall_time"] for d in declarer.decision_log)
        log(
            f"  PIMC summary: {len(declarer.decision_log)} decision(s), "
            f"total {total_pimc * 1000:.1f} ms, "
            f"mean {total_pimc * 1000 / len(declarer.decision_log):.1f} ms/decision"
        )
    else:
        log("  PIMC summary: no PIMC decisions ran "
            "(declarer lost / contract ended before handoff_trick).")

    all_worlds_survived = bool(declarer.decision_log) and all(
        d["chosen_worlds"] > 0 and d["chosen_survival"] == d["chosen_worlds"]
        for d in declarer.decision_log
    )
    if all_worlds_survived:
        log("  perfect-survival: yes (chosen move survived every sampled world at every PIMC decision)")
    else:
        log("  perfect-survival: no")

    return {
        "seed": seed,
        "won": won,
        "declarer_score": scores[0],
        "tricks_played": state.tricks_played,
        "declarer_tricks": declarer_tricks,
        "opening_hand": declarer.opening_hand_labels,
        "first_lead_label": declarer.first_lead_label,
        "first_lead_reason": declarer.first_lead_reason,
        "decision_log": declarer.decision_log,
        "all_worlds_survived": all_worlds_survived,
    }


async def main_async(args: argparse.Namespace) -> None:
    played = 0
    seed = args.seed
    attempts = 0
    max_attempts = args.num_games * 20
    perfect_examples: list[dict[str, Any]] = []
    while played < args.num_games and attempts < max_attempts:
        game = await run_one_game(
            seed=seed,
            num_worlds=args.worlds,
            handoff_trick=args.handoff_trick,
            opponent_variant=args.opponent,
        )
        seed += 1
        attempts += 1
        if game is not None:
            played += 1
            if game["all_worlds_survived"] and len(perfect_examples) < args.print_perfect_hands:
                perfect_examples.append(game)

    if played < args.num_games:
        log(
            f"\nStopped after {attempts} seeds — could only complete {played}/"
            f"{args.num_games} Berač games."
        )

    if args.print_perfect_hands > 0:
        log("\n=== hands where chosen move survived all sampled worlds ===")
        if not perfect_examples:
            log("  none found in this run")
        for i, game in enumerate(perfect_examples, start=1):
            log(
                f"  [{i}] seed={game['seed']} score={game['declarer_score']:+d} "
                f"tricks={game['tricks_played']}/12 declarer_tricks={game['declarer_tricks']}"
            )
            if game["first_lead_label"] is not None:
                log(
                    f"      first lead: {game['first_lead_label']} "
                    f"[{game['first_lead_reason']}]"
                )
            log(f"      opening hand: {' '.join(game['opening_hand'])}")
            for d in game["decision_log"]:
                log(
                    f"      trick {d['trick']:>2}: chose {d['chosen']} "
                    f"{d['chosen_survival']}/{d['chosen_worlds']}"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--num-games", type=int, default=1,
        help="number of Berač games to play (default: 1)",
    )
    ap.add_argument(
        "--worlds", type=int, default=2,
        help="PIMC world count (default: 2)",
    )
    ap.add_argument(
        "--handoff-trick", type=int, default=8,
        help=(
            "trick index (0-based) at which PIMC takes over. "
            "8 = last 4 tricks; 0 = PIMC from trick 1. Default: 8."
        ),
    )
    ap.add_argument(
        "--opponent", default="v5", choices=["v5", "m6"],
        help="stockskis variant for opponents AND pre-handoff declarer play "
             "(default: v5)",
    )
    ap.add_argument(
        "--seed", type=int, default=0xDEAD_BEEF,
        help="starting RNG seed (default: 0xDEADBEEF)",
    )
    ap.add_argument(
        "--print-perfect-hands", type=int, default=3,
        help=(
            "print up to N example hands where every chosen PIMC move "
            "survived all sampled worlds (default: 3)"
        ),
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
