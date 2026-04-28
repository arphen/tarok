"""Helpers for per-iteration learner bidding/contract diagnostics."""

from __future__ import annotations

from typing import Any


_CONTRACT_LABELS: dict[int, str] = {
    0: "klop",
    1: "three",
    2: "two",
    3: "one",
    4: "solo_three",
    5: "solo_two",
    6: "solo_one",
    7: "solo",
    8: "berac",
    9: "barvni_valat",
}

_PASS_LABEL = "pass"


def _contract_label(cid: int) -> str:
    return _CONTRACT_LABELS.get(cid, f"contract_{cid}")


def compute_learner_contract_stats(
    raw: dict[str, Any],
    learner_seats: list[int],
    learner_seat_per_game: Any = None,
) -> dict[str, dict[str, float | int]]:
    """Compute quick learner diagnostics per contract.

    Parameters
    ----------
    raw:
        Self-play result dict containing ``contracts``, ``declarers``,
        ``bid_contracts`` and ``scores`` arrays.
    learner_seats:
        Constant set of seats that are learners for the whole run. Used in
        the legacy (non-duplicate) path where the learner sits at fixed
        positions throughout.
    learner_seat_per_game:
        Optional per-game learner seat index (shape ``(n_games,)``). When
        provided, overrides ``learner_seats`` on a per-game basis — required
        in duplicate mode where the learner rotates across all four seats
        within each pod. Counting every seat's bid in that mode would
        incorrectly attribute opponent bots' bids to the learner.

    Returned shape:
      {
        "three": {"bids_made": 12, "bids_won": 5, "contracts_won": 3},
        ...
      }
    """
    contracts = raw.get("contracts")
    declarers = raw.get("declarers")
    bid_contracts = raw.get("bid_contracts")
    scores = raw.get("scores")
    reward_scores = raw.get("reward_scores", scores)
    if contracts is None or declarers is None or bid_contracts is None or scores is None:
        return {}

    n_games = len(contracts)
    if n_games == 0:
        return {}

    stats: dict[str, dict[str, float | int]] = {}

    def ensure(label: str) -> dict[str, float | int]:
        row = stats.get(label)
        if row is None:
            row = {
                "bids_made": 0,
                "bids_won": 0,
                "contracts_won": 0,
                # Bid-level reward aggregate: all games where learner made this bid,
                # regardless of whether the bid won.
                "bid_reward_sum": 0.0,
                "bid_reward_count": 0,
                # Played-level aggregates: games where learner actually played this
                # final contract as declarer. For klop (no declarer), count games
                # where final contract is klop and learner participated.
                "played_count": 0,
                "played_score_sum": 0.0,
                "played_reward_sum": 0.0,
            }
            stats[label] = row
        return row

    default_learner_set = {int(s) for s in learner_seats}

    for gid in range(n_games):
        declarer = int(declarers[gid])
        contract_id = int(contracts[gid])
        contract_label = _contract_label(contract_id)

        if learner_seat_per_game is not None:
            learner_set = {int(learner_seat_per_game[gid])}
        else:
            learner_set = default_learner_set

        # bids_made + avg_bid_reward: count learner bid decisions including pass.
        for seat in learner_set:
            bid_id = int(bid_contracts[gid][seat])
            bid_label = _contract_label(bid_id) if bid_id >= 0 else _PASS_LABEL
            bid_row = ensure(bid_label)
            bid_row["bids_made"] += 1
            bid_row["bid_reward_count"] += 1
            bid_row["bid_reward_sum"] += float(reward_scores[gid][seat])

        # bids_won/contracts_won: winner of bidding is declarer for the final contract.
        if declarer in learner_set:
            row = ensure(contract_label)
            row["bids_won"] += 1
            row["played_count"] += 1
            row["played_score_sum"] += float(scores[gid][declarer])
            row["played_reward_sum"] += float(reward_scores[gid][declarer])
            if int(scores[gid][declarer]) > 0:
                row["contracts_won"] += 1

        # Klop has no declarer in normal rules. Still report how often it was
        # played plus learner's average score/reward in those games.
        if contract_id == 0:
            for seat in learner_set:
                row = ensure(contract_label)
                row["played_count"] += 1
                row["played_score_sum"] += float(scores[gid][seat])
                row["played_reward_sum"] += float(reward_scores[gid][seat])

    # Keep output tidy and stable: hide contracts with no learner activity.
    return {
        k: v
        for k, v in stats.items()
        if int(v.get("bids_made", 0)) > 0 or int(v.get("played_count", 0)) > 0
    }
