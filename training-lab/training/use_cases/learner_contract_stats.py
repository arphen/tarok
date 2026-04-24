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


def _contract_label(cid: int) -> str:
    return _CONTRACT_LABELS.get(cid, f"contract_{cid}")


def compute_learner_contract_stats(
    raw: dict[str, Any],
    learner_seats: list[int],
) -> dict[str, dict[str, int]]:
    """Compute quick learner diagnostics per contract.

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
    if contracts is None or declarers is None or bid_contracts is None or scores is None:
        return {}

    n_games = len(contracts)
    if n_games == 0:
        return {}

    stats: dict[str, dict[str, int]] = {}

    def ensure(label: str) -> dict[str, int]:
        row = stats.get(label)
        if row is None:
            row = {"bids_made": 0, "bids_won": 0, "contracts_won": 0}
            stats[label] = row
        return row

    learner_set = {int(s) for s in learner_seats}

    for gid in range(n_games):
        declarer = int(declarers[gid])
        contract_id = int(contracts[gid])
        contract_label = _contract_label(contract_id)

        # bids_made: whenever a learner seat made a non-pass bid.
        for seat in learner_set:
            bid_id = int(bid_contracts[gid][seat])
            if bid_id >= 0:
                ensure(_contract_label(bid_id))["bids_made"] += 1

        # bids_won/contracts_won: winner of bidding is declarer for the final contract.
        if declarer in learner_set:
            row = ensure(contract_label)
            row["bids_won"] += 1
            if int(scores[gid][declarer]) > 0:
                row["contracts_won"] += 1

    # Keep output tidy and stable: hide completely empty contracts.
    return {k: v for k, v in stats.items() if any(vv > 0 for vv in v.values())}
