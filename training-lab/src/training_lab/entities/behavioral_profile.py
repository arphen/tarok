"""Behavioral profiles — evolved personality traits that bias agent policy.

Each trait is a float that shifts the network's raw logits before action
selection.  This lets evolution evolve *how* an agent plays (aggressive
bidder, cautious defender, bold announcer) independently from the learned
weights.

Traits are applied as additive biases on logits, scaled so that ±1 is a
meaningful but not overwhelming shift (logits typically live in [-3, 3]).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from training_lab.entities.encoding import (
    DecisionType,
    BID_ACTION_SIZE,
    CARD_ACTION_SIZE,
    ANNOUNCE_PASS,
    ANNOUNCE_TRULA,
    ANNOUNCE_KINGS,
    ANNOUNCE_PAGAT,
    ANNOUNCE_VALAT,
    KONTRA_GAME,
    KONTRA_TRULA,
    KONTRA_KINGS,
    KONTRA_PAGAT,
    KONTRA_VALAT,
)


# Logit bias scale factor
_BIAS_SCALE = 1.5

# Solo contract bid indices (4: solo_three at idx 5, solo_two at idx 6, solo_one at idx 7, solo at idx 8)
_SOLO_BID_INDICES = [5, 6, 7, 8]
# Non-pass bid indices (all except index 0)
_NONPASS_BID_INDICES = list(range(1, BID_ACTION_SIZE))

# Tarok card indices in deck (first 22 cards: pagat through mond + škis)
_TAROK_CARD_INDICES = list(range(22))

# Announcement action indices (non-pass, non-kontra)
_ANNOUNCE_INDICES = [ANNOUNCE_TRULA, ANNOUNCE_KINGS, ANNOUNCE_PAGAT, ANNOUNCE_VALAT]
# Kontra action indices
_KONTRA_INDICES = [KONTRA_GAME, KONTRA_TRULA, KONTRA_KINGS, KONTRA_PAGAT, KONTRA_VALAT]


@dataclass
class BehavioralProfile:
    """Evolved personality traits that bias agent decision-making.

    Each trait ranges from -1 to 1. Positive values encourage the behavior,
    negative values discourage it.  The special ``temperature`` trait controls
    exploration (higher = more random, lower = more greedy).
    """

    # --- Bidding traits ---
    bid_aggression: float = 0.0
    solo_propensity: float = 0.0

    # --- Card play traits ---
    trump_eagerness: float = 0.0
    defensive_caution: float = 0.0

    # --- Announcement traits ---
    announce_boldness: float = 0.0
    kontra_aggression: float = 0.0

    # --- Exploration traits ---
    temperature: float = 1.0
    explore_decay: float = 0.995
    explore_floor: float = 0.02

    def to_dict(self) -> dict[str, float]:
        return {
            "bid_aggression": round(self.bid_aggression, 4),
            "solo_propensity": round(self.solo_propensity, 4),
            "trump_eagerness": round(self.trump_eagerness, 4),
            "defensive_caution": round(self.defensive_caution, 4),
            "announce_boldness": round(self.announce_boldness, 4),
            "kontra_aggression": round(self.kontra_aggression, 4),
            "temperature": round(self.temperature, 4),
            "explore_decay": round(self.explore_decay, 5),
            "explore_floor": round(self.explore_floor, 4),
        }

    @staticmethod
    def from_dict(d: dict[str, float]) -> BehavioralProfile:
        return BehavioralProfile(**{k: v for k, v in d.items() if k in BehavioralProfile.__dataclass_fields__})

    @staticmethod
    def random(rng: random.Random | None = None) -> BehavioralProfile:
        """Generate a random behavioral profile."""
        r = rng or random.Random()
        return BehavioralProfile(
            bid_aggression=r.uniform(-1, 1),
            solo_propensity=r.uniform(-1, 1),
            trump_eagerness=r.uniform(-1, 1),
            defensive_caution=r.uniform(-1, 1),
            announce_boldness=r.uniform(-1, 1),
            kontra_aggression=r.uniform(-1, 1),
            temperature=r.uniform(0.3, 3.0),
            explore_decay=r.uniform(0.98, 0.999),
            explore_floor=r.uniform(0.0, 0.1),
        )

    def to_genes(self) -> list[float]:
        """Flatten to a gene list."""
        return [
            self.bid_aggression,
            self.solo_propensity,
            self.trump_eagerness,
            self.defensive_caution,
            self.announce_boldness,
            self.kontra_aggression,
            self.temperature,
            self.explore_decay,
            self.explore_floor,
        ]

    @staticmethod
    def from_genes(genes: list[float]) -> BehavioralProfile:
        """Reconstruct from a gene list."""
        return BehavioralProfile(
            bid_aggression=max(-1, min(1, genes[0])),
            solo_propensity=max(-1, min(1, genes[1])),
            trump_eagerness=max(-1, min(1, genes[2])),
            defensive_caution=max(-1, min(1, genes[3])),
            announce_boldness=max(-1, min(1, genes[4])),
            kontra_aggression=max(-1, min(1, genes[5])),
            temperature=max(0.3, min(3.0, genes[6])),
            explore_decay=max(0.98, min(0.999, genes[7])),
            explore_floor=max(0.0, min(0.1, genes[8])),
        )


# Gene bounds for mutation clipping
GENE_BOUNDS: list[tuple[float, float]] = [
    (-1.0, 1.0),    # bid_aggression
    (-1.0, 1.0),    # solo_propensity
    (-1.0, 1.0),    # trump_eagerness
    (-1.0, 1.0),    # defensive_caution
    (-1.0, 1.0),    # announce_boldness
    (-1.0, 1.0),    # kontra_aggression
    (0.3, 3.0),     # temperature
    (0.98, 0.999),  # explore_decay
    (0.0, 0.1),     # explore_floor
]

GENE_SIGMAS: list[float] = [
    0.3,    # bid_aggression
    0.3,    # solo_propensity
    0.3,    # trump_eagerness
    0.3,    # defensive_caution
    0.3,    # announce_boldness
    0.3,    # kontra_aggression
    0.4,    # temperature
    0.003,  # explore_decay
    0.02,   # explore_floor
]

NUM_GENES = len(GENE_BOUNDS)


def apply_behavioral_bias(
    logits: torch.Tensor,
    profile: BehavioralProfile,
    decision_type: DecisionType,
    legal_mask: torch.Tensor,
    is_defender: bool = False,
) -> torch.Tensor:
    """Apply behavioral trait biases to raw network logits.

    Operates in-place on logits (batch or single). Biases are only applied
    to legal actions (respects the mask).
    """
    bias = torch.zeros_like(logits)

    if decision_type == DecisionType.BID:
        for idx in _NONPASS_BID_INDICES:
            bias[..., idx] += profile.bid_aggression * _BIAS_SCALE
        for idx in _SOLO_BID_INDICES:
            bias[..., idx] += profile.solo_propensity * _BIAS_SCALE * 0.7

    elif decision_type == DecisionType.CARD_PLAY:
        for idx in _TAROK_CARD_INDICES:
            bias[..., idx] += profile.trump_eagerness * _BIAS_SCALE * 0.5
        if is_defender and abs(profile.defensive_caution) > 0.01:
            for idx in range(22, CARD_ACTION_SIZE):
                norm_pos = (idx - 22) / (CARD_ACTION_SIZE - 23) * 2 - 1
                bias[..., idx] -= norm_pos * profile.defensive_caution * _BIAS_SCALE * 0.4

    elif decision_type == DecisionType.ANNOUNCE:
        for idx in _ANNOUNCE_INDICES:
            bias[..., idx] += profile.announce_boldness * _BIAS_SCALE * 0.8
        for idx in _KONTRA_INDICES:
            bias[..., idx] += profile.kontra_aggression * _BIAS_SCALE * 0.6

    biased = logits + bias * legal_mask.float()
    return biased


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature before softmax."""
    return logits / max(temperature, 0.01)
