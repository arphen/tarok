"""Core abstractions for the Priority Rules Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Trace:
    """Audit trail returned alongside every engine result.

    Attributes:
        result: The computed output (winner index, card list, etc.)
        triggered_rule: Name of the rule that fired.
        priority: Priority value of the triggered rule.
        context: Optional dict of extra diagnostic info.
    """
    result: Any
    triggered_rule: str
    priority: int
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    """A single declarative rule loaded from YAML.

    Attributes:
        name: Human-readable rule identifier.
        priority: Higher = evaluated first.
        condition: Name of the registered condition function.
        action: Name of the registered action/filter/transform function.
        params: Extra static parameters from the YAML config.
    """
    name: str
    priority: int
    condition: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BanRule:
    """A negative filter that removes cards from the legal move list.

    Evaluated after the main cascading filter pipeline.
    """
    name: str
    priority: int
    condition: str
    filter: str
    params: dict[str, Any] = field(default_factory=dict)


def load_rules(raw: list[dict]) -> list[Rule]:
    """Parse a list of raw YAML dicts into sorted Rule objects (highest priority first)."""
    rules = []
    for entry in raw:
        rules.append(Rule(
            name=entry["name"],
            priority=entry["priority"],
            condition=entry["condition"],
            action=entry["action"],
            params=entry.get("params", {}),
        ))
    return sorted(rules, key=lambda r: r.priority, reverse=True)


def load_ban_rules(raw: list[dict]) -> list[BanRule]:
    """Parse a list of raw YAML dicts into sorted BanRule objects."""
    bans = []
    for entry in raw:
        bans.append(BanRule(
            name=entry["name"],
            priority=entry["priority"],
            condition=entry["condition"],
            filter=entry["filter"],
            params=entry.get("params", {}),
        ))
    return sorted(bans, key=lambda r: r.priority, reverse=True)
