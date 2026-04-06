"""Trick evaluation engine — priority-based match & transform."""

from __future__ import annotations

import yaml
from pathlib import Path

from tarok.engine.core import Rule, Trace, load_rules
from tarok.engine.conditions import (
    TrickContext,
    get_trick_condition,
    get_trick_transform,
)

_RULES_PATH = Path(__file__).parent / "rules" / "trick_eval_rules.yaml"
_rules: list[Rule] | None = None


def _get_rules() -> list[Rule]:
    global _rules
    if _rules is None:
        with open(_RULES_PATH) as f:
            data = yaml.safe_load(f)
        _rules = load_rules(data["rules"])
    return _rules


def reload_rules() -> None:
    """Force-reload rules from YAML (useful for testing)."""
    global _rules
    _rules = None


def evaluate_trick(ctx: TrickContext) -> Trace:
    """Evaluate a completed trick.

    Returns a Trace with:
        result: (winner_player_idx, trick_points)
        triggered_rule: name of the matching rule
        priority: priority of the matched rule
    """
    for rule in _get_rules():
        condition_fn = get_trick_condition(rule.condition)
        if condition_fn(ctx):
            transform_fn = get_trick_transform(rule.action)
            winner, points = transform_fn(ctx)
            return Trace(
                result=(winner, points),
                triggered_rule=rule.name,
                priority=rule.priority,
                context={"description": rule.params.get("description", "")},
            )

    raise RuntimeError("No trick evaluation rule matched — rules config is incomplete")
