"""Legal move generation engine — cascading filters + ban lists."""

from __future__ import annotations

import yaml
from pathlib import Path

from tarok.engine.core import Rule, BanRule, Trace, load_rules, load_ban_rules
from tarok.engine.conditions import (
    MoveContext,
    get_move_condition,
    get_move_filter,
    get_ban_filter,
)

_RULES_PATH = Path(__file__).parent / "rules" / "legal_moves_rules.yaml"
_rules: list[Rule] | None = None
_ban_rules: list[BanRule] | None = None


def _load() -> tuple[list[Rule], list[BanRule]]:
    global _rules, _ban_rules
    if _rules is None or _ban_rules is None:
        with open(_RULES_PATH) as f:
            data = yaml.safe_load(f)
        _rules = load_rules(data["rules"])
        _ban_rules = load_ban_rules(data.get("ban_rules", []))
    return _rules, _ban_rules


def reload_rules() -> None:
    """Force-reload rules from YAML (useful for testing)."""
    global _rules, _ban_rules
    _rules = None
    _ban_rules = None


def generate_legal_moves(ctx: MoveContext) -> Trace:
    """Generate the list of legal cards for the current player.

    Returns a Trace with:
        result: list[Card]  — playable cards
        triggered_rule: name of the pipeline rule that fired
        priority: priority of that rule
        context: includes any ban rules that also fired
    """
    rules, ban_rules = _load()

    if not ctx.hand:
        return Trace(result=[], triggered_rule="EmptyHand", priority=-1)

    # --- Cascading filter pipeline ---
    triggered_rule_name = ""
    triggered_priority = -1
    legal: list = []

    for rule in rules:
        cond_fn = get_move_condition(rule.condition)
        if cond_fn(ctx):
            filter_fn = get_move_filter(rule.action)
            result = filter_fn(ctx)
            if result:
                legal = result
                triggered_rule_name = rule.name
                triggered_priority = rule.priority
                break

    if not legal:
        # Should never happen if rules config has an "always/play_anything" fallback
        legal = list(ctx.hand)
        triggered_rule_name = "ImplicitFallback"
        triggered_priority = -1

    # --- Ban list pass ---
    applied_bans: list[str] = []
    for ban in ban_rules:
        cond_fn = get_move_condition(ban.condition)
        if cond_fn(ctx):
            ban_fn = get_ban_filter(ban.filter)
            filtered = ban_fn(ctx, legal)
            if len(filtered) < len(legal):
                applied_bans.append(ban.name)
                legal = filtered

    return Trace(
        result=legal,
        triggered_rule=triggered_rule_name,
        priority=triggered_priority,
        context={"applied_bans": applied_bans} if applied_bans else {},
    )
