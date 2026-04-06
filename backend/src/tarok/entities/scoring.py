"""Scoring rules for Slovenian Tarok.

Cards are counted in groups of 3: (sum of 3 cards) - 2.
Total game points = 70. Declarer team wins with > 35 points (i.e. ≥ 36).
Differences are always computed from 35.
"""

from __future__ import annotations

from tarok.entities.card import Card, PAGAT, MOND, SKIS, CardType, SuitRank
from tarok.entities.game_state import (
    Announcement,
    Contract,
    GameState,
    KontraLevel,
    Team,
    Trick,
)

TOTAL_GAME_POINTS = 70
POINT_HALF = 35  # Diff is computed from 35; winning means > 35 (≥ 36)


def compute_card_points(cards: list[Card]) -> int:
    """Count card points using the 'groups of 3' method.

    Standard Tarok counting:
      groups of 3 → sum(points) − 2
      leftover 2  → sum(points) − 1
      leftover 1  → point value as-is
    Total over all 54 cards = 70.
    """
    raw = sum(c.points for c in cards)
    n = len(cards)
    deduction = (n // 3) * 2
    if n % 3 == 2:
        deduction += 1
    return raw - deduction


def _collect_team_cards(tricks: list[Trick], state: GameState, team: Team) -> list[Card]:
    """Collect all cards won by a team."""
    cards: list[Card] = []
    for trick in tricks:
        winner = trick.winner()
        if state.get_team(winner) == team:
            cards.extend(c for _, c in trick.cards)
    return cards


def _has_trula(cards: list[Card]) -> bool:
    tarok_values = {c.value for c in cards if c.card_type == CardType.TAROK}
    return {PAGAT, MOND, SKIS}.issubset(tarok_values)


def _has_all_kings(cards: list[Card]) -> bool:
    kings = [c for c in cards if c.is_king]
    return len(kings) == 4


def _pagat_ultimo(tricks: list[Trick], team: Team, state: GameState) -> bool:
    """Did the team win the last trick with Pagat?"""
    if not tricks:
        return False
    last_trick = tricks[-1]
    winner = last_trick.winner()
    if state.get_team(winner) != team:
        return False
    # Check if Pagat was in the last trick played by the winning team
    for player, card in last_trick.cards:
        if card.card_type == CardType.TAROK and card.value == PAGAT:
            return state.get_team(player) == team
    return False


def _contract_multiplier(contract: Contract) -> int:
    """Base game value per the official table (docs/basics.md).

    tri=10, dva=20, ena=30
    solo tri=40, solo dva=50, solo ena=60, solo brez talona=80
    """
    return {
        Contract.KLOP: 0,
        Contract.THREE: 10,
        Contract.TWO: 20,
        Contract.ONE: 30,
        Contract.SOLO_THREE: 40,
        Contract.SOLO_TWO: 50,
        Contract.SOLO_ONE: 60,
        Contract.SOLO: 80,
        Contract.BERAC: 70,
        Contract.BARVNI_VALAT: 125,
    }[contract]


# --- Silent (tihi) bonus values ---
_SILENT_TRULA = 10
_SILENT_KINGS = 10
_SILENT_PAGAT_ULTIMO = 25
# --- Announced (napovedani) bonus values ---
_ANNOUNCED_TRULA = 20
_ANNOUNCED_KINGS = 20
_ANNOUNCED_PAGAT_ULTIMO = 50
_ANNOUNCED_VALAT = 500
_SILENT_VALAT = 250


def _score_klop(state: GameState) -> dict[int, int]:
    """Score a klop game (all players passed).

    Each player plays for themselves. Goal: avoid taking card points.
    - Player who captured > 35 card points: -70 (penalty for hoarding)
    - Player who took zero tricks: +70 (reward for clean play)
    - Everyone else: 0
    """
    player_cards: dict[int, list[Card]] = {p: [] for p in range(state.num_players)}
    player_trick_count: dict[int, int] = {p: 0 for p in range(state.num_players)}
    for trick in state.tricks:
        winner = trick.winner()
        player_cards[winner].extend(c for _, c in trick.cards)
        player_trick_count[winner] += 1

    scores: dict[int, int] = {p: 0 for p in range(state.num_players)}
    for p in range(state.num_players):
        card_points = compute_card_points(player_cards[p])
        if card_points > POINT_HALF:
            scores[p] = -TOTAL_GAME_POINTS  # -70
        elif player_trick_count[p] == 0:
            scores[p] = TOTAL_GAME_POINTS  # +70
    return scores


def _score_berac(state: GameState) -> dict[int, int]:
    """Score a berac game — declarer bids to take zero tricks.

    Berac is always solo (declarer vs 3 opponents).
    Win (+70 each opponent) if declarer took 0 tricks.
    Lose (-70 each opponent) if declarer took any trick.
    """
    assert state.declarer is not None
    declarer = state.declarer
    base = _contract_multiplier(Contract.BERAC)  # 70
    declarer_trick_count = sum(1 for t in state.tricks if t.winner() == declarer)

    if declarer_trick_count == 0:
        # Won: only declarer scores
        return {
            p: base if p == declarer else 0
            for p in range(state.num_players)
        }
    else:
        # Lost: only declarer scores (negative)
        return {
            p: -base if p == declarer else 0
            for p in range(state.num_players)
        }


def _get_kontra(state: GameState, key: str) -> int:
    """Return the kontra multiplier for a given target ('game' or announcement name)."""
    level = state.kontra_levels.get(key, KontraLevel.NONE)
    return level.value


def _score_barvni_valat(state: GameState) -> dict[int, int]:
    """Score a barvni valat (colour valat) game.

    Always solo (1v3). Declarer must take all 12 tricks.
    Suit cards beat taroks in this mode. Base value = 125.
    Win = +125 per opponent, Lose = -125 per opponent.
    Kontra/Re/Sub applies to the base.
    """
    assert state.declarer is not None
    declarer = state.declarer
    base = _contract_multiplier(Contract.BARVNI_VALAT)  # 125
    declarer_won_all = all(t.winner() == declarer for t in state.tricks)

    if not declarer_won_all:
        base = -base

    base *= _get_kontra(state, "game")

    return {
        p: base if p == declarer else 0
        for p in range(state.num_players)
    }


def score_game(state: GameState) -> dict[int, int]:
    """Compute final scores for all players. Returns player_idx -> point delta."""
    assert state.contract is not None

    if state.contract.is_klop:
        return _score_klop(state)

    if state.contract.is_berac:
        return _score_berac(state)

    if state.contract.is_barvni_valat:
        return _score_barvni_valat(state)

    assert state.declarer is not None

    all_tricks = state.tricks
    declarer_cards: list[Card] = []
    opponent_cards: list[Card] = []

    for trick in all_tricks:
        winner = trick.winner()
        team = state.get_team(winner)
        for _, card in trick.cards:
            if team == Team.DECLARER_TEAM:
                declarer_cards.append(card)
            else:
                opponent_cards.append(card)

    # Add put-down cards to declarer's pile
    declarer_cards.extend(state.put_down)

    declarer_points = compute_card_points(declarer_cards)
    declarer_won = declarer_points > POINT_HALF  # > 35 means ≥ 36

    # Base game score: contract value + |difference from 35|
    point_diff = abs(declarer_points - POINT_HALF)
    base_score = _contract_multiplier(state.contract) + point_diff

    if not declarer_won:
        base_score = -base_score

    # Apply kontra/re/sub to the base game
    base_score *= _get_kontra(state, "game")

    # --- Bonus scoring (silent & announced) ---
    bonus = 0

    # Collect which announcements were made and by which team
    announced_by_team: dict[Announcement, Team] = {}
    for player, announcements in state.announcements.items():
        team = state.get_team(player)
        for ann in announcements:
            announced_by_team[ann] = team

    # Trula
    decl_has_trula = _has_trula(declarer_cards)
    opp_has_trula = _has_trula(opponent_cards)
    trula_bonus = 0
    if Announcement.TRULA in announced_by_team:
        ann_team = announced_by_team[Announcement.TRULA]
        if ann_team == Team.DECLARER_TEAM:
            trula_bonus = _ANNOUNCED_TRULA if decl_has_trula else -_ANNOUNCED_TRULA
        else:
            trula_bonus = -_ANNOUNCED_TRULA if opp_has_trula else _ANNOUNCED_TRULA
        trula_bonus *= _get_kontra(state, Announcement.TRULA.value)
    else:
        if decl_has_trula:
            trula_bonus = _SILENT_TRULA
        elif opp_has_trula:
            trula_bonus = -_SILENT_TRULA
    bonus += trula_bonus

    # Kings
    decl_has_kings = _has_all_kings(declarer_cards)
    opp_has_kings = _has_all_kings(opponent_cards)
    kings_bonus = 0
    if Announcement.KINGS in announced_by_team:
        ann_team = announced_by_team[Announcement.KINGS]
        if ann_team == Team.DECLARER_TEAM:
            kings_bonus = _ANNOUNCED_KINGS if decl_has_kings else -_ANNOUNCED_KINGS
        else:
            kings_bonus = -_ANNOUNCED_KINGS if opp_has_kings else _ANNOUNCED_KINGS
        kings_bonus *= _get_kontra(state, Announcement.KINGS.value)
    else:
        if decl_has_kings:
            kings_bonus = _SILENT_KINGS
        elif opp_has_kings:
            kings_bonus = -_SILENT_KINGS
    bonus += kings_bonus

    # Pagat ultimo
    decl_pagat = _pagat_ultimo(all_tricks, Team.DECLARER_TEAM, state)
    opp_pagat = _pagat_ultimo(all_tricks, Team.OPPONENT_TEAM, state)
    pagat_bonus = 0
    if Announcement.PAGAT_ULTIMO in announced_by_team:
        ann_team = announced_by_team[Announcement.PAGAT_ULTIMO]
        if ann_team == Team.DECLARER_TEAM:
            pagat_bonus = _ANNOUNCED_PAGAT_ULTIMO if decl_pagat else -_ANNOUNCED_PAGAT_ULTIMO
        else:
            pagat_bonus = -_ANNOUNCED_PAGAT_ULTIMO if opp_pagat else _ANNOUNCED_PAGAT_ULTIMO
        pagat_bonus *= _get_kontra(state, Announcement.PAGAT_ULTIMO.value)
    else:
        if decl_pagat:
            pagat_bonus = _SILENT_PAGAT_ULTIMO
        elif opp_pagat:
            pagat_bonus = -_SILENT_PAGAT_ULTIMO
    bonus += pagat_bonus

    # Valat
    valat_bonus = 0
    if Announcement.VALAT in announced_by_team:
        ann_team = announced_by_team[Announcement.VALAT]
        all_won = all(state.get_team(t.winner()) == ann_team for t in all_tricks)
        if ann_team == Team.DECLARER_TEAM:
            valat_bonus = _ANNOUNCED_VALAT if all_won else -_ANNOUNCED_VALAT
        else:
            valat_bonus = -_ANNOUNCED_VALAT if all_won else _ANNOUNCED_VALAT
        valat_bonus *= _get_kontra(state, Announcement.VALAT.value)
    else:
        # Silent valat
        decl_all = all(state.get_team(t.winner()) == Team.DECLARER_TEAM for t in all_tricks)
        opp_all = all(state.get_team(t.winner()) == Team.OPPONENT_TEAM for t in all_tricks)
        if decl_all:
            valat_bonus = _SILENT_VALAT
        elif opp_all:
            valat_bonus = -_SILENT_VALAT
    bonus += valat_bonus

    total_declarer = base_score + bonus

    # Distribute scores — only declarer team scores, opponents get 0
    scores: dict[int, int] = {}
    for p in range(state.num_players):
        team = state.get_team(p)
        if team == Team.DECLARER_TEAM:
            scores[p] = total_declarer
        else:
            scores[p] = 0

    return scores


def score_game_breakdown(state: GameState) -> dict:
    """Compute final scores with a full justification breakdown.

    Returns a dict with:
      - scores: player_idx -> point delta
      - breakdown: human-readable explanation of the scoring
      - trick_summary: list of trick winners with cards
    """
    assert state.contract is not None

    if state.contract.is_klop:
        scores = _score_klop(state)
        trick_summary = _trick_summary(state)
        return {
            "scores": scores,
            "breakdown": {
                "contract": "Klop",
                "mode": "klop",
                "explanation": "All players passed. Each plays for themselves.",
                "lines": [
                    {"label": "Klop scoring", "detail": ">35 pts = -70, zero tricks = +70, else 0"}
                ],
            },
            "trick_summary": trick_summary,
        }

    if state.contract.is_berac:
        scores = _score_berac(state)
        trick_summary = _trick_summary(state)
        declarer_tricks = sum(1 for t in state.tricks if t.winner() == state.declarer)
        won = declarer_tricks == 0
        return {
            "scores": scores,
            "breakdown": {
                "contract": "Berac",
                "mode": "solo",
                "declarer_won": won,
                "explanation": f"Declarer bid zero tricks, took {declarer_tricks}.",
                "lines": [
                    {"label": "Base value", "value": 70},
                    {"label": "Result", "detail": "Won (0 tricks)" if won else f"Lost ({declarer_tricks} tricks)"},
                ],
            },
            "trick_summary": trick_summary,
        }

    if state.contract.is_barvni_valat:
        scores = _score_barvni_valat(state)
        trick_summary = _trick_summary(state)
        declarer_tricks = sum(1 for t in state.tricks if t.winner() == state.declarer)
        won = declarer_tricks == 12
        return {
            "scores": scores,
            "breakdown": {
                "contract": "Barvni Valat",
                "mode": "solo",
                "declarer_won": won,
                "explanation": f"Colour valat: suits beat taroks. Declarer took {declarer_tricks}/12 tricks.",
                "lines": [
                    {"label": "Base value", "value": 125},
                    {"label": "Result", "detail": "Won (all 12 tricks)" if won else f"Lost ({declarer_tricks} tricks)"},
                ],
            },
            "trick_summary": trick_summary,
        }

    # Normal contracts
    all_tricks = state.tricks
    declarer_cards: list[Card] = []
    opponent_cards: list[Card] = []

    for trick in all_tricks:
        winner = trick.winner()
        team = state.get_team(winner)
        for _, card in trick.cards:
            if team == Team.DECLARER_TEAM:
                declarer_cards.append(card)
            else:
                opponent_cards.append(card)

    declarer_cards_with_putdown = list(declarer_cards) + list(state.put_down)
    declarer_points = compute_card_points(declarer_cards_with_putdown)
    opponent_points = TOTAL_GAME_POINTS - declarer_points
    declarer_won = declarer_points > POINT_HALF

    point_diff = abs(declarer_points - POINT_HALF)
    contract_base = _contract_multiplier(state.contract)
    raw_base = contract_base + point_diff
    base_sign = 1 if declarer_won else -1
    kontra_game = _get_kontra(state, "game")
    base_score = base_sign * raw_base * kontra_game

    # Bonuses (mirror the logic in score_game)
    announced_by_team: dict[Announcement, Team] = {}
    for player, announcements in state.announcements.items():
        team = state.get_team(player)
        for ann in announcements:
            announced_by_team[ann] = team

    bonus_lines: list[dict] = []

    # Trula
    decl_has_trula = _has_trula(declarer_cards_with_putdown)
    opp_has_trula = _has_trula(opponent_cards)
    trula_bonus = 0
    if Announcement.TRULA in announced_by_team:
        ann_team = announced_by_team[Announcement.TRULA]
        collected = decl_has_trula if ann_team == Team.DECLARER_TEAM else opp_has_trula
        base_val = _ANNOUNCED_TRULA
        if ann_team == Team.DECLARER_TEAM:
            trula_bonus = base_val if collected else -base_val
        else:
            trula_bonus = -base_val if collected else base_val
        trula_bonus *= _get_kontra(state, Announcement.TRULA.value)
        bonus_lines.append({"label": "Trula (announced)", "value": trula_bonus})
    else:
        if decl_has_trula:
            trula_bonus = _SILENT_TRULA
            bonus_lines.append({"label": "Trula (silent)", "value": trula_bonus})
        elif opp_has_trula:
            trula_bonus = -_SILENT_TRULA
            bonus_lines.append({"label": "Trula (silent, opp)", "value": trula_bonus})

    # Kings
    decl_has_kings = _has_all_kings(declarer_cards_with_putdown)
    opp_has_kings = _has_all_kings(opponent_cards)
    kings_bonus = 0
    if Announcement.KINGS in announced_by_team:
        ann_team = announced_by_team[Announcement.KINGS]
        collected = decl_has_kings if ann_team == Team.DECLARER_TEAM else opp_has_kings
        base_val = _ANNOUNCED_KINGS
        if ann_team == Team.DECLARER_TEAM:
            kings_bonus = base_val if collected else -base_val
        else:
            kings_bonus = -base_val if collected else base_val
        kings_bonus *= _get_kontra(state, Announcement.KINGS.value)
        bonus_lines.append({"label": "Kings (announced)", "value": kings_bonus})
    else:
        if decl_has_kings:
            kings_bonus = _SILENT_KINGS
            bonus_lines.append({"label": "Kings (silent)", "value": kings_bonus})
        elif opp_has_kings:
            kings_bonus = -_SILENT_KINGS
            bonus_lines.append({"label": "Kings (silent, opp)", "value": kings_bonus})

    # Pagat ultimo
    decl_pagat = _pagat_ultimo(all_tricks, Team.DECLARER_TEAM, state)
    opp_pagat = _pagat_ultimo(all_tricks, Team.OPPONENT_TEAM, state)
    pagat_bonus = 0
    if Announcement.PAGAT_ULTIMO in announced_by_team:
        ann_team = announced_by_team[Announcement.PAGAT_ULTIMO]
        if ann_team == Team.DECLARER_TEAM:
            pagat_bonus = _ANNOUNCED_PAGAT_ULTIMO if decl_pagat else -_ANNOUNCED_PAGAT_ULTIMO
        else:
            pagat_bonus = -_ANNOUNCED_PAGAT_ULTIMO if opp_pagat else _ANNOUNCED_PAGAT_ULTIMO
        pagat_bonus *= _get_kontra(state, Announcement.PAGAT_ULTIMO.value)
        bonus_lines.append({"label": "Pagat ultimo (announced)", "value": pagat_bonus})
    else:
        if decl_pagat:
            pagat_bonus = _SILENT_PAGAT_ULTIMO
            bonus_lines.append({"label": "Pagat ultimo (silent)", "value": pagat_bonus})
        elif opp_pagat:
            pagat_bonus = -_SILENT_PAGAT_ULTIMO
            bonus_lines.append({"label": "Pagat ultimo (silent, opp)", "value": pagat_bonus})

    # Valat
    valat_bonus = 0
    if Announcement.VALAT in announced_by_team:
        ann_team = announced_by_team[Announcement.VALAT]
        all_won = all(state.get_team(t.winner()) == ann_team for t in all_tricks)
        if ann_team == Team.DECLARER_TEAM:
            valat_bonus = _ANNOUNCED_VALAT if all_won else -_ANNOUNCED_VALAT
        else:
            valat_bonus = -_ANNOUNCED_VALAT if all_won else _ANNOUNCED_VALAT
        valat_bonus *= _get_kontra(state, Announcement.VALAT.value)
        bonus_lines.append({"label": "Valat (announced)", "value": valat_bonus})
    else:
        decl_all = all(state.get_team(t.winner()) == Team.DECLARER_TEAM for t in all_tricks)
        opp_all = all(state.get_team(t.winner()) == Team.OPPONENT_TEAM for t in all_tricks)
        if decl_all:
            valat_bonus = _SILENT_VALAT
            bonus_lines.append({"label": "Valat (silent)", "value": valat_bonus})
        elif opp_all:
            valat_bonus = -_SILENT_VALAT
            bonus_lines.append({"label": "Valat (silent, opp)", "value": valat_bonus})

    total_bonus = trula_bonus + kings_bonus + pagat_bonus + valat_bonus
    total_declarer = base_score + total_bonus

    effectively_solo = state.contract.is_solo or state.partner is None
    mode = "solo" if effectively_solo else "2v2"

    scores: dict[int, int] = {}
    for p in range(state.num_players):
        team = state.get_team(p)
        if team == Team.DECLARER_TEAM:
            scores[p] = total_declarer
        else:
            scores[p] = 0

    # Build breakdown lines
    lines: list[dict] = [
        {"label": "Contract base value", "value": contract_base},
        {"label": f"Card points (declarer: {declarer_points}, opponents: {opponent_points})", "detail": f"{'Won' if declarer_won else 'Lost'} by {point_diff}"},
        {"label": "Game score (base + diff)", "value": base_sign * raw_base},
    ]
    if kontra_game > 1:
        kontra_name = {2: "Kontra", 4: "Re", 8: "Sub-kontra", 16: "Mort-kontra"}.get(kontra_game, f"×{kontra_game}")
        lines.append({"label": f"{kontra_name} on game (×{kontra_game})", "value": base_score})
    lines.extend(bonus_lines)
    if total_bonus != 0:
        lines.append({"label": "Total bonus", "value": total_bonus})
    lines.append({"label": "Total (declarer team)", "value": total_declarer})
    if effectively_solo:
        lines.append({"label": "Scoring mode", "detail": f"Solo — only declarer scores {total_declarer:+d}"})
    else:
        lines.append({"label": "Scoring mode", "detail": f"2v2 — each teammate gets {total_declarer:+d}, opponents get 0"})

    trick_summary = _trick_summary(state)

    return {
        "scores": scores,
        "breakdown": {
            "contract": state.contract.value,
            "mode": mode,
            "declarer_won": declarer_won,
            "declarer_points": declarer_points,
            "opponent_points": opponent_points,
            "lines": lines,
        },
        "trick_summary": trick_summary,
    }


def _trick_summary(state: GameState) -> list[dict]:
    """Build a per-trick summary with winner and card points."""
    summary = []
    for i, trick in enumerate(state.tricks):
        winner = trick.winner()
        cards = [(p, c.label, c.points) for p, c in trick.cards]
        pts = compute_card_points([c for _, c in trick.cards])
        summary.append({
            "trick_num": i + 1,
            "lead_player": trick.lead_player,
            "cards": [{"player": p, "label": lbl, "points": pt} for p, lbl, pt in cards],
            "winner": winner,
            "card_points": pts,
        })
    return summary
