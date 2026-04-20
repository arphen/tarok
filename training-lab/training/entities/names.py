"""Slovenian name generation and checkpoint name extraction."""

from __future__ import annotations

import random
import re
from pathlib import Path

_SL_FIRST = [
    "Ana", "Maja", "Eva", "Nina", "Sara", "Lara", "Anja", "Ema",
    "Katja", "Tina", "Živa", "Pia", "Lina", "Zala", "Neža",
    "Teja", "Rok", "Urša", "Janja", "Alja", "Špela", "Manca",
    "Petra", "Metka", "Monika", "Irena", "Andreja", "Brigita",
    "Vera", "Marta", "Klara", "Nataša", "Polona", "Mateja",
    "Lea", "Nika", "Hana", "Julija", "Lucija", "Tamara",
]
_SL_LAST = [
    "Novak", "Horvat", "Kovačič", "Krajnc", "Zupančič",
    "Potočnik", "Kos", "Golob", "Vidmar", "Kolar",
    "Mlakar", "Bizjak", "Žagar", "Turk", "Hribar",
    "Kavčič", "Hočevar", "Rupnik", "Debeljak", "Černe",
    "Gregorčič", "Vesel", "Kern", "Starič", "Oblak",
    "Pečnik", "Gorenc", "Šuštar", "Bogataj", "Kranjc",
]


def random_slovenian_name() -> str:
    return f"{random.choice(_SL_FIRST)}_{random.choice(_SL_LAST)}"


def _name_from_stem(stem: str) -> str | None:
    raw = stem.strip()
    if not raw:
        return None
    if raw in {"best", "_current", "current"}:
        return None
    if re.fullmatch(r"iter_\d+", raw):
        return None

    parts = [p for p in re.split(r"[_\-\s]+", raw) if p]
    if not parts:
        return None
    return "_".join(p[:1].upper() + p[1:] for p in parts)


def name_from_checkpoint(path: str) -> str | None:
    p = Path(path)
    parent = p.parent.name
    if parent and parent not in ("checkpoints", "training_run", "pinned", "hall_of_fame", "."):
        return parent

    stem_name = _name_from_stem(p.stem)
    if stem_name is not None:
        return stem_name

    m = re.match(r"hof_([A-Z]\w+_[A-Z][a-z]+)", p.stem)
    return m.group(1) if m else None
