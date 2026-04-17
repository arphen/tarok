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


def name_from_checkpoint(path: str) -> str | None:
    p = Path(path)
    parent = p.parent.name
    if parent and parent not in ("checkpoints", "training_run", "pinned", "hall_of_fame", "."):
        return parent
    m = re.match(r"hof_([A-Z]\w+_[A-Z][a-z]+)", p.stem)
    return m.group(1) if m else None
