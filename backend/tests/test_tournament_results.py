"""Tests for tournament result data integrity and loader."""

import json
import math
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tournament_results.json"


@pytest.fixture
def results():
    with open(DATA_PATH) as f:
        return json.load(f)


def test_data_file_exists():
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}"


def test_has_tournaments(results):
    assert "tournaments" in results
    assert len(results["tournaments"]) >= 1


@pytest.mark.parametrize("tid", [0, 1, 2])
def test_tournament_structure(results, tid):
    t = results["tournaments"][tid]
    for key in ("id", "num_rounds", "num_players", "standings"):
        assert key in t, f"Missing key {key} in tournament {tid}"
    assert t["num_rounds"] > 0
    assert t["num_players"] > 0


@pytest.mark.parametrize("tid", [0, 1, 2])
def test_standings_sorted_by_rank(results, tid):
    standings = results["tournaments"][tid]["standings"]
    ranks = [s["rank"] for s in standings]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize("tid", [0, 1, 2])
def test_avg_placement_matches_placements(results, tid):
    """avg_placement must equal the mean of the placements list."""
    for s in results["tournaments"][tid]["standings"]:
        expected = sum(s["placements"]) / len(s["placements"])
        assert math.isclose(s["avg_placement"], expected, abs_tol=0.01), (
            f"{s['model']}: avg_placement={s['avg_placement']} "
            f"but mean(placements)={expected:.2f}"
        )


@pytest.mark.parametrize("tid", [0, 1, 2])
def test_first_places_count(results, tid):
    """first_places must equal count of 1s in placements."""
    for s in results["tournaments"][tid]["standings"]:
        expected = s["placements"].count(1)
        assert s["first_places"] == expected, (
            f"{s['model']}: first_places={s['first_places']} but counted {expected}"
        )


@pytest.mark.parametrize("tid", [0, 1, 2])
def test_top2_count(results, tid):
    """top2 must equal count of placements <= 2."""
    for s in results["tournaments"][tid]["standings"]:
        expected = sum(1 for p in s["placements"] if p <= 2)
        assert s["top2"] == expected, (
            f"{s['model']}: top2={s['top2']} but counted {expected}"
        )


@pytest.mark.parametrize("tid", [0, 1, 2])
def test_top_half_count(results, tid):
    """top_half must equal count of placements <= 4."""
    for s in results["tournaments"][tid]["standings"]:
        expected = sum(1 for p in s["placements"] if p <= 4)
        assert s["top_half"] == expected, (
            f"{s['model']}: top_half={s['top_half']} but counted {expected}"
        )


@pytest.mark.parametrize("tid", [0, 1, 2])
def test_placements_length_matches_rounds(results, tid):
    t = results["tournaments"][tid]
    for s in t["standings"]:
        assert len(s["placements"]) == t["num_rounds"], (
            f"{s['model']}: {len(s['placements'])} placements but "
            f"{t['num_rounds']} rounds"
        )


@pytest.mark.parametrize("tid", [0, 1, 2])
def test_standings_sorted_by_avg(results, tid):
    standings = results["tournaments"][tid]["standings"]
    avgs = [s["avg_placement"] for s in standings]
    assert avgs == sorted(avgs)


def test_loader_top_models(results):
    """Loader returns models sorted by combined avg across tournaments they appear in."""
    from tarok.adapters.ai.tournament_results import top_models

    best = top_models(results, n=3)
    assert len(best) <= 3
    assert all(isinstance(m, str) for m in best)
