"""Tests for checkpoint name resolution."""

from __future__ import annotations

from training.entities.names import _name_from_stem, name_from_checkpoint, random_slovenian_name


def test_name_from_checkpoint_uses_parent_for_named_run_dirs() -> None:
    assert name_from_checkpoint("data/checkpoints/Eva_Golob/iter_010.pt") == "Eva_Golob"


def test_name_from_checkpoint_derives_from_hall_of_fame_stem() -> None:
    assert (
        name_from_checkpoint("data/checkpoints/hall_of_fame/eva_golob.pt")
        == "Eva_Golob"
    )


def test_name_from_checkpoint_ignores_generic_checkpoint_stems() -> None:
    assert name_from_checkpoint("data/checkpoints/hall_of_fame/best.pt") is None
    assert name_from_checkpoint("data/checkpoints/hall_of_fame/_current.pt") is None
    assert name_from_checkpoint("data/checkpoints/hall_of_fame/iter_010.pt") is None


def test_name_from_stem_rejects_empty_or_iter_like_values() -> None:
    assert _name_from_stem("") is None
    assert _name_from_stem("   ") is None
    assert _name_from_stem("iter_010") is None


def test_random_slovenian_name_has_expected_format() -> None:
    value = random_slovenian_name()
    assert "_" in value
    first, last = value.split("_", maxsplit=1)
    assert first
    assert last
