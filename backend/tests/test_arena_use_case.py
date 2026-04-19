"""Tests for arena use-case helpers."""

from tarok.use_cases.arena import agent_type_to_seat_label


def test_agent_type_to_seat_label_maps_pozrl_aliases() -> None:
    assert agent_type_to_seat_label("stockskis_pozrl") == "bot_pozrl"
    assert agent_type_to_seat_label("bot_pozrl") == "bot_pozrl"
    assert agent_type_to_seat_label("pozrl") == "bot_pozrl"
