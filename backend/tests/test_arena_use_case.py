"""Tests for arena use-case helpers."""

from tarok.use_cases.arena import agent_type_to_seat_label


def test_agent_type_to_seat_label_maps_pozrl_aliases() -> None:
    assert agent_type_to_seat_label("stockskis_pozrl") == "bot_pozrl"
    assert agent_type_to_seat_label("bot_pozrl") == "bot_pozrl"
    assert agent_type_to_seat_label("pozrl") == "bot_pozrl"


def test_agent_type_to_seat_label_maps_centaur() -> None:
    # Bot Arena needs to route NN-plus-endgame-solver agents through the
    # Rust "centaur" seat label so that the centaur bidding heuristic
    # (guaranteed-points solo gate etc.) actually fires. Without this
    # mapping, "rl" would be the only NN option and would invoke the raw
    # NN bid head, which during duplicate-centaur training is never
    # PPO-updated and so produces near-uniform, chaotic solo bids.
    assert agent_type_to_seat_label("centaur") == "centaur"
    assert agent_type_to_seat_label("stockskis_centaur") == "centaur"


def test_agent_type_to_seat_label_rejects_unknown() -> None:
    assert agent_type_to_seat_label("not-a-real-agent") is None


def test_agent_type_to_seat_label_maps_v3_3p() -> None:
    """3-player Tarok needs the bot_v3_3p seat label so the arena can
    populate non-NN seats with a 3p-rules-aware heuristic bot."""
    assert agent_type_to_seat_label("stockskis_v3_3p") == "bot_v3_3p"
    assert agent_type_to_seat_label("bot_v3_3p") == "bot_v3_3p"
    assert agent_type_to_seat_label("v3_3p") == "bot_v3_3p"


def test_agent_type_to_seat_label_maps_m6_3p() -> None:
    """3-player arena should allow selecting the m6-based 3p heuristic."""
    assert agent_type_to_seat_label("stockskis_m6_3p") == "bot_m6_3p"
    assert agent_type_to_seat_label("bot_m6_3p") == "bot_m6_3p"
    assert agent_type_to_seat_label("m6_3p") == "bot_m6_3p"
