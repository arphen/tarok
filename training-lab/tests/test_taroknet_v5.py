"""Tests for TarokNetV5 (actor-only variant; docs/double_rl.md §2.7)."""

from __future__ import annotations

import pytest
import torch

from tarok_model.encoding import CARD_ACTION_SIZE, STATE_SIZE, DecisionType, GameMode
from tarok_model.network import TarokNetV4, TarokNetV5


def _state(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, STATE_SIZE)


def test_v5_has_no_critic_head_and_no_oracle_backbone() -> None:
    net = TarokNetV5(hidden_size=64)
    assert not hasattr(net, "critic")
    assert not hasattr(net, "critic_backbone")
    assert not hasattr(net, "critic_res_blocks")
    assert net.oracle_critic_enabled is False


def test_v5_rejects_oracle_critic_true() -> None:
    with pytest.raises(ValueError, match="actor-only"):
        TarokNetV5(hidden_size=64, oracle_critic=True)


def test_v5_forward_returns_zero_values() -> None:
    net = TarokNetV5(hidden_size=64)
    net.eval()
    with torch.no_grad():
        logits, value = net(_state(3), decision_type=DecisionType.CARD_PLAY, game_mode=GameMode.PARTNER_PLAY)
    assert logits.shape == (3, CARD_ACTION_SIZE)
    assert value.shape == (3, 1)
    assert torch.all(value == 0.0)


def test_v5_forward_batch_returns_zero_values() -> None:
    net = TarokNetV5(hidden_size=64)
    net.eval()
    states = _state(4)
    dts = [DecisionType.CARD_PLAY, DecisionType.BID, DecisionType.CARD_PLAY, DecisionType.TALON_PICK]
    gms = [GameMode.PARTNER_PLAY, GameMode.PARTNER_PLAY, GameMode.SOLO, GameMode.PARTNER_PLAY]
    with torch.no_grad():
        logits, values = net.forward_batch(states, dts, game_modes=gms)
    assert logits.shape == (4, CARD_ACTION_SIZE)
    assert values.shape == (4,)
    assert torch.all(values == 0.0)


def test_v5_get_critic_features_raises() -> None:
    net = TarokNetV5(hidden_size=64)
    with pytest.raises(RuntimeError, match="actor-only"):
        net.get_critic_features(torch.randn(1, 10))


def test_v5_has_strictly_fewer_params_than_v4_with_oracle() -> None:
    v4 = TarokNetV4(hidden_size=128, oracle_critic=True)
    v5 = TarokNetV5(hidden_size=128, oracle_critic=False)
    v4_params = sum(p.numel() for p in v4.parameters())
    v5_params = sum(p.numel() for p in v5.parameters())
    # V5 drops oracle backbone + critic head → noticeably smaller.
    assert v5_params < v4_params
    shrink = 1.0 - v5_params / v4_params
    assert shrink > 0.15, f"Expected >=15% shrink vs v4-oracle, got {shrink:.1%}"


def test_v5_checkpoint_loads_into_v4_with_strict_false() -> None:
    """v5 state_dict is a subset of v4's — v4 can load it via strict=False."""
    v5 = TarokNetV5(hidden_size=64)
    v4 = TarokNetV4(hidden_size=64, oracle_critic=False)
    missing, unexpected = v4.load_state_dict(v5.state_dict(), strict=False)
    # v4 has a critic head that v5 doesn't → those keys are missing.
    assert any("critic" in key for key in missing)
    # v5 has no keys v4 doesn't recognise.
    assert unexpected == []
