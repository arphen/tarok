from __future__ import annotations

import torch

from tarok_model.encoding import GameMode, STATE_SIZE
from tarok_model.network import TarokNetV4


def _state_with_contract_idx(contract_idx: int) -> torch.Tensor:
    s = torch.zeros(STATE_SIZE, dtype=torch.float32)
    # Contract slice in encoding is [220:230]
    s[220 + contract_idx] = 1.0
    return s


def test_mode_from_contract_idx_expected_mapping() -> None:
    assert TarokNetV4._mode_from_contract_idx(TarokNetV4._KLOP_IDX) == GameMode.KLOP_BERAC
    assert TarokNetV4._mode_from_contract_idx(TarokNetV4._BERAC_IDX) == GameMode.KLOP_BERAC
    assert TarokNetV4._mode_from_contract_idx(TarokNetV4._BARVNI_VALAT_IDX) == GameMode.COLOR_VALAT
    assert TarokNetV4._mode_from_contract_idx(TarokNetV4._THREE_IDX) == GameMode.PARTNER_PLAY
    assert TarokNetV4._mode_from_contract_idx(TarokNetV4._ONE_IDX) == GameMode.PARTNER_PLAY
    assert TarokNetV4._mode_from_contract_idx(TarokNetV4._SOLO_IDX) == GameMode.SOLO


def test_infer_modes_from_state_uses_contract_slice_correctly() -> None:
    net = TarokNetV4(hidden_size=32, oracle_critic=False)
    states = torch.stack([
        _state_with_contract_idx(TarokNetV4._KLOP_IDX),
        _state_with_contract_idx(TarokNetV4._SOLO_THREE_IDX),
        _state_with_contract_idx(TarokNetV4._BERAC_IDX),
        _state_with_contract_idx(TarokNetV4._BARVNI_VALAT_IDX),
        torch.zeros(STATE_SIZE, dtype=torch.float32),  # no contract -> default partner
    ])

    modes = net._infer_modes_from_state(states)
    assert modes == [
        GameMode.KLOP_BERAC,
        GameMode.SOLO,
        GameMode.KLOP_BERAC,
        GameMode.COLOR_VALAT,
        GameMode.PARTNER_PLAY,
    ]


def test_python_contract_indices_match_rust_exported_constants() -> None:
    try:
        import tarok_engine as te
    except Exception:
        # Test environments without Rust extension can skip this contract check.
        return

    assert int(te.CONTRACT_KLOP) == TarokNetV4._KLOP_IDX
    assert int(te.CONTRACT_THREE) == TarokNetV4._THREE_IDX
    assert int(te.CONTRACT_TWO) == TarokNetV4._TWO_IDX
    assert int(te.CONTRACT_ONE) == TarokNetV4._ONE_IDX
    assert int(te.CONTRACT_SOLO_THREE) == TarokNetV4._SOLO_THREE_IDX
    assert int(te.CONTRACT_SOLO_TWO) == TarokNetV4._SOLO_TWO_IDX
    assert int(te.CONTRACT_SOLO_ONE) == TarokNetV4._SOLO_ONE_IDX
    assert int(te.CONTRACT_SOLO) == TarokNetV4._SOLO_IDX
    assert int(te.CONTRACT_BERAC) == TarokNetV4._BERAC_IDX
    assert int(te.CONTRACT_BARVNI_VALAT) == TarokNetV4._BARVNI_VALAT_IDX