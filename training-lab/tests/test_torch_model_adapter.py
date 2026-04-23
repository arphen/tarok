"""Regression tests for TorchModelAdapter checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.adapters.modeling.torch_model_adapter import TorchModelAdapter


def test_load_weights_supports_saved_checkpoint_dict(tmp_path: Path) -> None:
    adapter = TorchModelAdapter()
    weights = adapter.create_new(hidden_size=64, oracle=True, model_arch="v4")
    checkpoint_path = tmp_path / "iter_001.pt"

    adapter.save_checkpoint(
        weights=weights,
        hidden_size=64,
        oracle=True,
        model_arch="v4",
        iteration=1,
        loss=0.5,
        placement=2.0,
        path=str(checkpoint_path),
    )

    loaded_weights, hidden_size, oracle, model_arch = adapter.load_weights(str(checkpoint_path))

    assert hidden_size == 64
    assert oracle is True
    assert model_arch == "v4"
    assert loaded_weights.keys() == weights.keys()


def test_load_weights_supports_torchscript_inference_artifact(tmp_path: Path) -> None:
    adapter = TorchModelAdapter()
    weights = adapter.create_new(hidden_size=64, oracle=False, model_arch="v4")
    script_path = tmp_path / "_current.pt"

    adapter.export_for_inference(
        weights=weights,
        hidden_size=64,
        oracle=False,
        model_arch="v4",
        path=str(script_path),
    )

    loaded_weights, hidden_size, oracle, model_arch = adapter.load_weights(str(script_path))

    assert hidden_size == 64
    assert oracle is False
    assert model_arch == "v4"
    assert loaded_weights.keys() == weights.keys()


def test_v5_checkpoint_round_trips_through_adapter(tmp_path: Path) -> None:
    adapter = TorchModelAdapter()
    weights = adapter.create_new(hidden_size=64, oracle=False, model_arch="v5")
    # v5 has no critic head nor oracle backbone keys.
    assert not any("critic" in k for k in weights)

    checkpoint_path = tmp_path / "iter_001.pt"
    adapter.save_checkpoint(
        weights=weights,
        hidden_size=64,
        oracle=False,
        model_arch="v5",
        iteration=1,
        loss=0.5,
        placement=2.0,
        path=str(checkpoint_path),
    )
    loaded, hs, oracle, arch = adapter.load_weights(str(checkpoint_path))
    assert arch == "v5"
    assert hs == 64
    assert oracle is False
    assert loaded.keys() == weights.keys()


def test_v5_export_for_inference_produces_torchscript(tmp_path: Path) -> None:
    adapter = TorchModelAdapter()
    weights = adapter.create_new(hidden_size=64, oracle=False, model_arch="v5")
    script_path = tmp_path / "_current.pt"
    adapter.export_for_inference(
        weights=weights,
        hidden_size=64,
        oracle=False,
        model_arch="v5",
        path=str(script_path),
    )
    # Reload — exported artifact's state_dict round-trips cleanly.
    loaded, hs, oracle, arch = adapter.load_weights(str(script_path))
    assert hs == 64
    assert oracle is False
    # Exported TorchScript artifact uses the fallback model_arch="v4" tag
    # because it has no metadata carrier — this is the existing behaviour
    # for v4 exports too. We still verify the state_dict is v5-shaped.
    assert not any("critic" in k for k in loaded)


def test_v5_rejects_oracle_true() -> None:
    adapter = TorchModelAdapter()
    with pytest.raises(ValueError, match="actor-only"):
        adapter.create_new(hidden_size=64, oracle=True, model_arch="v5")