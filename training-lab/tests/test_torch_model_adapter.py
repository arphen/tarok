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