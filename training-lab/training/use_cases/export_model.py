"""Use case: export model weights to a TorchScript inference file."""

from __future__ import annotations

from typing import Any

from training.entities.model_identity import ModelIdentity
from training.ports.model_port import ModelPort


class ExportModel:
    """Write the updated weights to the TorchScript file used by the engine.

    Single responsibility: translate from in-memory model weights to the
    on-disk inference artefact that self-play and benchmarking will read.
    """

    def __init__(self, model: ModelPort) -> None:
        self._model = model

    def execute(self, new_weights: Any, identity: ModelIdentity, ts_path: str) -> None:
        self._model.export_for_inference(
            new_weights,
            identity.hidden_size,
            identity.oracle_critic,
            identity.model_arch,
            ts_path,
        )
