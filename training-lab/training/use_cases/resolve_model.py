"""Use case: load or create a model."""

from __future__ import annotations

from training.entities.model_identity import ModelIdentity
from training.entities.names import name_from_checkpoint, random_slovenian_name
from training.ports.model_port import ModelPort


class ResolveModel:
    def __init__(self, model: ModelPort):
        self._model = model

    def from_checkpoint(self, path: str) -> tuple[ModelIdentity, dict]:
        weights, hidden, oracle = self._model.load_weights(path)
        name = name_from_checkpoint(path) or random_slovenian_name()
        identity = ModelIdentity(name=name, hidden_size=hidden, oracle_critic=oracle, is_new=False)
        return identity, weights

    def from_scratch(self, hidden_size: int = 256, oracle: bool = True) -> tuple[ModelIdentity, dict]:
        name = random_slovenian_name()
        weights = self._model.create_new(hidden_size, oracle)
        identity = ModelIdentity(name=name, hidden_size=hidden_size, oracle_critic=oracle, is_new=True)
        return identity, weights
