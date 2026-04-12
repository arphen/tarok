"""CPU compute backend — baseline implementation.

No special optimisations; tensors stay on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training_lab.entities.encoding import DecisionType
from training_lab.ports.compute_backend import ComputeBackendPort


class CpuBackend(ComputeBackendPort):
    """Plain CPU inference and training."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def name(self) -> str:
        return "cpu"

    def prepare_network(self, network: nn.Module) -> nn.Module:
        return network.to(self.device)

    def forward_batch(
        self,
        network: nn.Module,
        states: torch.Tensor,
        decision_types: list[DecisionType],
        oracle_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, values = network.forward_batch(states, decision_types, oracle_states)
        return logits, values

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def stack_to_device(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensors)

    def tensor_to_device(
        self,
        data: list,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype)
