"""CPU compute backend — baseline implementation.

No special optimisations; tensors stay on CPU.  This is the safe default
that works everywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tarok.adapters.ai.compute.backend import ComputeBackend
from tarok.adapters.ai.encoding import DecisionType


class CpuBackend(ComputeBackend):
    """Plain CPU inference and training."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def name(self) -> str:
        return "cpu"

    # ------------------------------------------------------------------
    # Network lifecycle
    # ------------------------------------------------------------------

    def prepare_network(self, network: nn.Module) -> nn.Module:
        return network.to(self.device)

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        network: nn.Module,
        states: torch.Tensor,
        decision_types: list[DecisionType],
        oracle_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Already on CPU — just delegate
        with torch.no_grad():
            logits, values = network.forward_batch(states, decision_types, oracle_states)
        return logits, values

    # ------------------------------------------------------------------
    # Training step helpers
    # ------------------------------------------------------------------

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor  # no-op on CPU

    def stack_to_device(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensors)

    def tensor_to_device(
        self,
        data: list,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype)
