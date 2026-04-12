"""ComputeBackendPort — abstract interface for device-aware NN operations.

Every method that touches tensor placement, inference, or training-step
mechanics goes through this contract.  Implementations are free to add
device-specific optimisations (mixed precision, torch.compile, pinned
memory, CUDA streams, etc.) behind the same API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from training_lab.entities.encoding import DecisionType


class ComputeBackendPort(ABC):
    """Contract for device-aware neural-network operations."""

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def prepare_network(self, network: nn.Module) -> nn.Module:
        """Move network onto the backend device and apply optimisations.

        Returns the (possibly wrapped) module.
        """

    @abstractmethod
    def forward_batch(
        self,
        network: nn.Module,
        states: torch.Tensor,
        decision_types: list[DecisionType],
        oracle_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a batched forward pass and return (logits, values) on CPU."""

    @abstractmethod
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a single tensor to the backend device."""

    @abstractmethod
    def stack_to_device(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """torch.stack a list of tensors and place on the backend device."""

    @abstractmethod
    def tensor_to_device(
        self,
        data: list,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create a 1-D tensor from a Python list and place on the backend device."""
