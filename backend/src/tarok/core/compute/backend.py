"""ComputeBackend — abstract interface (port) for device-aware NN operations.

Every method that touches tensor placement, inference, or training-step
mechanics goes through this contract.  Implementations are free to add
device-specific optimisations (mixed precision, torch.compile, pinned
memory, CUDA streams, etc.) behind the same API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from tarok.core.encoding import DecisionType


class ComputeBackend(ABC):
    """Contract for device-aware neural-network operations."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The torch device managed by this backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g. ``"cpu"``, ``"cuda:0"``)."""

    # ------------------------------------------------------------------
    # Network lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_network(self, network: nn.Module) -> nn.Module:
        """Move *network* onto the backend device and apply any
        device-specific compilation / optimisation.

        Returns the (possibly wrapped) module.  Callers should use the
        returned reference from this point on.
        """

    # ------------------------------------------------------------------
    # Batched inference (self-play hot path)
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_batch(
        self,
        network: nn.Module,
        states: torch.Tensor,
        decision_types: list[DecisionType],
        oracle_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a batched forward pass and return ``(logits, values)``
        on **CPU** so callers never need to know the device.

        *states* and *oracle_states* arrive on CPU; the backend moves
        them to its device, runs inference, and moves results back.
        """

    # ------------------------------------------------------------------
    # Training step helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a single tensor to the backend device."""

    @abstractmethod
    def stack_to_device(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """``torch.stack`` a list of tensors and place the result on
        the backend device.  Implementations may use pinned memory or
        async transfers for better throughput.
        """

    @abstractmethod
    def tensor_to_device(
        self,
        data: list,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create a 1-D tensor from a plain Python list and place it
        on the backend device.
        """
