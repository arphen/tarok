"""GPU (CUDA / MPS) compute backend — accelerated inference and training.

Applies device-specific optimisations transparent to callers:
* Automatic mixed precision (float16) for inference
* torch.compile for the forward path
* Non-blocking transfers for CPU → GPU copies
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training_lab.entities.encoding import DecisionType
from training_lab.ports.compute_backend import ComputeBackendPort


class GpuBackend(ComputeBackendPort):
    """CUDA / MPS accelerated inference and training."""

    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps is not available")

        self._device = torch.device(device)
        self._is_cuda = self._device.type == "cuda"
        self._is_mps = self._device.type == "mps"

        self._use_amp = self._is_cuda
        self._use_compile = (
            self._is_cuda
            and hasattr(torch, "compile")
            and torch.cuda.get_device_capability(self._device)[0] >= 7
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def name(self) -> str:
        return str(self._device)

    def prepare_network(self, network: nn.Module) -> nn.Module:
        network = network.to(self._device)
        if self._use_compile:
            try:
                network = torch.compile(network, mode="reduce-overhead")
            except Exception:
                pass
        return network

    def forward_batch(
        self,
        network: nn.Module,
        states: torch.Tensor,
        decision_types: list[DecisionType],
        oracle_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states_dev = states.to(self._device, non_blocking=self._is_cuda)
        oracle_dev = (
            oracle_states.to(self._device, non_blocking=self._is_cuda)
            if oracle_states is not None
            else None
        )

        with torch.no_grad():
            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    logits, values = network.forward_batch(
                        states_dev, decision_types, oracle_dev,
                    )
            else:
                logits, values = network.forward_batch(
                    states_dev, decision_types, oracle_dev,
                )

        return logits.cpu(), values.cpu()

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device, non_blocking=self._is_cuda)

    def stack_to_device(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(tensors)
        return stacked.to(self._device, non_blocking=self._is_cuda)

    def tensor_to_device(
        self,
        data: list,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        t = torch.tensor(data, dtype=dtype)
        return t.to(self._device, non_blocking=self._is_cuda)
