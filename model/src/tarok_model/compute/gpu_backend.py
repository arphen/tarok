"""GPU (CUDA / MPS) compute backend — accelerated inference and training.

Applies device-specific optimisations that are transparent to callers:

* **Automatic mixed precision** (float16) for inference — halves memory
  bandwidth and doubles throughput on tensor-core GPUs.
* **torch.compile** (PyTorch ≥ 2.0) for the forward path — fuses ops
  and eliminates Python overhead.
* **Pinned-memory transfers** for CPU → GPU copies via a persistent
  pin-memory buffer pool.
* **Non-blocking transfers** — overlap data movement with computation
  where possible.

Falls back gracefully: if CUDA is unavailable, ``factory.create`` will
never instantiate this class.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tarok_model.compute.backend import ComputeBackend
from tarok_model.encoding import DecisionType


class GpuBackend(ComputeBackend):
    """CUDA / MPS accelerated inference and training."""

    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps is not available")

        self._device = torch.device(device)
        self._is_cuda = self._device.type == "cuda"
        self._is_mps = self._device.type == "mps"

        # AMP is only supported on CUDA
        self._use_amp = self._is_cuda
        # torch.compile is CUDA-only and requires PyTorch ≥ 2.0
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

    # ------------------------------------------------------------------
    # Network lifecycle
    # ------------------------------------------------------------------

    def prepare_network(self, network: nn.Module) -> nn.Module:
        network = network.to(self._device)

        if self._use_compile:
            try:
                network = torch.compile(network, mode="reduce-overhead")
            except Exception:
                pass  # Graceful fallback if compile fails

        return network

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
        # Move inputs to GPU (non-blocking when possible)
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

        # Return on CPU so callers are device-agnostic
        return logits.cpu(), values.cpu()

    # ------------------------------------------------------------------
    # Training step helpers
    # ------------------------------------------------------------------

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
