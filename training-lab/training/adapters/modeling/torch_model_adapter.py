"""Adapter: PyTorch model persistence + TorchScript export."""

from __future__ import annotations

import gc
import shutil
import warnings
import zipfile
from typing import Any

import torch

from tarok_model.network import TarokNetV4, TarokNetV5
from tarok_model.network_3p import TarokNet3

from training.ports import ModelPort


_SUPPORTED_ARCHES = ("v4", "v5", "v3p")


def build_model(hidden_size: int, oracle: bool, model_arch: str):
    """Construct a Tarok network for the given architecture.

    Exposed as module-level so other adapters (e.g. ``PPOAdapter``) can
    receive it via constructor injection, keeping network-class branching
    in one place.
    """
    if model_arch == "v4":
        return TarokNetV4(hidden_size=hidden_size, oracle_critic=oracle)
    if model_arch == "v5":
        if oracle:
            raise ValueError("TarokNetV5 is actor-only; oracle_critic is not supported.")
        return TarokNetV5(hidden_size=hidden_size, oracle_critic=False)
    if model_arch == "v3p":
        return TarokNet3(hidden_size=hidden_size, oracle_critic=oracle)
    raise ValueError(
        f"Unsupported model_arch={model_arch}. Supported: {_SUPPORTED_ARCHES}."
    )


# Backwards-compat alias for internal callers in this module.
_build_model = build_model


class TorchModelAdapter(ModelPort):
    def load_weights(self, checkpoint_path: str) -> tuple[dict, int, bool, str]:
        cp = _load_checkpoint_payload(checkpoint_path)
        if isinstance(cp, dict):
            sd = cp.get("model_state_dict", cp)
            model_arch = cp.get("model_arch")
        else:
            sd = _unwrap_exported_state_dict(cp.state_dict())
            model_arch = "v4"

        hidden_size = sd["shared.0.weight"].shape[0]
        oracle = any(k.startswith("critic_backbone") for k in sd)
        if model_arch not in _SUPPORTED_ARCHES:
            raise ValueError(
                f"Unsupported checkpoint architecture '{model_arch}'. "
                f"Supported: {_SUPPORTED_ARCHES}."
            )
        return sd, hidden_size, oracle, model_arch

    def create_new(self, hidden_size: int, oracle: bool, model_arch: str) -> dict:
        model = _build_model(hidden_size, oracle, model_arch)
        state = model.state_dict()
        del model
        _cleanup_torch_native_memory()
        return state

    def export_for_inference(self, weights: dict, hidden_size: int, oracle: bool, model_arch: str, path: str) -> None:
        model = _build_model(hidden_size, oracle, model_arch)
        model.load_state_dict(weights, strict=False)
        model.eval()
        try:
            if model_arch == "v3p":
                _export_torchscript_v3p(model, path)
            else:
                _export_torchscript(model, path)
        finally:
            del model
            _cleanup_torch_native_memory()

    def save_checkpoint(
        self, weights: dict, hidden_size: int, oracle: bool, model_arch: str,
        iteration: int, loss: float, placement: float, path: str,
    ) -> None:
        try:
            torch.save({
                "model_state_dict": weights,
                "hidden_size": hidden_size,
                "oracle_critic": oracle,
                "model_arch": model_arch,
                "iteration": iteration,
                "loss": loss,
                "placement": placement,
            }, path)
        finally:
            _cleanup_torch_native_memory()

    def copy_best(self, src: str, dst: str) -> None:
        shutil.copy2(src, dst)


def _export_torchscript(model: TarokNetV4, path: str) -> None:
    class _Wrapper(torch.nn.Module):
        def __init__(self, base: TarokNetV4):
            super().__init__()
            self.base = base
            # Actor-only variants (v5) have no critic head — the exported
            # tuple still needs a value slot, so we emit zeros.
            self.has_critic = hasattr(base, "critic")

        def forward(self, x: torch.Tensor):
            s = self.base.shared(x)
            s = self.base.res_blocks(s)
            cf = self.base._extract_card_features(x)
            a = self.base.card_attention(cf)
            f = self.base.fuse(torch.cat([s, a], dim=-1))
            if self.has_critic:
                value = self.base.critic(f).squeeze(-1)
            else:
                value = torch.zeros(f.shape[0], dtype=f.dtype, device=f.device)
            return (
                self.base.bid_head(f),
                self.base.king_head(f),
                self.base.talon_head(f),
                self.base.card_logits_for_export(f, x),
                value,
            )

    from tarok_model.encoding import STATE_SIZE

    w = _Wrapper(model)
    w.eval()
    example = torch.randn(1, STATE_SIZE)
    traced = None
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(w, example, check_trace=False)
        traced.save(path)
    finally:
        del traced
        del example
        del w
        _cleanup_torch_native_memory()


def _export_torchscript_v3p(model: TarokNet3, path: str) -> None:
    """Export a TarokNet3 to TorchScript matching the Rust 4p superset shapes.

    The Rust `NeuralNetPlayer` is variant-agnostic: it expects a 5-tuple
    of `(bid[B,9], king[B,4], talon[B,6], card[B,54], value[B])`. The 3p
    network's bid head only outputs 8 logits, so we right-pad with a
    large-negative constant; the king head doesn't exist in 3p, so we
    emit zeros (the Rust side never reads king for 3p decisions).
    """
    bid_pad_neg = -1e9

    class _AllHeads3P(torch.nn.Module):
        def __init__(self, base: TarokNet3):
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor):
            f = self.base.res_blocks(self.base.shared(x))
            bid8 = self.base.bid_head(f)
            pad = torch.full(
                (bid8.shape[0], 1), bid_pad_neg,
                dtype=bid8.dtype, device=bid8.device,
            )
            bid9 = torch.cat([bid8, pad], dim=-1)
            king = torch.zeros(
                f.shape[0], 4, dtype=f.dtype, device=f.device,
            )
            talon = self.base.talon_head(f)
            card = self.base.card_head(f)
            value = self.base.critic(f).squeeze(-1)
            return (bid9, king, talon, card, value)

    w = _AllHeads3P(model)
    w.eval()
    example = torch.randn(1, model.state_size)
    traced = None
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(w, example, check_trace=False)
        traced.save(path)
    finally:
        del traced
        del example
        del w
        _cleanup_torch_native_memory()


def _cleanup_torch_native_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _load_checkpoint_payload(checkpoint_path: str) -> dict[str, Any] | torch.jit.RecursiveScriptModule:
    if _is_torchscript_archive(checkpoint_path):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.jit\.load` is deprecated\. Please switch to `torch\.export`\.",
                category=DeprecationWarning,
            )
            return torch.jit.load(checkpoint_path, map_location="cpu")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _unwrap_exported_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if "shared.0.weight" in state_dict:
        return state_dict
    if any(key.startswith("base.") for key in state_dict):
        return {
            key.removeprefix("base."): value
            for key, value in state_dict.items()
            if key.startswith("base.")
        }
    return state_dict


def _is_torchscript_archive(checkpoint_path: str) -> bool:
    if not zipfile.is_zipfile(checkpoint_path):
        return False

    with zipfile.ZipFile(checkpoint_path) as archive:
        names = archive.namelist()

    return any(name.endswith("constants.pkl") or "/code/" in f"/{name}" for name in names)
