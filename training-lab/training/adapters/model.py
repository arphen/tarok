"""Adapter: PyTorch model persistence + TorchScript export."""

from __future__ import annotations

import shutil

import torch

from tarok_model.network import TarokNetV4

from training.ports import ModelPort


class TorchModelAdapter(ModelPort):
    def load_weights(self, checkpoint_path: str) -> tuple[dict, int, bool, str]:
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = cp.get("model_state_dict", cp)
        hidden_size = sd["shared.0.weight"].shape[0]
        oracle = any(k.startswith("critic_backbone") for k in sd)
        model_arch = cp.get("model_arch")
        if model_arch != "v4":
            raise ValueError(
                f"Unsupported checkpoint architecture '{model_arch}'. Only 'v4' checkpoints are supported."
            )
        return sd, hidden_size, oracle, model_arch

    def create_new(self, hidden_size: int, oracle: bool, model_arch: str) -> dict:
        if model_arch != "v4":
            raise ValueError(f"Unsupported model_arch={model_arch}. Only 'v4' is supported.")
        model = TarokNetV4(hidden_size=hidden_size, oracle_critic=oracle)
        return model.state_dict()

    def export_for_inference(self, weights: dict, hidden_size: int, oracle: bool, model_arch: str, path: str) -> None:
        if model_arch != "v4":
            raise ValueError(f"Unsupported model_arch={model_arch}. Only 'v4' is supported.")
        model = TarokNetV4(hidden_size=hidden_size, oracle_critic=oracle)
        model.load_state_dict(weights)
        model.eval()
        _export_torchscript(model, path)

    def save_checkpoint(
        self, weights: dict, hidden_size: int, oracle: bool, model_arch: str,
        iteration: int, loss: float, placement: float, path: str,
    ) -> None:
        torch.save({
            "model_state_dict": weights,
            "hidden_size": hidden_size,
            "oracle_critic": oracle,
            "model_arch": model_arch,
            "iteration": iteration,
            "loss": loss,
            "placement": placement,
        }, path)

    def copy_best(self, src: str, dst: str) -> None:
        shutil.copy2(src, dst)


def _export_torchscript(model: TarokNetV4, path: str) -> None:
    class _Wrapper(torch.nn.Module):
        def __init__(self, base: TarokNetV4):
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor):
            s = self.base.shared(x)
            s = self.base.res_blocks(s)
            cf = self.base._extract_card_features(x)
            a = self.base.card_attention(cf)
            f = self.base.fuse(torch.cat([s, a], dim=-1))
            return (
                self.base.bid_head(f),
                self.base.king_head(f),
                self.base.talon_head(f),
                self.base.card_logits_for_export(f, x),
                self.base.critic(f).squeeze(-1),
            )

    w = _Wrapper(model)
    w.eval()
    with torch.no_grad():
        traced = torch.jit.trace(w, torch.randn(1, 450), check_trace=False)
    traced.save(path)
