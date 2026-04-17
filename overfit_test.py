from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training-lab"))
sys.path.insert(0, str(ROOT / "model" / "src"))

import torch


def _preload_torch_dylibs() -> None:
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = f"{torch_lib}:{existing}" if existing else str(torch_lib)
    for name in ["libc10.dylib", "libtorch.dylib", "libtorch_cpu.dylib", "libtorch_python.dylib"]:
        path = torch_lib / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


_preload_torch_dylibs()

import tarok_engine as te

from tarok_model.network import TarokNetV4
from training.adapters.model import TorchModelAdapter
from training.adapters.ppo import PPOAdapter
from training.entities import TrainingConfig


def _export_random_model(hidden_size: int, oracle_critic: bool) -> tuple[dict[str, torch.Tensor], str]:
    weights = TarokNetV4(hidden_size=hidden_size, oracle_critic=oracle_critic).state_dict()
    tmp = tempfile.NamedTemporaryFile(prefix="tarok_overfit_", suffix=".pt", delete=False)
    tmp.close()
    TorchModelAdapter().export_for_inference(
        weights=weights,
        hidden_size=hidden_size,
        oracle=oracle_critic,
        model_arch="v4",
        path=tmp.name,
    )
    return weights, tmp.name


def run_overfit_test() -> None:
    hidden_size = 256
    oracle_critic = True
    device = "cpu"

    print("Generating 1 PPO self-play game for overfit test...")
    weights, model_path = _export_random_model(hidden_size=hidden_size, oracle_critic=oracle_critic)
    try:
        raw_data = te.run_self_play(
            n_games=1,
            concurrency=1,
            model_path=model_path,
            explore_rate=0.0,
            seat_config="nn,bot_v5,bot_v5,bot_v5",
            include_replay_data=False,
            include_oracle_states=True,
        )

        print(f"Batch size: {raw_data['n_experiences']} experiences.")

        config = TrainingConfig(
            model_arch="v4",
            lr=0.001,             # Bump this slightly to 1e-3 for faster memorization
            ppo_epochs=1,
            batch_size=8192,
            imitation_coef=0.0,
            entropy_coef=0.0,     # Let it be 100% certain (0.0 penalty)
           # clip_epsilon=100.0,   # Turn off the PPO clipping wall
            device=device,
        )

        adapter = PPOAdapter()
        adapter.setup(weights=weights, config=config, device=device)

        print()
        print("Starting Overfit Loop...")
        print(f"{'Step':<6} | {'Pol Loss':<10} | {'Val Loss':<10} | {'IL Loss':<10} | {'Entropy':<10}")
        print("-" * 60)

        for step in range(1, 501):
            metrics, _ = adapter.update(raw_data, nn_seats=[0])

            if step % 50 == 0 or step == 1:
                p_loss = metrics["policy_loss"]
                v_loss = metrics["value_loss"]
                i_loss = metrics.get("il_loss", 0.0)
                ent = metrics["entropy"]
                print(f"{step:<6} | {p_loss:<10.4f} | {v_loss:<10.4f} | {i_loss:<10.4f} | {ent:<10.4f}")
    finally:
        Path(model_path).unlink(missing_ok=True)


if __name__ == "__main__":
    run_overfit_test()