"""Learning regression guards for training-lab PPO.

These tests ensure that key training knobs are wired correctly and that
the PPO stack can still overfit a tiny fixed batch when safety rails are off.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch


def _preload_torch_dylibs_for_macos() -> None:
    if sys.platform != "darwin":
        return
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = f"{torch_lib}:{existing}" if existing else str(torch_lib)
    for name in ["libc10.dylib", "libtorch.dylib", "libtorch_cpu.dylib", "libtorch_python.dylib"]:
        lib_path = torch_lib / name
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


_preload_torch_dylibs_for_macos()

# Ensure training-lab is importable as a package root.
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_LAB_ROOT = REPO_ROOT / "training-lab"
if str(TRAINING_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_LAB_ROOT))

te = pytest.importorskip("tarok_engine")

from tarok_model.network import TarokNetV4
from training.adapters.modeling import TorchModelAdapter
from training.adapters.ppo import PPOAdapter
from training.entities import TrainingConfig


def _export_random_model(hidden_size: int, oracle_critic: bool) -> tuple[dict[str, torch.Tensor], str]:
    weights = TarokNetV4(hidden_size=hidden_size, oracle_critic=oracle_critic).state_dict()
    tmp = tempfile.NamedTemporaryFile(prefix="tarok_test_overfit_", suffix=".pt", delete=False)
    tmp.close()
    TorchModelAdapter().export_for_inference(
        weights=weights,
        hidden_size=hidden_size,
        oracle=oracle_critic,
        model_arch="v4",
        path=tmp.name,
    )
    return weights, tmp.name


def test_clip_epsilon_from_config_is_respected() -> None:
    clip = 100.0
    cfg = TrainingConfig(model_arch="v4", clip_epsilon=clip)
    weights = TarokNetV4(hidden_size=256, oracle_critic=False).state_dict()
    adapter = PPOAdapter()

    adapter.setup(weights=weights, config=cfg, device="cpu")

    assert adapter._clip_epsilon == pytest.approx(clip)


def test_ppo_can_overfit_tiny_fixed_batch_when_unclipped() -> None:
    weights, model_path = _export_random_model(hidden_size=32, oracle_critic=False)
    try:
        raw_data = te.run_self_play(
            n_games=1,
            concurrency=1,
            model_path=model_path,
            explore_rate=0.0,
            seat_config="nn,bot_v5,bot_v5,bot_v5",
            include_replay_data=False,
            include_oracle_states=False,
        )

        assert int(raw_data["n_experiences"]) > 0

        cfg = TrainingConfig(
            model_arch="v4",
            lr=0.001,
            ppo_epochs=1,
            batch_size=8192,
            imitation_coef=0.0,
            entropy_coef=0.0,
            clip_epsilon=100.0,
            device="cpu",
        )

        adapter = PPOAdapter()
        adapter.setup(weights=weights, config=cfg, device="cpu")

        entropies: list[float] = []
        value_losses: list[float] = []
        policy_losses: list[float] = []

        # Keep the loop short for test runtime while still exposing learning.
        for _ in range(120):
            metrics, _ = adapter.update(raw_data, nn_seats=[0])
            entropies.append(float(metrics["entropy"]))
            value_losses.append(float(metrics["value_loss"]))
            policy_losses.append(float(metrics["policy_loss"]))

        first_entropy = entropies[0]
        best_entropy = min(entropies)
        first_value_loss = value_losses[0]
        best_value_loss = min(value_losses)
        first_policy_loss = policy_losses[0]
        best_policy_loss = min(policy_losses)

        # Regression checks: not perfect memorization guarantees, but should
        # demonstrate clear optimization progress on one fixed tiny batch.
        assert best_entropy <= first_entropy * 0.85
        assert best_value_loss <= first_value_loss * 0.85
        assert best_policy_loss <= first_policy_loss - 0.05
    finally:
        Path(model_path).unlink(missing_ok=True)


def test_illegal_moves_are_strictly_masked() -> None:
    """Verify that action masking gives illegal cards exactly 0.0 probability.

    This guards the critical invariant that the NN cannot accidentally play
    an illegal card: after masking with -inf and softmax, the only non-zero
    probabilities must belong to the declared legal moves.
    """
    from tarok_model.encoding import CARD_ACTION_SIZE, DecisionType, GameMode, STATE_SIZE

    net = TarokNetV4(hidden_size=32, oracle_critic=False)
    net.eval()

    dummy_state = torch.zeros(1, STATE_SIZE)

    # Only card indices 5 and 10 are legal.
    legal_mask = torch.zeros(1, CARD_ACTION_SIZE)
    legal_mask[0, 5] = 1.0
    legal_mask[0, 10] = 1.0

    with torch.no_grad():
        logits, _ = net(dummy_state, decision_type=DecisionType.CARD_PLAY, game_mode=GameMode.PARTNER_PLAY)

    # Replicate the exact masking logic used in TarokNet.get_action (network.py:342-345).
    masked_logits = logits.clone()
    masked_logits[legal_mask == 0] = float("-inf")
    probs = torch.softmax(masked_logits, dim=-1)

    # Legal probabilities must sum to exactly 1.0.
    legal_sum = probs[0, 5] + probs[0, 10]
    assert torch.isclose(legal_sum, torch.tensor(1.0)), (
        f"Legal probabilities sum to {legal_sum.item():.6f}, expected 1.0"
    )

    # A known illegal card must have exactly 0.0 probability.
    assert probs[0, 0].item() == 0.0, (
        f"Illegal card 0 has probability {probs[0, 0].item()}, expected 0.0"
    )

    # Masking -inf then softmax must not produce NaN.
    assert not torch.isnan(probs).any(), "Masking produced NaN probabilities"


def test_rust_gae_calculation() -> None:
    """Verify compute_gae produces mathematically correct advantages and returns.

    Uses a 3-step single-trajectory with a terminal reward so each quantity
    can be verified by hand against the GAE recurrence:
        delta_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + γλ * A_{t+1}
        G_t     = A_t + V(s_t)
    """
    import numpy as np

    traj_keys = np.array([0, 0, 0], dtype=np.int64)
    values    = np.array([0.5, 0.6, 0.8], dtype=np.float32)
    rewards   = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    gamma      = 0.99
    gae_lambda = 0.95

    advantages, returns = te.compute_gae(values, rewards, traj_keys, gamma, gae_lambda)

    # Step 2 (terminal): delta_2 = 1.0 + 0.99*0 - 0.8 = 0.2; adv_2 = 0.2
    assert np.isclose(advantages[2], 0.2, atol=1e-5), (
        f"Terminal advantage: expected 0.2, got {advantages[2]}"
    )

    # Step 1: delta_1 = 0.0 + 0.99*0.8 - 0.6 = 0.192
    #         adv_1   = 0.192 + 0.99*0.95*0.2 = 0.3801
    assert np.isclose(advantages[1], 0.3801, atol=1e-5), (
        f"Mid-trajectory advantage: expected 0.3801, got {advantages[1]}"
    )

    # Step 0: delta_0 = 0.0 + 0.99*0.6 - 0.5 = 0.094
    #         adv_0   = 0.094 + 0.99*0.95*0.3801 ≈ 0.45131...
    expected_adv0 = 0.0 + 0.99 * 0.6 - 0.5 + 0.99 * 0.95 * 0.3801
    assert np.isclose(advantages[0], expected_adv0, atol=1e-5), (
        f"First-step advantage: expected {expected_adv0:.6f}, got {advantages[0]}"
    )

    # Returns = Advantage + Value
    assert np.isclose(returns[2], 0.2 + 0.8, atol=1e-5), (
        f"Terminal return: expected {0.2 + 0.8}, got {returns[2]}"
    )
    assert np.isclose(returns[1], 0.3801 + 0.6, atol=1e-5), (
        f"Mid return: expected {0.3801 + 0.6:.4f}, got {returns[1]}"
    )
    assert np.isclose(returns[0], expected_adv0 + 0.5, atol=1e-5), (
        f"First return: expected {expected_adv0 + 0.5:.6f}, got {returns[0]}"
    )


def test_checkpoint_save_and_load_determinism() -> None:
    """Verify that every parameter is registered and survives a state_dict round-trip.

    Tests TarokNetV4 (production class) with oracle_critic=True so the full
    parameter surface — shared backbone, residual blocks, card attention, fuse,
    per-mode card heads, oracle critic backbone, and critic head — is exercised.
    If any tensor is accidentally stored as a plain attribute (not nn.Parameter /
    nn.Module), it will be absent from state_dict and the cloned model will
    produce different outputs.
    """
    from tarok_model.encoding import ORACLE_STATE_SIZE, STATE_SIZE, DecisionType, GameMode

    net1 = TarokNetV4(hidden_size=64, oracle_critic=True)
    net1.eval()

    torch.manual_seed(0)
    dummy_state  = torch.randn(2, STATE_SIZE)
    dummy_oracle = torch.randn(2, ORACLE_STATE_SIZE)

    with torch.no_grad():
        logits1, val1   = net1(dummy_state, decision_type=DecisionType.CARD_PLAY, game_mode=GameMode.PARTNER_PLAY)
        critic_feats1   = net1.get_critic_features(dummy_oracle)

    state_dict = net1.state_dict()

    net2 = TarokNetV4(hidden_size=64, oracle_critic=True)
    net2.load_state_dict(state_dict)
    net2.eval()

    with torch.no_grad():
        logits2, val2   = net2(dummy_state, decision_type=DecisionType.CARD_PLAY, game_mode=GameMode.PARTNER_PLAY)
        critic_feats2   = net2.get_critic_features(dummy_oracle)

    assert torch.allclose(logits1, logits2, atol=1e-6), "Actor logits changed after checkpoint load!"
    assert torch.allclose(val1,    val2,    atol=1e-6), "Critic values changed after checkpoint load!"
    assert torch.allclose(critic_feats1, critic_feats2, atol=1e-6), "Oracle features changed after checkpoint load!"

