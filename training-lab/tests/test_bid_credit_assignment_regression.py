"""Bidding-head credit-assignment regression test.

Hypothesis under test
---------------------
The PPO bidding head receives strong-enough gradient signal from terminal
rewards that, given a fixed bidding state where bidding *berač* yields a
~5% win rate (expected reward < 0) while passing yields 0, the network
should *learn to bid berač less often* over a small number of PPO updates.

Why a synthetic test
--------------------
We deliberately bypass the Rust self-play stack and the cardplay head:

* No engine games are run — every "experience" is a 1-step trajectory with
  decision_type = BID.
* Rewards are injected via ``precomputed_rewards`` so the test isolates the
  bidding-head gradient from confounders (cardplay shaping, contract-base
  scoring, partner reward attribution, GAE bootstrap from a noisy critic).
* Only two bid actions are legal: pass (idx 0) and berač (idx 2). This
  matches the 3p bid layout in
  [engine-rs/src/encoding_3p.rs](../../engine-rs/src/encoding_3p.rs).

If this test fails, the duplicate-RL bidding head is *not* getting a
useful gradient even when the reward signal is unambiguous, which would
explain why berač win-rate stays near random in real training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure training-lab is importable as a package root.
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_LAB_ROOT = REPO_ROOT / "training-lab"
if str(TRAINING_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_LAB_ROOT))

from tarok_model.encoding import BID_ACTION_SIZE, DecisionType
from tarok_model.network_3p import DEFAULT_STATE_SIZE_3P, TarokNet3
from training.adapters.ppo import PPOAdapter
from training.entities import TrainingConfig

# 3p bid action layout (see engine-rs/src/encoding_3p.rs::BID_IDX_TO_CONTRACT_3P).
PASS_IDX = 0
BERAC_IDX = 2

# Probability with which "playing berač" is rewarded as a win in this synthetic
# environment. Below 50% means bidding berač has negative expected value vs the
# safe pass alternative (reward 0).
BERAC_WIN_RATE = 0.05


def _bid_probabilities(network: TarokNet3, state: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Return the masked softmax over the BID head for a single state."""
    network.eval()
    with torch.no_grad():
        # network_3p forward: pads 8-logit BID head to 9 to match v4 width.
        # Use evaluate_action with a dummy action so we hit the same code path
        # the trainer uses, then re-derive probs from logits via a separate
        # forward to keep this readable.
        from tarok_model.network_3p import DecisionType3P

        logits, _ = network.forward(state.unsqueeze(0), DecisionType3P.BID)
        # Pad to BID_ACTION_SIZE (=9) the same way evaluate_action does.
        if logits.shape[-1] < BID_ACTION_SIZE:
            pad_cols = BID_ACTION_SIZE - logits.shape[-1]
            pad = torch.full((1, pad_cols), -1e9, dtype=logits.dtype)
            logits = torch.cat([logits, pad], dim=-1)
        masked = logits.clone()
        masked[legal_mask.unsqueeze(0) == 0] = float("-inf")
        probs = torch.softmax(masked, dim=-1).squeeze(0)
    network.train()
    return probs


def _build_synthetic_bid_batch(
    *,
    network: TarokNet3,
    fixed_state: torch.Tensor,
    legal_mask_row: torch.Tensor,
    n_steps: int,
    rng: np.random.Generator,
) -> dict:
    """Sample ``n_steps`` bid actions from the *current* policy at ``fixed_state``
    and produce a raw_experiences dict consumable by ``PPOAdapter.update``.

    Each step is its own one-step trajectory (game_id = step_idx, player = 0)
    so terminal-reward credit goes to the single step that produced the bid.
    """
    from tarok_model.network_3p import DecisionType3P

    state_dim = fixed_state.shape[-1]

    # Sample N actions from the current policy.
    network.eval()
    with torch.no_grad():
        logits, _ = network.forward(fixed_state.unsqueeze(0), DecisionType3P.BID)
        if logits.shape[-1] < BID_ACTION_SIZE:
            pad_cols = BID_ACTION_SIZE - logits.shape[-1]
            pad = torch.full((1, pad_cols), -1e9, dtype=logits.dtype)
            logits = torch.cat([logits, pad], dim=-1)
        masked = logits.clone()
        masked[legal_mask_row.unsqueeze(0) == 0] = float("-inf")
        probs = torch.softmax(masked, dim=-1).squeeze(0).numpy()
    network.train()

    sampled_actions = rng.choice(BID_ACTION_SIZE, size=n_steps, p=probs).astype(np.int64)
    sampled_log_probs = np.log(np.clip(probs[sampled_actions], 1e-8, 1.0)).astype(np.float32)

    # Synthetic rewards: bidding berač wins +1 with prob BERAC_WIN_RATE else -1.
    # Passing always yields 0.
    rewards = np.zeros(n_steps, dtype=np.float32)
    berac_mask = sampled_actions == BERAC_IDX
    n_berac = int(berac_mask.sum())
    if n_berac > 0:
        outcomes = rng.random(n_berac) < BERAC_WIN_RATE
        rewards[berac_mask] = np.where(outcomes, 1.0, -1.0).astype(np.float32)
    # actions == PASS_IDX → reward already 0.

    states = fixed_state.detach().numpy().astype(np.float32)
    states_batch = np.broadcast_to(states, (n_steps, state_dim)).copy()
    legal_masks = np.broadcast_to(
        legal_mask_row.numpy().astype(np.float32), (n_steps, BID_ACTION_SIZE)
    ).copy()

    raw: dict = {
        "states": states_batch,
        "actions": sampled_actions,
        "log_probs": sampled_log_probs,
        "values": np.zeros(n_steps, dtype=np.float32),
        "decision_types": np.full(n_steps, DecisionType.BID.value, dtype=np.int8),
        "game_modes": np.zeros(n_steps, dtype=np.int8),
        "legal_masks": legal_masks,
        "game_ids": np.arange(n_steps, dtype=np.int64),
        "players": np.zeros(n_steps, dtype=np.int8),
        # Required by prepare_batched even when precomputed_rewards is set.
        "scores": np.zeros((n_steps, 4), dtype=np.float32),
        "behavioral_clone_mask": np.zeros(n_steps, dtype=bool),
        "precomputed_rewards": rewards,
    }
    return raw


def test_bidding_head_learns_to_avoid_low_winrate_berac() -> None:
    """With berač winning only 5% of the time and passing being neutral, the
    bidding head should reduce its berač probability after a few PPO updates.

    This is a credit-assignment regression test. If it ever fails, the bid
    head is not learning from terminal rewards even when the signal is
    overwhelming — which is exactly the failure mode docs/double_rl.md §4.1
    is supposed to prevent.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Small TarokNet3 — enough capacity to differentiate the two legal bid
    # actions on a fixed state, small enough that PPO converges in seconds.
    network = TarokNet3(hidden_size=64, oracle_critic=False)
    weights = network.state_dict()

    cfg = TrainingConfig(
        model_arch="v3p",
        lr=5e-3,
        ppo_epochs=4,
        batch_size=4096,
        imitation_coef=0.0,
        # Drop bid-head entropy bonus so the policy is allowed to commit to
        # the higher-EV action quickly. We are testing the *gradient*, not
        # the exploration schedule.
        entropy_coef=0.0,
        bid_entropy_coef=0.0,
        clip_epsilon=100.0,  # disable PPO clipping for fast learning signal
        device="cpu",
    )

    adapter = PPOAdapter()
    adapter.setup(weights=weights, config=cfg, device="cpu")

    # Re-bind to the *same* network instance the adapter holds, so we can
    # measure the bid distribution it produces step-by-step.
    inner_network = adapter._network  # type: ignore[attr-defined]
    assert isinstance(inner_network, TarokNet3)

    fixed_state = torch.zeros(DEFAULT_STATE_SIZE_3P, dtype=torch.float32)
    # Use a non-zero state so the BID head doesn't degenerate to its bias term
    # only — small deterministic perturbation per dimension.
    fixed_state += torch.linspace(-0.3, 0.3, DEFAULT_STATE_SIZE_3P)

    legal_mask_row = torch.zeros(BID_ACTION_SIZE, dtype=torch.float32)
    legal_mask_row[PASS_IDX] = 1.0
    legal_mask_row[BERAC_IDX] = 1.0

    initial_probs = _bid_probabilities(inner_network, fixed_state, legal_mask_row)
    initial_p_berac = float(initial_probs[BERAC_IDX].item())

    # Sanity: with random init both legal actions should get non-trivial mass.
    # If init collapses to one action the test would be vacuous.
    assert 0.05 < initial_p_berac < 0.95, (
        f"random-init bid distribution is degenerate (p_berac={initial_p_berac:.4f}); "
        "test cannot prove learning."
    )

    p_berac_history: list[float] = [initial_p_berac]

    n_iterations = 25
    n_steps_per_iter = 1024

    for _ in range(n_iterations):
        raw = _build_synthetic_bid_batch(
            network=inner_network,
            fixed_state=fixed_state,
            legal_mask_row=legal_mask_row,
            n_steps=n_steps_per_iter,
            rng=rng,
        )
        adapter.update(raw, nn_seats=[0])
        probs = _bid_probabilities(inner_network, fixed_state, legal_mask_row)
        p_berac_history.append(float(probs[BERAC_IDX].item()))

    final_p_berac = p_berac_history[-1]

    # Primary assertion: berač probability dropped substantially.
    assert final_p_berac < initial_p_berac - 0.20, (
        f"Bid head did not learn to avoid berač despite a 5% win rate. "
        f"p_berac: initial={initial_p_berac:.4f} → final={final_p_berac:.4f} "
        f"(history: {[round(p, 3) for p in p_berac_history]})"
    )
    # Secondary assertion: the trajectory is monotone-ish (best ≤ initial).
    assert min(p_berac_history) <= initial_p_berac, (
        "p_berac never decreased during training — gradient sign may be flipped."
    )


def test_bidding_head_learns_to_prefer_winning_berac() -> None:
    """Mirror-image control test: when berač wins 95% of the time, the head
    should *increase* its berač probability instead of decreasing it.

    This guards against a sign error in the credit-assignment chain that
    would let the failure mode in the previous test pass for the wrong
    reason (e.g. always pushing the policy toward pass).
    """
    torch.manual_seed(1)
    rng = np.random.default_rng(1)

    network = TarokNet3(hidden_size=64, oracle_critic=False)
    weights = network.state_dict()

    cfg = TrainingConfig(
        model_arch="v3p",
        lr=5e-3,
        ppo_epochs=4,
        batch_size=4096,
        imitation_coef=0.0,
        entropy_coef=0.0,
        bid_entropy_coef=0.0,
        clip_epsilon=100.0,
        device="cpu",
    )
    adapter = PPOAdapter()
    adapter.setup(weights=weights, config=cfg, device="cpu")
    inner_network = adapter._network  # type: ignore[attr-defined]
    assert isinstance(inner_network, TarokNet3)

    fixed_state = torch.zeros(DEFAULT_STATE_SIZE_3P, dtype=torch.float32)
    fixed_state += torch.linspace(-0.3, 0.3, DEFAULT_STATE_SIZE_3P)
    legal_mask_row = torch.zeros(BID_ACTION_SIZE, dtype=torch.float32)
    legal_mask_row[PASS_IDX] = 1.0
    legal_mask_row[BERAC_IDX] = 1.0

    initial_p_berac = float(
        _bid_probabilities(inner_network, fixed_state, legal_mask_row)[BERAC_IDX].item()
    )
    assert 0.05 < initial_p_berac < 0.95

    # Flip the win rate: berač now wins 95% of the time → +EV vs pass.
    high_win_rate = 0.95
    n_iterations = 25
    n_steps_per_iter = 1024

    for _ in range(n_iterations):
        # Locally inline the batch builder with the high win rate.
        from tarok_model.network_3p import DecisionType3P

        inner_network.eval()
        with torch.no_grad():
            logits, _ = inner_network.forward(fixed_state.unsqueeze(0), DecisionType3P.BID)
            if logits.shape[-1] < BID_ACTION_SIZE:
                pad = torch.full((1, BID_ACTION_SIZE - logits.shape[-1]), -1e9, dtype=logits.dtype)
                logits = torch.cat([logits, pad], dim=-1)
            masked = logits.clone()
            masked[legal_mask_row.unsqueeze(0) == 0] = float("-inf")
            probs = torch.softmax(masked, dim=-1).squeeze(0).numpy()
        inner_network.train()

        actions = rng.choice(BID_ACTION_SIZE, size=n_steps_per_iter, p=probs).astype(np.int64)
        log_probs = np.log(np.clip(probs[actions], 1e-8, 1.0)).astype(np.float32)
        rewards = np.zeros(n_steps_per_iter, dtype=np.float32)
        berac_mask = actions == BERAC_IDX
        n_berac = int(berac_mask.sum())
        if n_berac > 0:
            outcomes = rng.random(n_berac) < high_win_rate
            rewards[berac_mask] = np.where(outcomes, 1.0, -1.0).astype(np.float32)

        raw = {
            "states": np.broadcast_to(
                fixed_state.numpy(), (n_steps_per_iter, DEFAULT_STATE_SIZE_3P)
            ).copy(),
            "actions": actions,
            "log_probs": log_probs,
            "values": np.zeros(n_steps_per_iter, dtype=np.float32),
            "decision_types": np.full(n_steps_per_iter, DecisionType.BID.value, dtype=np.int8),
            "game_modes": np.zeros(n_steps_per_iter, dtype=np.int8),
            "legal_masks": np.broadcast_to(
                legal_mask_row.numpy().astype(np.float32),
                (n_steps_per_iter, BID_ACTION_SIZE),
            ).copy(),
            "game_ids": np.arange(n_steps_per_iter, dtype=np.int64),
            "players": np.zeros(n_steps_per_iter, dtype=np.int8),
            "scores": np.zeros((n_steps_per_iter, 4), dtype=np.float32),
            "behavioral_clone_mask": np.zeros(n_steps_per_iter, dtype=bool),
            "precomputed_rewards": rewards,
        }
        adapter.update(raw, nn_seats=[0])

    final_p_berac = float(
        _bid_probabilities(inner_network, fixed_state, legal_mask_row)[BERAC_IDX].item()
    )

    assert final_p_berac > initial_p_berac + 0.20, (
        f"Bid head failed to *increase* berač probability despite 95% win rate. "
        f"p_berac: initial={initial_p_berac:.4f} → final={final_p_berac:.4f}"
    )


# ---------------------------------------------------------------------------
# Realistic-trajectory variant: bid step + N cardplay steps + terminal reward
# ---------------------------------------------------------------------------


def _build_realistic_trajectory_batch(
    *,
    network: TarokNet3,
    bid_state: torch.Tensor,
    bid_legal_mask: torch.Tensor,
    cardplay_state: torch.Tensor,
    cardplay_legal_mask: torch.Tensor,
    n_games: int,
    n_cardplay_steps: int,
    rng: np.random.Generator,
    win_rate_when_berac: float,
) -> dict:
    """Each game = 1 BID step + ``n_cardplay_steps`` CARD_PLAY steps. The terminal
    reward sits on the last cardplay step (matching the default reward path).

    The cardplay actions are filler (legal, sampled uniformly) — we only care
    whether the BID head learns from the terminal reward propagated back via
    GAE through ``n_cardplay_steps`` worth of critic bootstraps.
    """
    from tarok_model.network_3p import DecisionType3P

    # Sample bid action per game from the current policy.
    network.eval()
    with torch.no_grad():
        logits, _ = network.forward(bid_state.unsqueeze(0), DecisionType3P.BID)
        if logits.shape[-1] < BID_ACTION_SIZE:
            pad = torch.full((1, BID_ACTION_SIZE - logits.shape[-1]), -1e9, dtype=logits.dtype)
            logits = torch.cat([logits, pad], dim=-1)
        masked = logits.clone()
        masked[bid_legal_mask.unsqueeze(0) == 0] = float("-inf")
        bid_probs = torch.softmax(masked, dim=-1).squeeze(0).numpy()
    network.train()

    bid_actions = rng.choice(BID_ACTION_SIZE, size=n_games, p=bid_probs).astype(np.int64)
    bid_log_probs = np.log(np.clip(bid_probs[bid_actions], 1e-8, 1.0)).astype(np.float32)

    from tarok_model.encoding import CARD_ACTION_SIZE

    legal_card_indices = np.flatnonzero(cardplay_legal_mask.numpy())
    assert legal_card_indices.size > 0

    # Build per-game trajectories. Order in raw arrays must be sorted by
    # (game_id, player) for prepare_batched's GAE; we emit games sequentially.
    steps_per_game = 1 + n_cardplay_steps
    n_total = n_games * steps_per_game

    states = np.zeros((n_total, DEFAULT_STATE_SIZE_3P), dtype=np.float32)
    actions = np.zeros(n_total, dtype=np.int64)
    log_probs = np.zeros(n_total, dtype=np.float32)
    decision_types = np.zeros(n_total, dtype=np.int8)
    legal_masks = np.zeros((n_total, CARD_ACTION_SIZE), dtype=np.float32)
    game_modes = np.zeros(n_total, dtype=np.int8)
    game_ids = np.zeros(n_total, dtype=np.int64)
    players = np.zeros(n_total, dtype=np.int8)
    rewards = np.zeros(n_total, dtype=np.float32)

    bid_legal_padded = np.zeros(CARD_ACTION_SIZE, dtype=np.float32)
    bid_legal_padded[: BID_ACTION_SIZE] = bid_legal_mask.numpy()
    cardplay_legal_np = cardplay_legal_mask.numpy().astype(np.float32)
    bid_state_np = bid_state.numpy()
    cardplay_state_np = cardplay_state.numpy()

    for g in range(n_games):
        base = g * steps_per_game

        # BID step at offset 0.
        states[base] = bid_state_np
        actions[base] = bid_actions[g]
        log_probs[base] = bid_log_probs[g]
        decision_types[base] = DecisionType.BID.value
        legal_masks[base] = bid_legal_padded
        game_modes[base] = 1  # KLOP_BERAC mode (unused for BID head, but for cardplay)
        game_ids[base] = g
        players[base] = 0

        # CARD_PLAY filler steps.
        for s in range(1, steps_per_game):
            row = base + s
            states[row] = cardplay_state_np
            actions[row] = int(rng.choice(legal_card_indices))
            log_probs[row] = float(np.log(1.0 / legal_card_indices.size))
            decision_types[row] = DecisionType.CARD_PLAY.value
            legal_masks[row] = cardplay_legal_np
            game_modes[row] = 1
            game_ids[row] = g
            players[row] = 0

        # Terminal reward on the last cardplay step (= matches the default
        # is-terminal logic in prepare_batched: last row of the trajectory).
        if bid_actions[g] == BERAC_IDX:
            outcome = rng.random() < win_rate_when_berac
            rewards[base + steps_per_game - 1] = 1.0 if outcome else -1.0
        else:
            rewards[base + steps_per_game - 1] = 0.0

    raw: dict = {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "values": np.zeros(n_total, dtype=np.float32),
        "decision_types": decision_types,
        "game_modes": game_modes,
        "legal_masks": legal_masks,
        "game_ids": game_ids,
        "players": players,
        "scores": np.zeros((n_games, 4), dtype=np.float32),
        "behavioral_clone_mask": np.zeros(n_total, dtype=bool),
        "precomputed_rewards": rewards,
    }
    return raw


def test_bidding_head_learns_with_realistic_trajectory_shape() -> None:
    """End-to-end credit-assignment: bid step + 16 cardplay steps, terminal
    reward on the last cardplay step. This mirrors how real games look:
    the bid happens once at the start, then a long cardplay tail produces
    one terminal reward.

    If this fails but the 1-step variant above passes, the failure is in
    GAE / critic-bootstrap reward propagation across cardplay steps —
    which is exactly the failure the user is observing in real training.
    """
    torch.manual_seed(2)
    rng = np.random.default_rng(2)

    network = TarokNet3(hidden_size=64, oracle_critic=False)
    weights = network.state_dict()

    cfg = TrainingConfig(
        model_arch="v3p",
        lr=5e-3,
        ppo_epochs=4,
        batch_size=4096,
        imitation_coef=0.0,
        entropy_coef=0.0,
        bid_entropy_coef=0.0,
        clip_epsilon=100.0,
        # Use gamma=1.0 so the terminal reward reaches the bid step
        # without geometric decay through 16 cardplay steps. Matches what a
        # finite-horizon-aware training run would do.
        gamma=1.0,
        gae_lambda=1.0,
        device="cpu",
    )
    adapter = PPOAdapter()
    adapter.setup(weights=weights, config=cfg, device="cpu")
    inner_network = adapter._network  # type: ignore[attr-defined]

    bid_state = torch.linspace(-0.3, 0.3, DEFAULT_STATE_SIZE_3P, dtype=torch.float32)
    cardplay_state = torch.linspace(0.3, -0.3, DEFAULT_STATE_SIZE_3P, dtype=torch.float32)
    bid_legal_mask = torch.zeros(BID_ACTION_SIZE, dtype=torch.float32)
    bid_legal_mask[PASS_IDX] = 1.0
    bid_legal_mask[BERAC_IDX] = 1.0

    from tarok_model.encoding import CARD_ACTION_SIZE

    cardplay_legal_mask = torch.zeros(CARD_ACTION_SIZE, dtype=torch.float32)
    cardplay_legal_mask[:8] = 1.0  # 8 legal cards is plenty

    initial_p_berac = float(
        _bid_probabilities(inner_network, bid_state, bid_legal_mask)[BERAC_IDX].item()
    )
    assert 0.05 < initial_p_berac < 0.95

    n_iterations = 30
    n_games_per_iter = 256
    n_cardplay_steps = 16

    p_berac_history = [initial_p_berac]
    for _ in range(n_iterations):
        raw = _build_realistic_trajectory_batch(
            network=inner_network,
            bid_state=bid_state,
            bid_legal_mask=bid_legal_mask,
            cardplay_state=cardplay_state,
            cardplay_legal_mask=cardplay_legal_mask,
            n_games=n_games_per_iter,
            n_cardplay_steps=n_cardplay_steps,
            rng=rng,
            win_rate_when_berac=BERAC_WIN_RATE,
        )
        adapter.update(raw, nn_seats=[0])
        p_berac_history.append(
            float(_bid_probabilities(inner_network, bid_state, bid_legal_mask)[BERAC_IDX].item())
        )

    final_p_berac = p_berac_history[-1]

    assert final_p_berac < initial_p_berac - 0.15, (
        f"Bid head did not learn from terminal-only rewards across a 16-step "
        f"cardplay tail. p_berac: {initial_p_berac:.4f} → {final_p_berac:.4f}. "
        f"History (every 5th): {[round(p, 3) for p in p_berac_history[::5]]}"
    )
