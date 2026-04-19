#!/usr/bin/env python3
"""CartPole sanity adapter for validating custom training loops via Gymnasium.

This script is intentionally isolated in adapters so core Tarok use cases remain
unchanged. It trains a tiny actor-critic policy on CartPole-v1 and reports
learning progress.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - exercised at runtime
    raise SystemExit(
        "gymnasium is not installed. Install with: pip install gymnasium"
    ) from exc


class CartPoleActorCritic(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, 2)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CartPole sanity training via Gymnasium")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps-per-update", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--value-clip-epsilon",
        type=float,
        default=0.2,
        help="PPO-style clipping epsilon for value updates",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--target-avg-reward",
        type=float,
        default=475.0,
        help="Report PASS when moving average reaches this value",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if target average reward is not reached",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_steps = rewards.shape[0]
    advantages = torch.zeros(n_steps, dtype=torch.float32)
    gae = 0.0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_value = last_value
        else:
            next_value = float(values[t + 1])

        non_terminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_value * non_terminal - float(values[t])
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def _build_batch(
    obs_list: list[np.ndarray],
    actions_list: list[int],
    log_probs_list: list[float],
    rewards_list: list[float],
    dones_list: list[bool],
    values_list: list[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> RolloutBatch:
    observations = torch.tensor(np.asarray(obs_list), dtype=torch.float32)
    actions = torch.tensor(actions_list, dtype=torch.int64)
    old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32)
    rewards = torch.tensor(rewards_list, dtype=torch.float32)
    dones = torch.tensor(dones_list, dtype=torch.float32)
    values = torch.tensor(values_list, dtype=torch.float32)

    advantages, returns = _compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    return RolloutBatch(
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=values,
        returns=returns,
        advantages=advantages,
    )


def _ppo_update(
    model: CartPoleActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    ppo_epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coef: float,
    value_clip_epsilon: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> tuple[float, float, float, float]:
    n = batch.actions.shape[0]
    effective_minibatch = min(minibatch_size, n)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    value_target_stds: list[float] = []

    for _ in range(ppo_epochs):
        permutation = torch.randperm(n)
        for start in range(0, n, effective_minibatch):
            idx = permutation[start : start + effective_minibatch]
            obs_mb = batch.observations[idx]
            actions_mb = batch.actions[idx]
            old_log_probs_mb = batch.old_log_probs[idx]
            old_values_mb = batch.old_values[idx]
            returns_mb = batch.returns[idx]
            advantages_mb = batch.advantages[idx]

            logits, values = model(obs_mb)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_mb)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_mb)
            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            # Normalize value targets per minibatch so critic gradients stay bounded.
            ret_mean = returns_mb.mean()
            ret_std = returns_mb.std(unbiased=False).clamp_min(1e-6)
            value_target_stds.append(float(ret_std.item()))

            norm_values = (values - ret_mean) / ret_std
            norm_old_values = (old_values_mb - ret_mean) / ret_std
            norm_returns = (returns_mb - ret_mean) / ret_std

            if value_clip_epsilon > 0.0:
                norm_v_pred_clipped = norm_old_values + torch.clamp(
                    norm_values - norm_old_values,
                    -value_clip_epsilon,
                    value_clip_epsilon,
                )
                v_loss_unclipped = torch.nn.functional.mse_loss(
                    norm_values,
                    norm_returns,
                    reduction="none",
                )
                v_loss_clipped = torch.nn.functional.mse_loss(
                    norm_v_pred_clipped,
                    norm_returns,
                    reduction="none",
                )
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = torch.nn.functional.mse_loss(norm_values, norm_returns)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    return (
        float(np.mean(policy_losses)),
        float(np.mean(value_losses)),
        float(np.mean(entropies)),
        float(np.mean(value_target_stds)) if value_target_stds else 0.0,
    )


def run_cartpole_sanity(args: argparse.Namespace) -> tuple[bool, float]:
    _set_seed(args.seed)
    env = gym.make("CartPole-v1", max_episode_steps=args.max_steps)

    model = CartPoleActorCritic(hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rolling_rewards: deque[float] = deque(maxlen=args.window)
    best_avg = 0.0
    episodes_done = 0

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0

    obs_list: list[np.ndarray] = []
    actions_list: list[int] = []
    log_probs_list: list[float] = []
    rewards_list: list[float] = []
    dones_list: list[bool] = []
    values_list: list[float] = []

    policy_loss = 0.0
    value_loss = 0.0
    entropy = 0.0
    value_target_std = 0.0
    n_updates = 0

    def _policy_entropy_for_obs(current_obs: np.ndarray) -> float:
        with torch.no_grad():
            logits, _ = model(torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0))
            return float(Categorical(logits=logits).entropy().item())

    while episodes_done < args.episodes:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = bool(terminated or truncated)

        obs_list.append(np.asarray(obs, dtype=np.float32))
        actions_list.append(int(action.item()))
        log_probs_list.append(float(log_prob.item()))
        rewards_list.append(float(reward))
        dones_list.append(done)
        values_list.append(float(value.item()))

        episode_reward += float(reward)
        obs = next_obs

        if done:
            episodes_done += 1
            rolling_rewards.append(episode_reward)
            avg_reward = float(np.mean(rolling_rewards))
            best_avg = max(best_avg, avg_reward)
            if episodes_done % args.print_every == 0 or episodes_done == 1:
                live_entropy = _policy_entropy_for_obs(obs)
                update_entropy = f"{entropy:7.4f}" if n_updates > 0 else "   n/a "
                print(
                    f"episode={episodes_done:4d} reward={episode_reward:6.1f} "
                    f"avg_{args.window}={avg_reward:7.2f} p_loss={policy_loss:9.4f} "
                    f"v_loss={value_loss:9.4f} ent(upd)={update_entropy} "
                    f"ent(live)={live_entropy:7.4f} v_tgt_std={value_target_std:7.3f} "
                    f"updates={n_updates:3d}"
                )

            obs, _ = env.reset(seed=args.seed + episodes_done)
            episode_reward = 0.0

        need_update = len(obs_list) >= args.steps_per_update
        final_flush = episodes_done >= args.episodes and len(obs_list) > 0
        if need_update or final_flush:
            with torch.no_grad():
                last_value = 0.0
                if not dones_list[-1]:
                    last_obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    _, last_val_t = model(last_obs_t)
                    last_value = float(last_val_t.item())

            batch = _build_batch(
                obs_list=obs_list,
                actions_list=actions_list,
                log_probs_list=log_probs_list,
                rewards_list=rewards_list,
                dones_list=dones_list,
                values_list=values_list,
                last_value=last_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

            policy_loss, value_loss, entropy, value_target_std = _ppo_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                ppo_epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_epsilon=args.clip_epsilon,
                value_coef=args.value_coef,
                value_clip_epsilon=args.value_clip_epsilon,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
            )
            n_updates += 1

            obs_list.clear()
            actions_list.clear()
            log_probs_list.clear()
            rewards_list.clear()
            dones_list.clear()
            values_list.clear()

    env.close()

    passed = best_avg >= args.target_avg_reward
    status = "PASS" if passed else "WARN"
    print(
        f"[{status}] best_avg_{args.window}={best_avg:.2f} "
        f"target={args.target_avg_reward:.2f} episodes={args.episodes}"
    )
    return passed, best_avg


def main() -> None:
    args = _parse_args()
    passed, _ = run_cartpole_sanity(args)
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
