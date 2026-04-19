"""PPO adapter package.

Public API is intentionally stable so existing imports keep working:

    from training.adapters.ppo import PPOAdapter
    from training.adapters.ppo import load_human_experiences, merge_experiences
"""

from training.adapters.ppo.jsonl_human_replay import load_human_experiences, merge_experiences
from training.adapters.ppo.expert_replay import load_expert_experiences
from training.adapters.ppo.torch_ppo import PPOAdapter

__all__ = [
    "PPOAdapter",
    "load_human_experiences",
    "load_expert_experiences",
    "merge_experiences",
]
