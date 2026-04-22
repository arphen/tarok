"""Duplicate-RL adapters.

These adapters implement the three ports introduced by the duplicate RL
feature:

* :class:`training.ports.duplicate_pairing_port.DuplicatePairingPort`
* :class:`training.ports.duplicate_reward_port.DuplicateRewardPort`
* :class:`training.ports.selfplay_port.SelfPlayPort.run_seeded_pods` extension

See ``docs/double_rl.md`` for the end-to-end design.
"""
