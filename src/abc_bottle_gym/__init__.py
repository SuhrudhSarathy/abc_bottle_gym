"""Standalone Gymnasium port of ABC's put-bottles-in-bin MuJoCo task.

Adapted from https://github.com/amazon-far/abc (ABC project, ``abc_minimal``),
licensed Apache-2.0. See README.md and NOTICE for attribution details.
"""

from gymnasium.envs.registration import register

from abc_bottle_gym.config import PutBottlesSimConfig
from abc_bottle_gym.env import PutBottlesEnv

__all__ = ["PutBottlesEnv", "PutBottlesSimConfig"]
__version__ = "0.1.0"

register(
    id="PutBottlesInBin-v0",
    entry_point="abc_bottle_gym.env:PutBottlesEnv",
    max_episode_steps=None,  # the env truncates itself (see max_episode_steps kwarg)
)
