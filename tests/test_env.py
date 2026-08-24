from __future__ import annotations

import numpy as np
import pytest

import abc_bottle_gym  # noqa: F401
import gymnasium as gym
from abc_bottle_gym.env import PutBottlesEnv


def make_env(**kwargs) -> PutBottlesEnv:
    return gym.make("PutBottlesInBin-v0", **kwargs).unwrapped


def test_registered():
    env = gym.make("PutBottlesInBin-v0")
    assert env is not None
    env.close()


def test_reset_obs_matches_spaces():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["state"].shape == (14,)
    assert info["num_active_bottles"] == 6
    env.close()


def test_step_obs_matches_spaces_and_reward_bounds():
    env = make_env()
    env.reset(seed=1)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert 0.0 <= reward <= 1.0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_same_seed_is_deterministic():
    env = make_env()
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    np.testing.assert_allclose(obs1["state"], obs2["state"])
    assert info1["randomization"]["bottle_states"] == info2["randomization"]["bottle_states"]
    assert info1["randomization"]["bin_state"] == info2["randomization"]["bin_state"]
    env.close()


def test_different_seeds_differ():
    env = make_env()
    _, info1 = env.reset(seed=1)
    _, info2 = env.reset(seed=2)
    assert info1["randomization"]["bottle_states"] != info2["randomization"]["bottle_states"]
    env.close()


def test_action_bounds_cover_home_pose():
    env = make_env()
    env.reset(seed=0)
    home = np.array(env.scene.init_q, dtype=np.float32)
    assert env.action_space.contains(home)
    env.close()


def test_truncation_at_max_episode_steps():
    # Construct directly: gym.make() intercepts `max_episode_steps` itself
    # (for its own TimeLimit wrapper) rather than forwarding it to __init__.
    env = PutBottlesEnv(max_episode_steps=2)
    env.reset(seed=0)
    home = np.array(env.scene.init_q, dtype=np.float32)
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(home)
    assert truncated
    env.close()


def test_include_images_populates_observation():
    env = make_env(include_images=True)
    obs, info = env.reset(seed=0)
    assert set(obs["images"].keys()) == {"top", "left", "right"}
    for image in obs["images"].values():
        assert image.shape == (3, 168, 224)
        assert image.dtype == np.uint8
    env.close()


def test_gymnasium_check_env():
    check_env = pytest.importorskip("gymnasium.utils.env_checker").check_env
    env = PutBottlesEnv()
    check_env(env, skip_render_check=True)
    env.close()
