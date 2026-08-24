"""Run a random-action rollout and print per-step task metrics.

Usage:
    uv run examples/random_rollout.py
    uv run examples/random_rollout.py --seed 3 --steps 100
"""

from __future__ import annotations

import argparse

import numpy as np

import abc_bottle_gym  # noqa: F401  (registers "PutBottlesInBin-v0")
import gymnasium as gym


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    env = gym.make("PutBottlesInBin-v0")
    obs, info = env.reset(seed=args.seed)
    print(f"reset: state.shape={obs['state'].shape} bottles_in_bin={info['num_bottles_in_bin']}")

    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step:03d} reward={reward:.3f} "
            f"bottles={info['num_bottles_in_bin']}/{info['num_active_bottles']} "
            f"success={info['success']}"
        )
        if terminated or truncated:
            print("episode ended, resetting")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
