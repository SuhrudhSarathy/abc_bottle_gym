"""Roll out random actions and save a top/left/right camera video to disk.

Requires the "video" extra: `uv sync --extra video`

Usage:
    uv run --extra video examples/save_video.py --seed 0 --steps 60 --out rollout.mp4
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np

import abc_bottle_gym  # noqa: F401
import gymnasium as gym


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--out", type=str, default="rollout.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env = gym.make("PutBottlesInBin-v0", render_mode="rgb_array").unwrapped
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    writer.append_data(env.render())
    for step in range(args.steps):
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        writer.append_data(env.render())
        if terminated or truncated:
            break
    writer.close()
    env.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
