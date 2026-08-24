"""Watch random-action episodes in MuJoCo's interactive 3D viewer.

The viewer window supports the usual MuJoCo controls: drag objects with
Ctrl + right-click, orbit with left-click drag, zoom with scroll. A fresh
scene (new bottle/bin scales and poses) is rebuilt every episode, matching
what a real `env.reset(seed=...)` call does -- so the viewer window is
recreated at each reset.

Usage:
    uv run examples/interactive_viewer.py
    uv run examples/interactive_viewer.py --seed 0 --episodes 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import abc_bottle_gym  # noqa: F401
import gymnasium as gym


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = gym.make("PutBottlesInBin-v0", render_mode="human").unwrapped
    rng = np.random.default_rng(args.seed)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        print(f"episode {episode}: seed={args.seed + episode}")
        done = False
        while not done:
            t0 = time.perf_counter()
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Real-time-ish playback: one action = control_decimation physics
            # steps = control_decimation * timestep seconds of sim time.
            sim_dt = env.scene.control_decimation * env.scene.timestep
            sleep_s = sim_dt - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
        print(f"episode {episode} done: bottles={info['num_bottles_in_bin']}/{info['num_active_bottles']}")

    env.close()


if __name__ == "__main__":
    main()
