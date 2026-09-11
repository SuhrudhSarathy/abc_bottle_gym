# abc-bottle-gym

A standalone [Gymnasium](https://gymnasium.farama.org/) environment for the
**put-bottles-in-bin** MuJoCo task from the
[ABC project](https://github.com/amazon-far/abc) ([abc.bot](https://abc.bot)).
Two bimanual YAM arms must pick 6 loose bottles off a table and drop them
into a bin.

This package extracts just the task/simulation from ABC's
`abc_minimal/eval_policy.py` so it can be installed and tested by anyone with
`pip`/`uv`, without `torch`, the DiT policy, CLIP, DINOv3, `viser`, or a
pretrained checkpoint.

**This requires an NVIDIA GPU with a working CUDA driver.** Physics and
rendering both run on MuJoCo-Warp (`mujoco-warp` + `warp-lang`), the same
GPU simulator/renderer the upstream project uses by default for policy
training and eval. There is no CPU/native-MuJoCo fallback: native MuJoCo's
rendering (lighting, antialiasing, shadows) looks different enough from
MJWarp's that it's a real distribution-shift risk for any policy trained on
MJWarp-rendered images — if you're evaluating a checkpoint, you want the
exact renderer it was trained/tested against, not a lookalike.

## What matches the original

Matches exactly — this is the same default (GPU) code path the original
project runs, not a reimplementation:
- Scene XML, meshes, and per-episode randomization (bottle scale/pose, bin
  scale/pose), driven by the same `numpy.random.Generator(seed)` logic as
  upstream — the same `seed` produces the same scene.
- Physics: MJWarp's single-world (`nworld=1`) GPU stepper at
  `timestep=0.002`, `control_decimation=17` steps per action — the same
  simulator, not `mujoco.mj_step`.
- Rendering: MJWarp's GPU rasterizer via the same
  `mujoco_warp.create_render_context` / `render` calls, same default
  resolution (168x224).
- The 14-dim state/action layout (per arm: 6 joint positions in radians +
  1 normalized gripper value in `[0, 1]`, left arm then right arm) and the
  success/reward metric (fraction of bottles resting inside the bin volume).

No policy / DiT model / CLIP text encoder / DINOv3 vision backbone is
included — this package is the environment only. Bring your own policy
(random, scripted, RL-trained, or your own imitation-learning model) and
drive it through the standard Gymnasium `step`/`reset` API.

Note: GPU physics solvers are not bit-deterministic run-to-run (floating
point reduction order varies), so `gymnasium`'s own `check_env` will warn
that two rollouts from the same seed/actions are "similar" rather than
exactly equal — this is expected GPU-solver behavior, not a bug in the
scene/task logic (which *is* exactly reproducible; see `test_same_seed_is_deterministic`).

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
git clone git@github.com:SuhrudhSarathy/abc_bottle_gym.git
cd abc_bottle_gym
uv sync
```

Dependencies are pinned to the exact `mujoco`/`mujoco-warp`/`warp-lang`
combination the upstream ABC project tests against (its `uv.lock`) — these
three are versioned in lockstep, and drifting `mujoco-warp` ahead of
`warp-lang` hits a real kernel-metadata bug in warp's caching of
`mujoco-warp`'s dynamically-built collision kernels on repeated model
rebuilds (which this env does every `reset()`). Don't bump them
independently without retesting.

The first `reset()`/`step()` in a process JIT-compiles MJWarp's CUDA
kernels — this can take a few minutes the very first time (cached
afterwards under `~/.cache/warp/`, and reused across scene/seed variations
within the same process).

## Quick start

```python
import abc_bottle_gym  # registers "PutBottlesInBin-v0"
import gymnasium as gym
import numpy as np

env = gym.make("PutBottlesInBin-v0")
obs, info = env.reset(seed=0)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

- `obs["state"]`: `float32[14]` — arm/gripper state (see layout above).
- `obs["images"]` (only if `include_images=True` or `render_mode="rgb_array"`):
  `dict[str, uint8[3, H, W]]` per camera in `camera_keys` (default
  `("top", "left", "right")`, matching the DiT policy's default cameras;
  `"overhead"`, `"left_side"`, `"right_side"` are also available).
- `reward`: fraction of the 6 bottles currently inside the bin, in `[0, 1]`.
- `terminated`: all 6 bottles have ever simultaneously been in the bin.
- `truncated`: `max_episode_steps` (default 1800, matching the original's
  `120 chunks x 15 actions`) reached.
- `info`: full task-evaluation dict (`num_bottles_in_bin`,
  `bottles_in_bin`, `bottle_in_bin_mask`, ...); `info["randomization"]` is
  also included on `reset()`.

Constructor kwargs (`gym.make("PutBottlesInBin-v0", **kwargs)`):

| kwarg | default | purpose |
| --- | --- | --- |
| `render_mode` | `None` | `"human"` (interactive viewer) or `"rgb_array"` |
| `camera_keys` | `("top", "left", "right")` | which cameras to render |
| `include_images` | `False` | include `obs["images"]` without needing `render_mode="rgb_array"` |
| `height`, `width` | `168`, `224` | camera resolution (matches upstream default) |
| `max_episode_steps` | `1800` | truncation horizon |
| `scene` | `PutBottlesSimConfig()` | override randomization ranges / task thresholds |
| `gpu_id` | `None` | pin to a specific CUDA device index on multi-GPU machines |

## Examples

```bash
uv run examples/random_rollout.py                 # random actions, prints metrics
uv run examples/interactive_viewer.py              # opens MuJoCo's 3D viewer
uv run --extra video examples/save_video.py --out rollout.mp4   # renders an mp4
```

## Tests

```bash
uv run --extra dev pytest
```

## License

Apache License 2.0 (see `LICENSE`), the same license as the upstream ABC
project this is adapted from. See `NOTICE` for attribution details and the
citation for the original paper.

Bundled MuJoCo assets under `src/abc_bottle_gym/assets/assets/i2rt_yam/`
(the YAM robot arm) are from i2rt robotics and are separately licensed MIT —
see `src/abc_bottle_gym/assets/assets/i2rt_yam/LICENSE`
(mirrored at `third_party_licenses/i2rt_yam_LICENSE`).
