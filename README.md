# abc-bottle-gym

A standalone [Gymnasium](https://gymnasium.farama.org/) environment for the
**put-bottles-in-bin** MuJoCo task from the
[ABC project](https://github.com/amazon-far/abc) ([abc.bot](https://abc.bot)).
Two bimanual YAM arms must pick 6 loose bottles off a table and drop them
into a bin.

This package extracts just the task/simulation from ABC's
`abc_minimal/eval_policy.py` so it can be installed and tested by anyone with
`pip`/`uv` and MuJoCo — no CUDA, no `torch`, no `mujoco-warp`/`warp-lang`,
no `viser`, no pretrained checkpoint required.

## What matches the original, and what doesn't

Matches exactly:
- Scene XML, meshes, and per-episode randomization (bottle scale/pose, bin
  scale/pose), driven by the same `numpy.random.Generator(seed)` logic as
  upstream — the same `seed` produces the same scene.
- Physics: `mujoco.mj_step` at `timestep=0.002`,
  `control_decimation=17` steps per action. This is exactly what the
  upstream project calls its `--vanilla-physics` path, which its own README
  recommends for single (non-batched) environments as physically equivalent
  to its default GPU/MJWarp path.
- The 14-dim state/action layout (per arm: 6 joint positions in radians +
  1 normalized gripper value in `[0, 1]`, left arm then right arm) and the
  success/reward metric (fraction of bottles resting inside the bin volume).

Different by design:
- Rendering uses MuJoCo's built-in `mujoco.Renderer` / `mujoco.viewer`
  instead of the original's GPU-batched MJWarp renderer. Camera geometry and
  scene content are identical, so views are visually equivalent — just not
  bit-identical to MJWarp's rasterizer. MJWarp is built for batched
  GPU training and adds nothing for testing the task itself.
- No policy / DiT model / CLIP text encoder / DINOv3 vision backbone is
  included — this package is the environment only. Bring your own policy
  (random, scripted, RL-trained, or your own imitation-learning model) and
  drive it through the standard Gymnasium `step`/`reset` API.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
git clone <this-repo-url> abc-bottle-gym
cd abc-bottle-gym
uv sync
```

MuJoCo needs a rendering backend if you want camera images or the
interactive viewer (`MUJOCO_GL=glfw` for on-screen, `egl` or `osmesa` for
headless offscreen rendering). See the
[MuJoCo docs](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)
if `import mujoco` or rendering fails on your machine — this is a MuJoCo
requirement, unrelated to this package.

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
