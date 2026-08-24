"""Gymnasium environment for the ABC put-bottles-in-bin MuJoCo task.

This is a standalone port of ``PutBottlesEnv`` from the ABC project's
``abc_minimal/eval_policy.py``. It reproduces the original task exactly:

- Same scene XML, same per-episode randomization (bottle/bin scale, pose
  sampling), driven by the same ``numpy.random.Generator(seed)`` logic.
- Same physics: ``mujoco.mj_step`` at ``timestep=0.002`` with
  ``control_decimation=17`` steps per action -- identical to what the
  original calls its ``--vanilla-physics`` path, which the upstream project
  itself documents as the preferred (and physically equivalent) choice for
  single, non-batched environments.
- Same 14-dim state/action layout (6 arm joints + 1 gripper, per arm) and the
  same success/reward metric (fraction of bottles inside the bin volume).

The one intentional difference: camera frames are rendered with MuJoCo's
built-in ``mujoco.Renderer`` rather than the GPU-batched MJWarp renderer the
original uses for large-scale policy training. MJWarp/``warp-lang``/``viser``
are CUDA-only, multi-hundred-MB dependencies that add nothing to testing the
task itself, so they were dropped; scene geometry and camera placement are
unchanged, so rendered views are visually equivalent, not bit-identical.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from abc_bottle_gym.config import PutBottlesSimConfig
from abc_bottle_gym.scene import (
    PutBottlesEvaluator,
    _quat_mul,
    _quat_yaw,
    sample_bottle_pose,
    scene_xml,
)

# Joint ranges copied from put_bottle.xml (dm4340: joints 1-3, dm4310: joints 4-6).
# Position actuators use inheritrange="1", so ctrlrange == joint range exactly.
_ARM_JOINT_LOW = np.array([-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944], dtype=np.float32)
_ARM_JOINT_HIGH = np.array([3.05433, 3.66519, 3.66519, 1.5708, 1.5708, 2.0944], dtype=np.float32)
_ONE_ARM_LOW = np.concatenate([_ARM_JOINT_LOW, [0.0]]).astype(np.float32)
_ONE_ARM_HIGH = np.concatenate([_ARM_JOINT_HIGH, [1.0]]).astype(np.float32)

# Action/state layout: [left_j1..j6, left_gripper, right_j1..j6, right_gripper].
ACTION_LOW = np.concatenate([_ONE_ARM_LOW, _ONE_ARM_LOW]).astype(np.float32)
ACTION_HIGH = np.concatenate([_ONE_ARM_HIGH, _ONE_ARM_HIGH]).astype(np.float32)

DEFAULT_CAMERA_KEYS = ("top", "left", "right")
ALL_CAMERA_KEYS = ("top", "overhead", "left_side", "right_side", "left", "right")
DEFAULT_PROMPT = "sim put the plastic bottles in the bin"


class PutBottlesEnv(gym.Env):
    """Bimanual YAM arms put loose bottles into a bin (MuJoCo, Gymnasium API)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
        include_images: bool = False,
        height: int = 168,
        width: int = 224,
        max_episode_steps: int = 1800,
        scene: PutBottlesSimConfig | None = None,
        prompt: str = DEFAULT_PROMPT,
    ):
        super().__init__()
        for cam in camera_keys:
            if cam not in ALL_CAMERA_KEYS:
                raise ValueError(f"Unknown camera {cam!r}; choose from {ALL_CAMERA_KEYS}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unknown render_mode {render_mode!r}")

        self.render_mode = render_mode
        self.camera_keys = tuple(camera_keys)
        self.include_images = include_images or render_mode == "rgb_array"
        self.height = height
        self.width = width
        self.max_episode_steps = max_episode_steps
        self.scene = scene or PutBottlesSimConfig()
        self.prompt = prompt

        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        obs_spaces: dict[str, Any] = {
            "state": spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        }
        if self.include_images:
            obs_spaces["images"] = spaces.Dict(
                {
                    cam: spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8)
                    for cam in self.camera_keys
                }
            )
        self.observation_space = spaces.Dict(obs_spaces)

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.evaluator: PutBottlesEvaluator | None = None
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None
        self.qpos_indices: list[int] = []
        self.ctrl_indices: list[int] = []
        self.gripper_state_indices: set[int] = set()
        self.randomization: dict[str, Any] | None = None
        self._elapsed_steps = 0
        self._episode_seed: int | None = None

    # -- construction -----------------------------------------------------

    def _bind(self, xml: str) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = self.scene.timestep
        self.data = mujoco.MjData(self.model)
        self.evaluator = PutBottlesEvaluator(self.model, self.scene)

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.include_images:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        self.qpos_indices, self.ctrl_indices, self.gripper_state_indices = [], [], set()
        idx = 0
        for robot in ("left", "right"):
            for j in range(1, 7):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}_joint{j}")
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}_joint{j}")
                self.qpos_indices.append(int(self.model.jnt_qposadr[jid]))
                self.ctrl_indices.append(aid)
                idx += 1
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}_left_finger")
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}_gripper")
            self.qpos_indices.append(int(self.model.jnt_qposadr[jid]))
            self.ctrl_indices.append(aid)
            self.gripper_state_indices.add(idx)
            idx += 1

    def _set_freejoint(self, name: str, pos: list[float], quat: list[float]) -> None:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = int(self.model.jnt_qposadr[jid])
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat

    def _set_state(self, state: np.ndarray) -> None:
        for i, qpos_idx in enumerate(self.qpos_indices):
            val = float(state[i])
            if i in self.gripper_state_indices:
                val *= self.scene.gripper_ctrl_max
            self.data.qpos[qpos_idx] = val

    # -- gym API ------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        elif self._episode_seed is None:
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        else:
            self._episode_seed += 1
        rng = np.random.default_rng(self._episode_seed)

        scene = self.scene
        bottle_scales = rng.uniform(*scene.bottle_scale_range, size=scene.bottle_count).astype(np.float32)
        bin_scale = float(rng.uniform(*scene.bin_scale_range))
        self._bind(scene_xml(scene, bottle_scales, bin_scale))
        mujoco.mj_resetData(self.model, self.data)
        self._set_state(np.asarray(scene.init_q, dtype=np.float32))

        bin_yaw = float(rng.uniform(*scene.bin_yaw_range))
        bin_x0, bin_x1, bin_y0, bin_y1 = scene.bin_xy_range
        bin_pos = [
            float(rng.uniform(bin_x0, bin_x1)),
            float(rng.uniform(bin_y0, bin_y1)),
            float(scene.bin_z_scale * bin_scale),
        ]
        bin_quat = _quat_mul(_quat_yaw(bin_yaw), np.asarray(scene.bin_base_quat, dtype=np.float64))
        self._set_freejoint("bin_joint", bin_pos, bin_quat.tolist())

        occupied: list[tuple[np.ndarray, float]] = [
            (np.asarray(bin_pos[:2]), scene.bin_occupied_radius * bin_scale)
        ]
        bottle_states: dict[str, Any] = {}
        for index in range(scene.bottle_count):
            pos, quat, center, radius = sample_bottle_pose(
                rng, scene, index, float(bottle_scales[index]), occupied
            )
            name = f"bottle_{index + 1}_joint"
            self._set_freejoint(name, pos, quat.tolist())
            bottle_states[name] = {"pos": pos, "quat": quat.tolist(), "scale": [float(bottle_scales[index])]}
            occupied.append((center, radius))

        mujoco.mj_forward(self.model, self.data)
        self.evaluator.reset()
        self.randomization = {
            "seed": self._episode_seed,
            "bottle_states": bottle_states,
            "bin_state": {"pos": bin_pos, "quat": bin_quat.tolist(), "yaw": [bin_yaw]},
            "bottle_scales": bottle_scales.tolist(),
            "bin_scale": bin_scale,
        }
        self._elapsed_steps = 0

        if self.render_mode == "human":
            self._reset_human_viewer()

        obs = self._obs()
        info = {"randomization": self.randomization, **self.evaluator.evaluate(self.data.qpos)}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action shape {action.shape} != {self.action_space.shape}")
        self.data.ctrl[:] = self.action_to_ctrl(action)
        for _ in range(self.scene.control_decimation):
            mujoco.mj_step(self.model, self.data)
        self._elapsed_steps += 1

        task_eval = self.evaluator.evaluate(self.data.qpos)
        obs = self._obs()
        reward = float(task_eval["reward"])
        terminated = bool(task_eval["ever_success"])
        truncated = self._elapsed_steps >= self.max_episode_steps

        if self.render_mode == "human":
            self._sync_human_viewer()

        return obs, reward, terminated, truncated, dict(task_eval)

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            frames = [self.render_cameras()[cam].transpose(1, 2, 0) for cam in self.camera_keys]
            return np.concatenate(frames, axis=1)
        if self.render_mode == "human":
            self._sync_human_viewer()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # -- task-specific helpers ------------------------------------------------

    def get_state(self) -> np.ndarray:
        state = np.asarray(self.data.qpos[self.qpos_indices], dtype=np.float32).copy()
        for i in self.gripper_state_indices:
            state[i] = float(np.clip(state[i] / self.scene.gripper_ctrl_max, 0.0, 1.0))
        return state

    def action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for i, act_id in enumerate(self.ctrl_indices):
            val = float(action[i])
            if i in self.gripper_state_indices:
                val *= self.scene.gripper_ctrl_max
            ctrl[act_id] = val
        return ctrl

    def render_cameras(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        images = {}
        for name in self.camera_keys:
            self._renderer.update_scene(self.data, camera=name)
            images[name] = self._renderer.render().transpose(2, 0, 1).copy()
        return images

    def evaluate(self) -> dict[str, Any]:
        return self.evaluator.evaluate(self.data.qpos)

    def _obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {"state": self.get_state()}
        if self.include_images:
            obs["images"] = self.render_cameras()
        return obs

    # -- interactive viewer -------------------------------------------------

    def _reset_human_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        import mujoco.viewer

        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _sync_human_viewer(self) -> None:
        if self._viewer is None:
            self._reset_human_viewer()
        self._viewer.sync()
