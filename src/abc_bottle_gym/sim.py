"""MJWarp-backed physics + rendering, ported from the ABC project's
``abc_minimal/eval_policy.py`` (``MJWarpSim``, ``require_mjwarp``).

This is the same GPU-batched (``nworld=1``) MuJoCo-Warp simulator the
upstream project uses by default for policy training/eval -- both physics
stepping (``mjw.step``) and camera rendering happen on the GPU here, so
observations match what a checkpoint trained against the original pipeline
actually saw. This requires an NVIDIA GPU with a working CUDA driver;
there is no CPU fallback (native ``mujoco`` rendering/physics look and,
in the case of physics-adjacent numerics, behave differently enough from
MJWarp to be a real distribution-shift risk for a policy trained on MJWarp
rollouts).
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def require_mjwarp() -> None:
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "abc_bottle_gym requires mujoco_warp and warp (an NVIDIA CUDA GPU). "
            "Install them (they're regular pyproject dependencies -- `uv sync` "
            "should already have pulled them in) and make sure a CUDA-capable "
            "GPU/driver is visible, e.g. `nvidia-smi`."
        ) from exc


class MJWarpSim:
    """Single-world MJWarp physics + render wrapper around a mujoco.MjModel/MjData."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        height: int,
        width: int,
        gpu_id: int | None,
    ):
        require_mjwarp()
        import mujoco_warp as mjw
        import warp as wp

        self.mjw = mjw
        self.wp = wp
        self.model = model
        self.data = data
        self.height = height
        self.width = width
        self.nworld = 1
        if gpu_id is not None:
            wp.set_device(f"cuda:{gpu_id}")
        self.m_warp = mjw.put_model(model)
        self.d_warp = mjw.put_data(model, data, nworld=self.nworld, nconmax=model.nconmax, njmax=model.njmax)
        self.render_context = mjw.create_render_context(
            mjm=model,
            nworld=self.nworld,
            cam_res=(width, height),
            render_rgb=[True] * model.ncam,
            render_depth=[False] * model.ncam,
            use_textures=True,
            use_shadows=True,
        )
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.wp.synchronize()
        except Exception:
            pass
        self.render_context = None
        self.m_warp = None
        self.d_warp = None
        self.model = None
        self.data = None
        self.closed = True

    def _copy(self, target: Any, values: np.ndarray, dtype: Any) -> None:
        self.wp.copy(target, self.wp.from_numpy(np.asarray(values), dtype=dtype))

    def load_state(self) -> None:
        self.mjw.reset_data(self.m_warp, self.d_warp)
        self._copy(self.d_warp.qpos, np.asarray(self.data.qpos, dtype=np.float32)[None], self.wp.float32)
        if self.model.nv > 0:
            self._copy(self.d_warp.qvel, np.asarray(self.data.qvel, dtype=np.float32)[None], self.wp.float32)
        if self.model.nu > 0:
            self._copy(self.d_warp.ctrl, np.asarray(self.data.ctrl, dtype=np.float32)[None], self.wp.float32)
        if self.model.na > 0 and hasattr(self.d_warp, "act"):
            self._copy(self.d_warp.act, np.asarray(self.data.act, dtype=np.float32)[None], self.wp.float32)
        if self.model.nmocap > 0:
            self._copy(self.d_warp.mocap_pos, np.asarray(self.data.mocap_pos, dtype=np.float32)[None], self.wp.vec3f)
            self._copy(self.d_warp.mocap_quat, np.asarray(self.data.mocap_quat, dtype=np.float32)[None], self.wp.quatf)
        if hasattr(self.d_warp, "time"):
            self._copy(self.d_warp.time, np.asarray([self.data.time], dtype=np.float32), self.wp.float32)

    def forward(self) -> None:
        self.mjw.forward(self.m_warp, self.d_warp)

    def qpos(self) -> np.ndarray:
        return self.d_warp.qpos.numpy()[0].copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        ctrl = np.asarray(ctrl, dtype=np.float32)
        if ctrl.shape != (self.model.nu,):
            raise ValueError(f"Expected ctrl shape {(self.model.nu,)}, got {ctrl.shape}")
        self._copy(self.d_warp.ctrl, ctrl[None], self.wp.float32)

    def step(self, nstep: int) -> None:
        for _ in range(nstep):
            self.mjw.step(self.m_warp, self.d_warp)

    def render(self) -> np.ndarray:
        self.mjw.refit_bvh(self.m_warp, self.d_warp, self.render_context)
        self.mjw.render(self.m_warp, self.d_warp, self.render_context)
        rgba = self.render_context.rgb_data.numpy().view(np.uint8).reshape(
            self.nworld,
            self.model.ncam,
            self.height,
            self.width,
            4,
        )
        return rgba[0, :, :, :, :3][..., ::-1].copy()
