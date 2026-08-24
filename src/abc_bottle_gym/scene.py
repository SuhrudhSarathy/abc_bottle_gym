"""Scene-XML construction, pose sampling, and task evaluation.

Ported from the ABC project's ``abc_minimal/eval_policy.py``. The XML/pose
math (``scene_xml``, ``sample_bottle_pose``, the quaternion helpers) and the
``PutBottlesEvaluator`` success/reward metric are reproduced unmodified in
substance so a given ``seed`` reproduces the exact same scene and a given
``qpos`` trajectory scores the exact same reward as the original.

Only MJWarp-specific plumbing was dropped -- this module talks to a plain
``mujoco.MjModel``/``mujoco.MjData``.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from abc_bottle_gym.config import PutBottlesSimConfig

ASSETS_ROOT = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSETS_ROOT / "put_bottle.xml"


def _fmt(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _flat_bottle_quat(yaw: float) -> np.ndarray:
    flat = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0], dtype=np.float64)
    q = _quat_mul(_quat_yaw(yaw), flat)
    return q / np.linalg.norm(q)


def bottle_spawn_z(scene: PutBottlesSimConfig, index: int, scale: float = 1.0) -> float:
    return float(
        scene.table_z + scene.bottle_side_radii[index] * scale + scene.bottle_spawn_clearance
    )


def bottle_xy_footprint(
    scene: PutBottlesSimConfig,
    index: int,
    scale: float,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    length = float(scene.bottle_flat_lengths[index] * scale)
    half_width = float(scene.bottle_flat_half_widths[index] * scale)
    corners = np.array(
        [[0.0, -half_width], [0.0, half_width], [length, -half_width], [length, half_width]],
        dtype=np.float64,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    offsets = corners @ rot.T
    return offsets.min(axis=0), offsets.max(axis=0)


def sample_bottle_pose(
    rng: np.random.Generator,
    scene: PutBottlesSimConfig,
    index: int,
    scale: float,
    occupied: list[tuple[np.ndarray, float]],
) -> tuple[list[float], np.ndarray, np.ndarray, float]:
    candidate = None
    for _ in range(scene.bottle_sample_attempts):
        yaw = float(rng.uniform(-math.pi, math.pi))
        xy_min, xy_max = bottle_xy_footprint(scene, index, scale, yaw)
        table_x0, table_x1, table_y0, table_y1 = scene.table_bounds
        x_low, x_high = table_x0 - xy_min[0], table_x1 - xy_max[0]
        y_low, y_high = table_y0 - xy_min[1], table_y1 - xy_max[1]
        if x_low > x_high or y_low > y_high:
            continue
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_high))
        center = np.array([x, y], dtype=np.float64) + 0.5 * (xy_min + xy_max)
        radius = float(0.5 * np.linalg.norm(xy_max - xy_min))
        pos = [x, y, bottle_spawn_z(scene, index, scale)]
        quat = _flat_bottle_quat(yaw)
        candidate = (pos, quat, center, radius)
        if all(
            np.linalg.norm(center - c) > (radius + r + scene.bottle_collision_margin)
            for c, r in occupied
        ):
            return candidate
    if candidate is None:
        raise RuntimeError("Could not sample a bottle pose inside the table bounds")
    return candidate


def scene_xml(scene: PutBottlesSimConfig, bottle_scales: np.ndarray, bin_scale: float) -> str:
    root = ET.fromstring(SCENE_XML.read_text())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((ASSETS_ROOT / "assets").resolve()))
        compiler.set("texturedir", str((ASSETS_ROOT / "assets").resolve()))
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name", "")
        scale = np.asarray([float(v) for v in mesh.get("scale", "1 1 1").split()], dtype=np.float64)
        for idx in range(scene.bottle_count):
            if name.startswith(f"bottle_{idx}_"):
                mesh.set("scale", _fmt(scale * float(bottle_scales[idx])))
                break
        if name.startswith("water_bottle_"):
            mesh.set("scale", _fmt(scale * float(bin_scale)))
    return ET.tostring(root, encoding="unicode")


class PutBottlesEvaluator:
    """Success/reward metric: fraction of bottles resting inside the bin volume."""

    def __init__(self, model: mujoco.MjModel, scene: PutBottlesSimConfig):
        self.model = model
        self.scene = scene
        self.bottle_names, self.bottle_qpos_addrs = self._bottle_addrs()
        self.bin_qpos_adr = self._joint_qpos_adr("bin_joint")
        self.max_bottles = 0
        self.ever_success = False

    def reset(self) -> None:
        self.max_bottles = 0
        self.ever_success = False

    def _joint_qpos_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _bottle_addrs(self) -> tuple[list[str], np.ndarray]:
        entries = []
        for joint_id in range(self.model.njnt):
            name = self.model.jnt(joint_id).name
            match = re.fullmatch(r"bottle_(\d+)_joint", name or "")
            if match:
                idx = int(match.group(1))
                entries.append((idx, f"bottle_{idx}", int(self.model.jnt_qposadr[joint_id])))
        entries.sort()
        return [e[1] for e in entries], np.asarray([e[2] for e in entries], dtype=np.int32)

    def evaluate(self, qpos: np.ndarray) -> dict[str, Any]:
        qpos = np.asarray(qpos, dtype=np.float32)
        bin_pos = qpos[self.bin_qpos_adr : self.bin_qpos_adr + 3]
        bottle_pos = np.stack([qpos[adr : adr + 3] for adr in self.bottle_qpos_addrs])
        rel = bottle_pos - bin_pos[None]
        radial = np.linalg.norm(rel[:, :2], axis=1)
        radial_margin = self.scene.eval_bin_radius - radial
        lower_height_margin = rel[:, 2] - self.scene.eval_min_rel_z
        upper_height_margin = self.scene.eval_max_rel_z - rel[:, 2]
        in_bin = (radial_margin >= 0.0) & (lower_height_margin >= 0.0) & (upper_height_margin >= 0.0)
        num = int(in_bin.sum())
        active = len(self.bottle_names)
        self.max_bottles = max(self.max_bottles, num)
        success = num == active
        self.ever_success = self.ever_success or success
        return {
            "reward": float(num / max(active, 1)),
            "success": bool(success),
            "ever_success": bool(self.ever_success),
            "num_bottles_in_bin": num,
            "num_active_bottles": active,
            "max_bottles_in_bin_so_far": int(self.max_bottles),
            "bottle_in_bin_mask": [bool(x) for x in in_bin.tolist()],
            "bottles_in_bin": [name for name, ok in zip(self.bottle_names, in_bin) if bool(ok)],
            "bottle_names": list(self.bottle_names),
            "closest_radial_margin": float(radial_margin.max()),
            "closest_height_margin": float(lower_height_margin.max()),
            "closest_upper_height_margin": float(upper_height_margin.max()),
        }
