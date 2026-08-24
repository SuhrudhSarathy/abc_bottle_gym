"""Scene, randomization, and task-metric configuration for the put-bottles-in-bin task.

Ported from the ABC project's ``abc_minimal/config.py`` (``PutBottlesSimConfig``),
which has no dependency on torch/MJWarp and is reproduced here unmodified in
substance so scene generation and the success/reward metric match the
original exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PutBottlesSimConfig:
    """Scene, randomization, and task metric defaults for the put-bottles task."""

    gripper_ctrl_max: float = 0.0475
    bottle_count: int = 6
    init_q: tuple[float, ...] = (
        0.0, 1.047, 1.047, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.047, 1.047, 0.0, 0.0, 0.0, 0.0,
    )
    timestep: float = 0.002
    control_decimation: int = 17

    table_z: float = 0.75
    table_bounds: tuple[float, float, float, float] = (0.3025, 0.8975, -0.65, 0.65)
    bottle_spawn_clearance: float = 0.005
    bottle_sample_attempts: int = 200
    bottle_collision_margin: float = 0.04
    bottle_scale_range: tuple[float, float] = (0.9, 1.1)
    bottle_side_radii: tuple[float, ...] = (0.025667, 0.024014, 0.020589, 0.026359, 0.023689, 0.021823)
    bottle_flat_lengths: tuple[float, ...] = (0.166718, 0.165000, 0.156531, 0.160000, 0.166689, 0.159200)
    bottle_flat_half_widths: tuple[float, ...] = (0.025672, 0.024013, 0.020566, 0.025957, 0.023689, 0.021823)

    bin_scale_range: tuple[float, float] = (0.95, 1.05)
    bin_yaw_range: tuple[float, float] = (-0.75, 0.75)
    bin_xy_range: tuple[float, float, float, float] = (0.57, 0.73, -0.25, 0.25)
    bin_z_scale: float = 0.83
    bin_occupied_radius: float = 0.13
    bin_base_quat: tuple[float, float, float, float] = (0.70710678, 0.70710678, 0.0, 0.0)

    eval_bin_radius: float = 0.155
    eval_min_rel_z: float = -0.06
    eval_max_rel_z: float = 0.26
