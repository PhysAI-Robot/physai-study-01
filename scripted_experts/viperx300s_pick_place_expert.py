"""Privileged pick-and-place expert for the ALOHA follower arm (ViperX 300 S).

Research module: registers itself with ``physai.robots.registry`` on import.
Core never imports this module directly (see research/README.md).

The state machine is the SO-101 expert's (`so101_pick_place_expert`); what changes is
numbers and one lookup:

- the stock fingers' pads reach 0.036 m below the `pinch` site, so the pinch point is
  held `grasp_height` above the cube centre, putting the fingertips just above the table;
- the gripper joint is a finger position in metres, so "settled" uses a velocity
  threshold in m/s;
- the contact test looks for the stock `left_finger_pad_*` / `right_finger_pad_*`
  geoms instead of the SO-101's added pads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from physai.robots.registry import register_robot_policy
from physai.robots.viperx300s.kinematics import TOP_DOWN

from .so101_pick_place_expert import ExpertConfig, Phase, SO101PickPlaceExpert

# Phases after the grasp: the cube travels with the fingers, so their targets sit
# `grasp_height` above where the SO-101 expert puts them.
_CARRY_PHASES = (
    Phase.LIFT,
    Phase.TRANSFER,
    Phase.LOWER,
    Phase.RELEASE,
    Phase.RETREAT,
)


@dataclass
class ViperX300SExpertConfig(ExpertConfig):
    approach_dir: np.ndarray | None = field(default_factory=lambda: TOP_DOWN)
    hover_height: float = 0.08
    # Pinch point above the cube centre at the grasp. The fingertip pads end 0.036 m
    # below the pinch site, so 0.026 leaves the tips ~4 mm above the table.
    grasp_height: float = 0.026
    lift_height: float = 0.05
    place_height: float = 0.004
    # Normalised gripper (0 closed .. 1 open). A 28 mm cube first touches the pads at
    # a finger position of 0.0255 m = 0.125 of the stroke; 0 squeezes ~4.5 mm past it.
    gripper_open: float = 1.0
    gripper_touch: float = 0.125
    gripper_grip: float = 0.0
    pos_tol: float = 0.012
    max_phase_steps: int = 150
    max_joint_rate: float = 1.5
    approach_rate: float = 0.8
    gripper_velocity_tol: float = 0.004


class ViperX300SPickPlaceExpert(SO101PickPlaceExpert):
    """Privileged ViperX 300 S policy used to bootstrap visual-control datasets."""

    name = "scripted_pick_place"

    def __init__(self, kin, env, cfg: ViperX300SExpertConfig | None = None) -> None:
        super().__init__(kin, env, cfg or ViperX300SExpertConfig())
        self.last_ik: tuple[bool, float, float] = (True, 0.0, 0.0)

    def _solve(self, target_xyz: np.ndarray) -> np.ndarray:
        """IK for the pinch target; a near-converged solution is used, not dropped.

        The SO-101 expert keeps the previous command when IK has not met its (tight)
        tolerances. With the arm's six joints and the yaw pinned, the solver often
        stops a millimetre short of the orientation tolerance; discarding that
        solution froze the arm centimetres from its target.
        """
        direction = None if self.phase is Phase.RISE else self.cfg.approach_dir
        result = self.kin.ik_pinch(
            target_xyz, direction, q_init=self._q_cmd, max_iters=400
        )
        self.last_ik = (
            result.converged,
            result.position_error,
            result.orientation_error,
        )
        if result.position_error < 0.004 and result.orientation_error < 0.1:
            return result.qpos
        return self._q_cmd

    def _waypoint(self) -> tuple[np.ndarray, float]:
        target, grip = super()._waypoint()
        if self.phase in _CARRY_PHASES:
            target = target + np.array([0.0, 0.0, self.cfg.grasp_height])
        return target, grip

    def _cube_grasped(self) -> bool:
        cube_geom = self.env.cube_geom_id
        model = self.env.model
        pads = {
            geom
            for geom in range(model.ngeom)
            if (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            ).startswith(("left_finger_pad_", "right_finger_pad_"))
        }
        touching = set()
        for index in range(self.env.data.ncon):
            contact = self.env.data.contact[index]
            if cube_geom == contact.geom1 and contact.geom2 in pads:
                touching.add(contact.geom2)
            elif cube_geom == contact.geom2 and contact.geom1 in pads:
                touching.add(contact.geom1)
        left = any(
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                "left"
            )
            for g in touching
        )
        right = any(
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                "right"
            )
            for g in touching
        )
        # Both fingers on the cube: one finger pushing it against the table is not a grasp.
        return left and right


def make_scripted_policy(
    *, env, cfg: Any = None, **_: Any
) -> ViperX300SPickPlaceExpert:
    """Build the deterministic ViperX 300 S pick-and-place expert."""
    return ViperX300SPickPlaceExpert(env.kin, env, cfg=cfg)


register_robot_policy("viperx300s", "scripted", make_scripted_policy)


__all__ = [
    "ViperX300SExpertConfig",
    "ViperX300SPickPlaceExpert",
    "make_scripted_policy",
]
