"""Trace one expert episode around the grasp: where the fingers are, what touches the cube.

python research/scripts/viperx300s_diag.py            # trace, then a tracking-error comparison
"""

from __future__ import annotations

import argparse

import _core  # noqa: F401
import mujoco
import numpy as np
from viperx300s_check import make_env

from research.scripted_experts.viperx300s_pick_place_expert import (
    ViperX300SExpertConfig,
    ViperX300SPickPlaceExpert,
)


def run(seed, every, steps, tweak=None, label="", quiet=False):
    env = make_env(render=False)
    robot = env.robot
    model, data = robot.model, robot.data
    if tweak:
        tweak(robot)
    obs = env.reset(seed=seed)
    policy = ViperX300SPickPlaceExpert(env.kin, env, ViperX300SExpertConfig())
    policy.reset(obs)

    def name(g):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"

    cube_geom = robot.cube_geom_id
    errs = []
    for step in range(1, steps + 1):
        obs, *_ = env.step(policy.act(obs))
        pinch_err = np.linalg.norm(robot.pinch_center() - policy._waypoint()[0]) * 1000
        errs.append((step, policy.phase.name, pinch_err))
        if quiet or step % every:
            continue
        rot = data.site_xmat[robot.kin.site_id].reshape(3, 3)
        rel = rot.T @ (robot.cube_pos - robot.pinch_center())
        touching = sorted(
            {
                name(c.geom2 if c.geom1 == cube_geom else c.geom1)
                for c in (data.contact[i] for i in range(data.ncon))
                if cube_geom in (c.geom1, c.geom2)
            }
        )
        lag = (data.ctrl[robot.arm_act_ids] - data.qpos[robot.arm_qadr]) * 1000
        print(
            f"{step:4d} {policy.phase.name:8s} grip_q={data.qpos[robot.grip_qadr]:.4f} "
            f"cube_in_gripper(mm)={np.round(rel * 1000, 1).tolist()} "
            f"cube_z={robot.cube_pos[2]:.4f} pinch_err={pinch_err:.1f}mm "
            f"lag_mrad={np.round(lag).astype(int).tolist()} "
            f"act_force={np.round(data.actuator_force[robot.arm_act_ids], 1).tolist()} "
            f"ik={policy.last_ik[0]},{policy.last_ik[1] * 1000:.1f},{policy.last_ik[2]:.3f} "
            f"touch={touching}"
        )
    phases = {}
    for step, phase, err in errs:
        phases.setdefault(phase, []).append(err)
    print(
        f"[{label}] final phase {policy.phase.name}; median pinch error (mm) by phase: "
        + ", ".join(f"{p}={np.median(v):.1f}" for p, v in phases.items())
        + f"; cube z {robot.cube_pos[2]:.3f}"
    )
    env.close()


def no_friction(robot):
    robot.model.dof_frictionloss[robot.arm_vadr] = 0.0


def stiff(robot, k=4.0):
    no_friction(robot)
    ids = robot.arm_act_ids
    robot.model.actuator_gainprm[ids, 0] *= k
    robot.model.actuator_biasprm[ids, 1] *= k
    robot.model.dof_damping[robot.arm_vadr] *= np.sqrt(k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--steps", type=int, default=330)
    args = ap.parse_args()
    run(args.seed, args.every, args.steps, None, "as shipped")
    print("---- comparisons (quiet) ----")
    run(args.seed, args.every, args.steps, no_friction, "no joint friction", True)
    run(args.seed, args.every, args.steps, stiff, "no friction, 4x stiffer", True)
    run(
        args.seed,
        args.every,
        args.steps,
        lambda r: stiff(r, 10.0),
        "no friction, 10x stiffer",
        True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
