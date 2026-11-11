"""Q2b: how fast the gripper reading reaches its command, per engine (one process per engine).

    uv run python research/experiments/q2b_grip.py mujoco OUT.json
    OMNI_KIT_ACCEPT_EULA=YES uv run python research/experiments/q2b_grip.py isaac OUT.json

reset(seed=0), arm held at its reset joints, gripper command held at 1.0 for 40 steps, then
0.0 (and in a second run 0.5 -> 0.0 etc.) for 300 steps; the reading is observe().joint_state.position[-1].
Records the reading per control step and the first step at which it is within 0.02 rad of the
joint value of the command, and the largest per-step change.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from physai.contracts import Action, GripperCommand  # noqa: E402

sim, out = sys.argv[1], Path(sys.argv[2])
if sim == "mujoco":
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.robots.so101.scene import scene_defaults
    from physai.sim.mujoco import SingleCubePlaceSceneConfig

    env = SO101Env(
        EnvConfig(
            scene=SingleCubePlaceSceneConfig(**scene_defaults()),
            randomize_cube=False,
            render=False,
        )
    )
else:
    from physai.robots.so101.isaac_env import IsaacEnvConfig, SO101IsaacEnv
    from physai.sim.mujoco import SingleCubePlaceSceneConfig

    env = SO101IsaacEnv(
        IsaacEnvConfig(
            scene=SingleCubePlaceSceneConfig(),
            randomize_cube=False,
            cameras=("front", "wrist"),
            render=True,
        )
    )


def run(start: float, goal: float, hold: int = 40, steps: int = 300) -> dict:
    obs = env.reset(seed=0)
    arm = np.asarray(obs.joint_state.position, dtype=float)[:-1]
    reads = []
    for command, n in ((start, hold), (goal, steps)):
        for _ in range(n):
            obs, *_ = env.step(
                Action(joint_position=arm, gripper=GripperCommand(position=command))
            )
            reads.append(float(np.asarray(obs.joint_state.position)[-1]))
    reads = np.asarray(reads)
    after = reads[hold:]
    target = env.gripper_to_joint(goal)
    close = np.nonzero(np.abs(after - target) < 0.02)[0]
    return {
        "start_command": start,
        "goal_command": goal,
        "goal_joint": float(target),
        "reading_before": float(reads[hold - 1]),
        "reading_end": float(after[-1]),
        "steps_to_within_0.02": int(close[0]) + 1 if len(close) else None,
        "max_step_change_rad": float(np.abs(np.diff(reads[hold - 1 :])).max()),
        "first_10_changes": [float(v) for v in np.diff(reads[hold - 1 :])[:10]],
        "reading_after_20_steps": float(after[19]),
    }


result = {"sim": sim, "runs": [run(1.0, 0.0), run(0.0, 1.0), run(1.0, 0.5)]}
env.close()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
