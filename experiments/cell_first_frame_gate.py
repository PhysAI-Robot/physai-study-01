"""Gate of a physics-by-renderer cell: the first frames of seeds 1000-1004 in three settings, then the check (Appendix, gates of the cells).

    OMNI_KIT_ACCEPT_EULA=YES python research/experiments/cell_first_frame_gate.py dump isaac_own mi_cell/gate_isaac_own.npz
    OMNI_KIT_ACCEPT_EULA=YES python research/experiments/cell_first_frame_gate.py dump cell mi_cell/gate_cell.npz     # --sim mujoco --render isaac
    python research/experiments/cell_first_frame_gate.py dump mujoco mi_cell/gate_mujoco.npz
    python research/experiments/cell_first_frame_gate.py check mi_cell/gate_isaac_own.npz mi_cell/gate_cell.npz mi_cell/gate_mujoco.npz

Gate (fixed before any policy was scored in the cell): cube and target pixels of the front camera agree between the cell and Isaac Sim's own run to 1 px,
and the robot pose agrees with MuJoCo's (qpos of the cell equal to MuJoCo's; the arm silhouette of the cell equal to Isaac Sim's own).
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
SEEDS = range(1000, 1005)
import os

MANIFEST = os.environ.get(
    "GATE_MANIFEST", "research/configs/so101_randomized_pick_place.yaml"
)  # the ViperX manifest, with GATE_NO_SILHOUETTE=1 (its arm is black, the mask is the SO-101's yellow)
NO_SILHOUETTE = bool(os.environ.get("GATE_NO_SILHOUETTE"))

if sys.argv[1] == "dump":
    cell, out = sys.argv[2], sys.argv[3]
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.runtime import create_session

    kw = {
        "isaac_own": dict(simulator="isaac"),
        "cell": dict(simulator="mujoco", renderer="isaac"),
        "mujoco": dict(simulator="mujoco"),
        "im": dict(simulator="isaac", renderer="mujoco"),
    }[cell]  # im: Isaac Sim physics drawn by MuJoCo
    if os.environ.get("GATE_CUBE_START") and cell in (
        "cell",
        "mujoco",
    ):  # the cube start of Isaac Sim's run (table file)
        kw["robot_config"] = {"cube_start": os.environ["GATE_CUBE_START"]}
    manifest = with_overrides(
        load_manifest(MANIFEST), policy="idle", max_steps=10, **kw
    )
    env = create_session(manifest, render=True).runtime.robot
    front, wrist, qpos, cube, quat = [], [], [], [], []
    for seed in SEEDS:
        obs = env.reset(seed=seed)
        for _ in range(
            3
        ):  # the renderer settles (temporal anti-aliasing), as in isaac_frames.py
            obs = env.reset(seed=seed)
        front.append(obs.images["front"].data)
        wrist.append(obs.images["wrist"].data)
        qpos.append(np.asarray(obs.joint_state.position))
        cube.append(np.asarray(env.cube_pos))
        quat.append(
            np.asarray(env._cube_body.get_world_poses()[1])[0]
            if cell in ("isaac_own", "im")
            else np.zeros(4)
        )
    np.savez(
        out,
        front=np.asarray(front),
        wrist=np.asarray(wrist),
        qpos=np.asarray(qpos),
        cube=np.asarray(cube),
        quat=np.asarray(quat),
    )
    env.close()
    print("saved", out)
    raise SystemExit

from research.classical_control.so101_visual_servo import (
    ColorBlobDetector,
    TargetDiscDetector,
)
from research.imitation_learning.image_edits import robot_mask

if (
    sys.argv[1] == "check_im"
):  # IM gate: the cell of Isaac Sim physics drawn by MuJoCo against MuJoCo's own run
    im, mj = (np.load(p) for p in sys.argv[2:4])
    from research.imitation_learning.image_edits import robot_mask
    from research.classical_control.so101_visual_servo import (
        ColorBlobDetector,
        TargetDiscDetector,
    )

    for i, seed in enumerate(SEEDS):
        row = []
        for name, det in (
            ("cube", ColorBlobDetector()),
            ("target", TargetDiscDetector()),
        ):
            a, b = det.detect(mj["front"][i]), det.detect(im["front"][i])
            row.append(
                f"{name} {np.linalg.norm(a.pixel - b.pixel):.2f} px"
                if a and b
                else f"{name} missing"
            )
        ia, ib = robot_mask(mj["front"][i]), robot_mask(im["front"][i])
        print(
            f"seed {seed}: "
            + ", ".join(row)
            + (
                ""
                if NO_SILHOUETTE
                else f", arm silhouette IoU {(ia & ib).sum() / max((ia | ib).sum(), 1):.3f}"
            )
            + f", cube MuJoCo-IM {np.round((mj['cube'][i] - im['cube'][i]) * 1000, 1)} mm"
        )
    raise SystemExit
iso, cell, mj = (np.load(p) for p in sys.argv[2:5])
worst = {"cube": 0.0, "target": 0.0}
for i, seed in enumerate(SEEDS):
    row = []
    for name, det in (("cube", ColorBlobDetector()), ("target", TargetDiscDetector())):
        a, b = det.detect(iso["front"][i]), det.detect(cell["front"][i])
        d = float(np.linalg.norm(a.pixel - b.pixel)) if a and b else float("nan")
        worst[name] = max(worst[name], d)
        row.append(f"{name} {d:.2f} px")
    ia, ib = robot_mask(iso["front"][i]), robot_mask(cell["front"][i])
    iou = float((ia & ib).sum() / max((ia | ib).sum(), 1))
    print(
        f"seed {seed}: "
        + ", ".join(row)
        + ("" if NO_SILHOUETTE else f", arm silhouette IoU {iou:.3f}")
        + f", qpos cell-MuJoCo {np.abs(cell['qpos'][i] - mj['qpos'][i]).max():.2e}, cube cell-MuJoCo {np.abs(cell['cube'][i] - mj['cube'][i]).max():.2e}"
    )
print("worst:", worst)
