"""Count the pixels of the cube and the target pad in the ViperX cameras at episode start (MuJoCo).

    python research/scripts/viperx300s_visibility.py --seeds 1300
    python research/scripts/viperx300s_visibility.py --seeds 100 --start 900

Each object is counted by hiding it (alpha 0) and diffing the render, as for the SO-101 in study 1 M0.
Writes outputs/viperx300s/visibility.json (git-ignored) and prints the smallest counts per camera.
"""

from __future__ import annotations

import argparse
import json

import _core  # noqa: F401
import mujoco
import numpy as np
from viperx300s_check import OUT, make_env

THRESHOLD = 12  # per-channel difference that counts as changed


def hidden_diff(env, camera: str, geom: int, base: np.ndarray) -> int:
    alpha = env.model.geom_rgba[geom, 3]
    env.model.geom_rgba[geom, 3] = 0.0
    try:
        hidden = env.render_camera(camera)
    finally:
        env.model.geom_rgba[geom, 3] = alpha
    return int(
        (np.abs(base.astype(int) - hidden.astype(int)).max(axis=-1) > THRESHOLD).sum()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1300)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    env = make_env(render=True)
    geoms = {
        name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("cube_geom", "target_pad")
    }
    rows = []
    for seed in range(args.start, args.start + args.seeds):
        env.reset(seed=seed)
        row = {"seed": seed}
        for camera in ("front", "wrist"):
            base = env.render_camera(camera)
            for key, geom in geoms.items():
                row[f"{camera}_{key}"] = hidden_diff(env, camera, geom, base)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "visibility.json").write_text(json.dumps(rows), encoding="utf-8")
    for key in rows[0]:
        if key == "seed":
            continue
        counts = np.array([r[key] for r in rows])
        print(
            f"{key}: min {counts.min()} (seed {rows[counts.argmin()]['seed']}), "
            f"seeds with fewer than 20 px: {(counts < 20).sum()} of {len(rows)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
