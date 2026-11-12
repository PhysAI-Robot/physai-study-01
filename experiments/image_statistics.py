"""Image statistics of the measured picture differences (Appendix, Section "attribution"), from the compare_cameras.py dumps.

    uv run python research/experiments/image_statistics.py MUJOCO.npz ISAAC.npz OUT.json

Definitions (all on one pose: ARM_QPOS of compare_cameras.py, cube at the scene default, reset seed 0):
  - region_means (compare_cameras.py): mean RGB of "background" = upper third of the front image
    without arm/cube pixels, "table/floor" = lower half without arm/cube, "cube" = reddish pixels.
    Ratio = Isaac / MuJoCo of the mean over the three channels; per channel is also listed.
  - by region (image_edits.front_regions): disjoint masks arm / cube / disc / table (inside the table
    polygon projected from the MuJoCo camera calibration) / background (everything else that is not
    arm, cube, disc or very dark). Each image uses its own masks. Gain = per-channel ratio of mean RGB.
  - wrist: whole-image per-channel ratio (what WRIST_GAIN is) and the compare_cameras background (upper third).
  - Laplacian variance: variance of scipy.ndimage.laplace of the grey image (0.299 R + 0.587 G + 0.114 B,
    float), whole image; ratio = Isaac / MuJoCo. `sigma_match` is the Gaussian sigma (grid step 0.05) at which the
    blurred MuJoCo image has Isaac's Laplacian variance.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "scripts")]
import compare_cameras as cc  # noqa: E402
from research.imitation_learning import image_edits as ie  # noqa: E402

mj, isa = np.load(sys.argv[1]), np.load(sys.argv[2])


def table_mask() -> np.ndarray:
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.robots.so101.scene import scene_defaults
    from physai.sim.mujoco import SingleCubePlaceSceneConfig

    scene = SingleCubePlaceSceneConfig(**scene_defaults())
    env = SO101Env(EnvConfig(scene=scene, randomize_cube=False, render=True))
    env.reset(seed=0)
    frame = env.observe().images["front"]
    mask = ie.table_polygon_mask(
        frame.data.shape[:2],
        frame.intrinsics,
        frame.extrinsics,
        scene.table_pos,
        scene.table_size,
    )
    env.close()
    return mask


def grey(image):
    return image.astype(float) @ np.array([0.299, 0.587, 0.114])


def lap_var(image):
    return float(ndimage.laplace(grey(image)).var())


def ratio(a, b):
    return None if a is None or b is None else (np.asarray(b) / np.asarray(a))


def lst(x):
    return None if x is None else [round(float(v), 4) for v in np.atleast_1d(x)]


out = {"definitions": __doc__}
table = table_mask()

# compare_cameras region means (upper third / lower half / cube)
for cam in ("front", "wrist"):
    rm_m, rm_i = cc.region_means(mj[cam]), cc.region_means(isa[cam])
    out[f"{cam}_region_means"] = {
        r: {
            "mujoco": lst(rm_m[r]),
            "isaac": lst(rm_i[r]),
            "ratio_of_mean": None
            if rm_m[r] is None or rm_i[r] is None
            else round(float(rm_i[r].mean() / rm_m[r].mean()), 4),
            "ratio_per_channel": lst(ratio(rm_m[r], rm_i[r])),
        }
        for r in rm_m
    }
out["pixel_difference_outside_arm"] = {
    cam: round(cc.pixel_difference(mj[cam], isa[cam]), 2) for cam in ("front", "wrist")
}

# by region, front
reg_m, reg_i = (
    ie.front_regions(mj["front"], table),
    ie.front_regions(isa["front"], table),
)
by = {}
for name in ("background", "table", "cube", "arm", "disc"):
    mm, ii = reg_m[name], reg_i[name]
    if mm.sum() < 20 or ii.sum() < 20:
        by[name] = {"pixels": [int(mm.sum()), int(ii.sum())], "gain": None}
        continue
    a, b = mj["front"][mm].mean(axis=0), isa["front"][ii].mean(axis=0)
    by[name] = {
        "pixels": [int(mm.sum()), int(ii.sum())],
        "mujoco": lst(a),
        "isaac": lst(b),
        "gain": lst(b / a),
        "ratio_of_mean": round(float(b.mean() / a.mean()), 4),
    }
out["front_by_region"] = by

# wrist whole image
a, b = mj["wrist"].reshape(-1, 3).mean(axis=0), isa["wrist"].reshape(-1, 3).mean(axis=0)
out["wrist_whole_image_gain"] = lst(b / a)
out["wrist_whole_image_ratio_of_mean"] = round(float(b.mean() / a.mean()), 4)

# Laplacian variance
lap = {}
for cam in ("front", "wrist"):
    vm, vi = lap_var(mj[cam]), lap_var(isa[cam])
    sigma_match = None
    for s in np.arange(0.0, 4.0001, 0.05):
        blurred = (
            ndimage.gaussian_filter(mj[cam], sigma=(s, s, 0)) if s > 0 else mj[cam]
        )
        if lap_var(blurred) <= vi:
            sigma_match = round(float(s), 2)
            break
    old = ie.BLUR_SIGMA[cam]
    v_old = lap_var(ndimage.gaussian_filter(mj[cam], sigma=(old, old, 0)))
    lap[cam] = {
        "var_mujoco": vm,
        "var_isaac": vi,
        "ratio_isaac_over_mujoco": vi / vm,
        "sigma_match": sigma_match,
        "old_sigma": old,
        "ratio_given_by_old_sigma": v_old / vm,
        "isaac_over_old_sigma_ratio": vi / v_old,
    }
out["laplacian"] = lap

Path(sys.argv[3]).write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "definitions"}, indent=2))
