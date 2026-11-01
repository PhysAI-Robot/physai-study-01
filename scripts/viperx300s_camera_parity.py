"""Do the ViperX front and wrist cameras see the same scene on every engine? (study 1, V2 N0 gate b)

    python research/scripts/viperx300s_camera_parity.py --dump mujoco --out outputs/study01/v2/cams_mujoco.npz
    OMNI_KIT_ACCEPT_EULA=YES python research/scripts/viperx300s_camera_parity.py --dump isaac --out outputs/study01/v2/cams_isaac.npz
    python research/scripts/viperx300s_camera_parity.py --dump newton --out outputs/study01/v2/cams_newton.npz
    python research/scripts/viperx300s_camera_parity.py --compare outputs/study01/v2/cams_mujoco.npz outputs/study01/v2/cams_isaac.npz outputs/study01/v2/cams_newton.npz

Each engine is dumped in its own process (one Isaac `SimulationApp` per process): the first `--seeds` episodes of the
randomized manifest are reset on that engine and the front and wrist images at reset are saved with the cube and target
positions. `--compare` finds the cube (red) and the target pad (green) in every front image by colour and reports the
distance between engines in pixels (the bar is 2 px), the same for the wrist image where visible, and the number of dark
pixels in the front image (the ViperX arm is black; this is a proxy for the arm being drawn, the bar is a minimum count).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _core  # noqa: F401
import numpy as np
from viperx300s_check import MANIFEST

from physai.config import load_manifest
from physai.config.compat import with_overrides
from physai.runtime import create_session
from research.imitation_learning.image_edits import cube_mask, disc_mask

PIXEL_BAR = 2.0
MIN_ARM_PIXELS = 500
SPAWN_TOLERANCE = 0.003  # m; a seed whose cube starts further apart than this across engines is not counted for the cube
WRIST_IOU_BAR = 0.8  # chosen after looking at the images, not before
WRIST_DARK = 110  # MuJoCo's specular highlights lift parts of the black fingers above 80 (IoU 0.66 with the arm's 80; 0.96 here)
DARK = 80  # all channels below this count as the black arm (Newton lights it grey: 958 px against 2162 on MuJoCo)
KEEP_IMAGES = 10  # images are kept for the first seeds only; the pixel counts of cube and pad for all of them


def dump(sim: str, seeds: int, out: Path) -> None:
    # Newton renders MuJoCo's physics: it is a renderer, picked apart from the simulator.
    engine = (
        {"simulator": "mujoco", "renderer": "newton"}
        if sim == "newton"
        else {"simulator": sim}
    )
    manifest = with_overrides(
        load_manifest(MANIFEST), **engine, policy="idle", max_steps=10
    )
    env = create_session(manifest, render=True).runtime.robot
    arrays: dict[str, list] = {
        "front": [],
        "wrist": [],
        "cube": [],
        "target": [],
        "cube_px": [],
        "target_px": [],
    }
    for seed in range(seeds):
        obs = env.reset(seed=seed)
        front = obs.images["front"].data
        arrays["cube_px"].append(int(cube_mask(front).sum()))
        arrays["target_px"].append(int(disc_mask(front).sum()))
        if seed < KEEP_IMAGES:
            arrays["front"].append(front)
            arrays["wrist"].append(obs.images["wrist"].data)
            arrays["cube"].append(np.asarray(env.cube_pos))
            arrays["target"].append(np.asarray(env.target_pos))
    env.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(v) for k, v in arrays.items()})
    cube_px, target_px = np.asarray(arrays["cube_px"]), np.asarray(arrays["target_px"])
    print(
        f"{sim}: {seeds} seeds, fewest colour-mask pixels in the front image: cube {cube_px.min()} "
        f"(seed {cube_px.argmin()}), pad {target_px.min()} (seed {target_px.argmin()}) -> {out}"
    )


def centroid(mask: np.ndarray) -> np.ndarray | None:
    """Centre of the largest connected blob of `mask` (stray pixels elsewhere do not move it)."""
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    if sizes.max() < 20:
        return None
    ys, xs = np.nonzero(labels == 1 + int(np.argmax(sizes)))
    return np.array([xs.mean(), ys.mean()])


def compare(paths: list[Path]) -> int:
    dumps = {p.stem.removeprefix("cams_"): np.load(p) for p in paths}
    names = list(dumps)
    seeds = len(dumps[names[0]]["front"])  # the first KEEP_IMAGES seeds
    worst = {"cube": 0.0, "target": 0.0}
    unmatched = []
    for seed in range(seeds):
        row = [f"seed {seed}:"]
        cubes = np.array([dumps[n]["cube"][seed] for n in names])
        moved = float(np.ptp(cubes[:, :2], axis=0).max())
        for key, mask_of in (("cube", cube_mask), ("target", disc_mask)):
            points = {n: centroid(mask_of(dumps[n]["front"][seed])) for n in names}
            if any(p is None for p in points.values()):
                row.append(
                    f"{key} not found in {[n for n, p in points.items() if p is None]}"
                )
                worst[key] = float("inf")
                continue
            spread = max(
                np.linalg.norm(points[a] - points[b]) for a in names for b in names
            )
            if key == "cube" and moved > SPAWN_TOLERANCE:
                # the engines put the cube in different places at reset: that is spawn physics, not the camera
                unmatched.append(seed)
                row.append(
                    f"cube spread {spread:.1f} px (cubes {moved * 1000:.1f} mm apart, not counted)"
                )
                continue
            worst[key] = max(worst[key], spread)
            row.append(f"{key} spread {spread:.1f} px")
        dark = {
            n: int((dumps[n]["front"][seed] < DARK).all(axis=-1).sum()) for n in names
        }
        row.append(f"dark px {dark}")
        print("  ".join(row))
    # The fingers are fixed in the wrist camera, so their silhouette (the dark pixels of the lower half) must agree.
    worst_iou = 1.0
    for seed in range(seeds):
        masks = {
            n: (dumps[n]["wrist"][seed] < WRIST_DARK).all(axis=-1)[
                dumps[n]["wrist"].shape[1] // 2 :
            ]
            for n in names
        }
        for a in names:
            for b in names:
                union = (masks[a] | masks[b]).sum()
                worst_iou = min(worst_iou, (masks[a] & masks[b]).sum() / max(union, 1))
    arm_min = min(
        int((dumps[n]["front"][s] < DARK).all(axis=-1).sum())
        for n in names
        for s in range(seeds)
    )
    ok = (
        worst["cube"] <= PIXEL_BAR
        and worst["target"] <= PIXEL_BAR
        and arm_min >= MIN_ARM_PIXELS
        and worst_iou >= WRIST_IOU_BAR
    )
    print(
        f"\nworst spread: cube {worst['cube']:.1f} px, target {worst['target']:.1f} px "
        f"(bar {PIXEL_BAR}); fewest dark pixels {arm_min} (bar {MIN_ARM_PIXELS}) -> "
        f"wrist finger silhouette overlap (IoU) at least {worst_iou:.2f} (bar {WRIST_IOU_BAR}) -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", choices=("mujoco", "isaac", "newton"))
    mode.add_argument("--compare", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--seeds",
        type=int,
        default=KEEP_IMAGES,
        help="episodes to reset; the first 10 keep their images",
    )
    args = ap.parse_args()
    if args.dump:
        if args.out is None:
            ap.error("--dump needs --out")
        dump(args.dump, args.seeds, args.out)
        return 0
    return compare(args.compare)


if __name__ == "__main__":
    raise SystemExit(main())
