"""Do two renderers draw the same scene? A check to run before scores from two engines are compared (study 1, V2 N10).

    python research/scripts/render_parity.py poses --manifest M --out poses.npz
    python research/scripts/render_parity.py dump  --manifest M --poses poses.npz --engine mujoco|newton|newton:old|isaac --out E.npz
    python research/scripts/render_parity.py compare A.npz B.npz [--truth-old OLD.npz --truth-new NEW.npz]

`poses` saves the full MuJoCo state at the HOME pose (the first frame of an episode) and at a grasp pose (the first step at which
the scripted expert has lifted the cube). `dump` draws both poses with one renderer, in front and wrist images, from the same state
(the physics is MuJoCo's; the renderer is `--render`). Isaac Sim goes in its own process. `compare` reports, for each pose and camera:
(a) the mask of local differences, (b) the brightness ratio of three regions, (c) the flagged regions. Verdict: "defect" when a
flagged region is larger than FLAG_AREA pixels, else "global shift" when a region ratio is outside RATIO_BAND, else "same".

A local difference is the brightness ratio of the two images minus its own smooth version (a median over a WINDOW px window), so a
change of brightness or colour that is smooth over the picture, which every pair of engines has, is not flagged and an object that
one renderer draws and the other does not is. The constants below were fixed in the study file before the first run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _core  # noqa: F401
import numpy as np

WINDOW = 31  # px; the smooth part of the log brightness ratio (larger than any object it must keep)
LOCAL_RATIO = (
    1.25  # a pixel is local when its ratio differs from the smooth one by this factor
)
MIN_REGION = 8  # px; connected flagged pixels below this are not listed (edge aliasing)
FLAG_AREA = 50  # px; a flagged region above this makes the verdict "defect"
RATIO_BAND = (
    0.9,
    1.1,
)  # a region whose mean brightness ratio is outside this is a global shift
EPS = 8.0  # grey levels added before the ratio so that black does not blow it up
CAMERAS = ("front", "wrist")
POSES = ("home", "grasp")


def _manifest(path: str, **kw):
    from physai.config import load_manifest
    from physai.config.compat import with_overrides

    return with_overrides(load_manifest(path), policy="idle", max_steps=10, **kw)


def make_poses(manifest: str, seed: int, out: Path) -> None:
    from physai.policy import create_policy
    from physai.runtime import create_session

    env = create_session(_manifest(manifest), render=False).runtime.robot
    obs = env.reset(seed=seed)
    qpos = {"home": env.data.qpos.copy()}
    z0 = float(env.cube_pos[2])
    policy = create_policy("scripted", env=env)
    policy.reset(obs)
    for _ in range(600):
        obs = env.step(policy.act(obs))[0]
        if env.cube_pos[2] > z0 + 0.01:
            qpos["grasp"] = env.data.qpos.copy()
            break
    assert "grasp" in qpos, "the scripted expert did not lift the cube"
    np.savez(out, **qpos)
    env.close()
    print("poses ->", out)


def dump(manifest: str, poses: Path, engine: str, out: Path) -> None:
    import mujoco

    if (
        engine == "newton:old"
    ):  # the Newton renderer as it was before commit e95354a (the scene exactly as exported)
        from physai.sim.newton import core, render

        render.drawable_scene_mjcf = lambda spec, model, robot_dir: core.scene_mjcf(
            spec, model, robot_dir
        )
    from physai.robots.rendering import start_isaac_first
    from physai.runtime import create_session

    renderer = engine.split(":")[0]
    start_isaac_first("mujoco", renderer)
    env = create_session(
        _manifest(manifest, simulator="mujoco", renderer=renderer), render=True
    ).runtime.robot
    saved = np.load(poses)
    frames = {}
    for pose in POSES:
        env.reset(seed=1000)
        env.data.qpos[:] = saved[pose]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        for camera in CAMERAS:
            for _ in range(
                3
            ):  # Isaac Sim's anti-aliasing settles over a few frames; the others draw the same each time
                image = np.asarray(env.render_camera(camera))
            frames[f"{pose}_{camera}"] = image
    env.close()
    np.savez(out, **frames)
    print(f"{engine}: {len(frames)} images -> {out}")


def luminance(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64).mean(axis=2)


def local_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixels where B differs from A by more than their smooth brightness ratio does."""
    from scipy import ndimage

    ratio = np.log((luminance(b) + EPS) / (luminance(a) + EPS))
    local = ratio - ndimage.median_filter(ratio, size=WINDOW, mode="nearest")
    mask = np.abs(local) > np.log(LOCAL_RATIO)
    return ndimage.binary_opening(
        mask, structure=np.ones((3, 3), bool)
    )  # edges moved by a pixel are 1-2 px wide


def regions(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Brightness ratio B / A of the far floor, the mid floor and the table, from the side columns (the arm is in the middle)."""
    height, width = a.shape[:2]
    side = np.zeros((height, width), bool)
    side[:, : int(0.3 * width)] = side[:, int(0.7 * width) :] = True
    bands = {
        "far": (0, height // 3),
        "mid": (height // 3, int(0.6 * height)),
        "table": (int(0.7 * height), height),
    }
    ya, yb = luminance(a), luminance(b)
    out = {}
    for name, (top, bottom) in bands.items():
        keep = side.copy()
        keep[:top] = keep[bottom:] = False
        out[name] = (
            float(np.median(yb[keep]) / max(np.median(ya[keep]), 1e-6))
            if keep.any()
            else float("nan")
        )
    return out


def flagged(mask: np.ndarray) -> list[dict]:
    from scipy import ndimage

    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    found = []
    for i, box in enumerate(ndimage.find_objects(labels), start=1):
        area = int((labels[box] == i).sum())
        if area >= MIN_REGION:
            found.append(
                {
                    "area": area,
                    "x": [box[1].start, box[1].stop],
                    "y": [box[0].start, box[0].stop],
                }
            )
    return sorted(found, key=lambda r: -r["area"])


def compare(a_path: Path, b_path: Path, truth: tuple[Path, Path] | None) -> dict:
    from scipy import ndimage

    a, b = np.load(a_path), np.load(b_path)
    report, verdicts = {}, []
    covered = truth_pixels = outside = flagged_pixels = 0
    for key in a.files:
        mask = local_mask(a[key], b[key])
        regs, ratios = flagged(mask), regions(a[key], b[key])
        verdict = (
            "defect"
            if regs and regs[0]["area"] > FLAG_AREA
            else "global shift"
            if not all(RATIO_BAND[0] <= r <= RATIO_BAND[1] for r in ratios.values())
            else "same"
        )
        verdicts.append(verdict)
        report[key] = {
            "verdict": verdict,
            "flagged_pixels": int(mask.sum()),
            "regions": regs[:5],
            "brightness_ratio": ratios,
        }
        if truth:
            gt = (
                np.load(truth[0])[key].astype(int) != np.load(truth[1])[key].astype(int)
            ).any(axis=2)
            near = ndimage.binary_dilation(gt, iterations=3)
            covered += int((mask & gt).sum())
            truth_pixels += int(gt.sum())
            outside += int((mask & ~near).sum())
            flagged_pixels += int(mask.sum())
    if truth:
        report["truth"] = {
            "pixels": truth_pixels,
            "covered": covered,
            "coverage": covered / max(truth_pixels, 1),
            "flagged": flagged_pixels,
            "outside_share": outside / max(flagged_pixels, 1),
        }
    report["verdict"] = (
        "defect"
        if "defect" in verdicts
        else "global shift"
        if "global shift" in verdicts
        else "same"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poses")
    p.add_argument("--manifest", required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("dump")
    d.add_argument("--manifest", required=True)
    d.add_argument("--poses", type=Path, required=True)
    d.add_argument(
        "--engine", required=True, choices=("mujoco", "newton", "newton:old", "isaac")
    )
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--truth-old", type=Path)
    c.add_argument("--truth-new", type=Path)
    c.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.cmd == "poses":
        make_poses(args.manifest, args.seed, args.out)
    elif args.cmd == "dump":
        dump(args.manifest, args.poses, args.engine, args.out)
    else:
        report = compare(
            args.a, args.b, (args.truth_old, args.truth_new) if args.truth_old else None
        )
        if args.json:
            args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
        for key, row in report.items():
            if isinstance(row, dict) and "verdict" in row:
                print(
                    f"{key:>14}: {row['verdict']:<13} regions {[r['area'] for r in row['regions']]} ratios "
                    + " ".join(
                        f"{k} {v:.2f}" for k, v in row["brightness_ratio"].items()
                    )
                )
        if "truth" in report:
            print("truth:", report["truth"])
        print("verdict:", report["verdict"])


if __name__ == "__main__":
    main()
