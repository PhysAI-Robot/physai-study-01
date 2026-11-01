"""End-to-end check of the ViperX 300 S robot and its scripted expert; writes a report.

    python research/scripts/viperx300s_check.py --seeds 20
    python research/scripts/viperx300s_check.py --seeds 100 --start 1000 --no-images

Writes into outputs/viperx300s/ (git-ignored):
  reach.json     IK reachability of hover / grasp / carry poses over the cube and target ranges
  episodes.json  one row per seed: success, steps, retries, where it ended, cube-to-target distance
  *.png          front and wrist camera frames at reset, at the grasp and at the end of seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _core  # noqa: F401
import numpy as np

from physai.config import load_manifest
from physai.config.compat import with_overrides
from physai.runtime import create_session
from physai.tasks import TaskRuntime
from research.scripted_experts.viperx300s_pick_place_expert import (
    ViperX300SExpertConfig,
    ViperX300SPickPlaceExpert,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "viperx300s"
MANIFEST = (
    ROOT / "research" / "configs" / "viperx300s_single_cube_randomized_place.yaml"
)


def make_env(render: bool, max_steps: int = 600) -> TaskRuntime:
    """The research ViperX session, whose sampler is the core manifest's, for the caller to drive."""
    manifest = with_overrides(
        load_manifest(MANIFEST), simulator="mujoco", policy="idle", max_steps=max_steps
    )
    return create_session(manifest, render=render).runtime.robot


def save(name: str, image: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(OUT / f"{name}.png")
    except (
        ImportError,
        OSError,
        ValueError,
    ) as exc:  # the report survives without pictures
        print(f"could not save {name}.png: {exc}")


def reach(env: TaskRuntime, cfg: ViperX300SExpertConfig) -> dict:
    """Is a top-down pose reachable at every height the expert asks for?"""
    kin = env.kin
    rest = env.rest_z
    heights = {
        "hover": rest + cfg.hover_height,
        "grasp": rest + cfg.grasp_height,
        "carry": rest + cfg.grasp_height + cfg.lift_height,
        "place": rest + cfg.grasp_height + cfg.place_height,
    }
    cube_x, cube_y = env.cfg.cube_x_range, env.cfg.cube_y_range
    target_x, target_y = env.cfg.target_x_range, env.cfg.target_y_range
    result = {}
    for name, z in heights.items():
        ranges = (
            (target_x, target_y) if name in ("carry", "place") else (cube_x, cube_y)
        )
        grid = [
            (x, y)
            for x in np.linspace(*ranges[0], 6)
            for y in np.linspace(*ranges[1], 6)
        ]
        ok = 0
        worst = 0.0
        q = None
        for x, y in grid:
            solution = kin.ik_pinch(np.array([x, y, z]), q_init=q)
            q = solution.qpos
            ok += int(solution.converged)
            worst = max(worst, solution.position_error)
        result[name] = {
            "z": round(z, 4),
            "reachable": f"{ok}/{len(grid)}",
            "worst_err_m": worst,
        }
    return result


def run_episode(
    env: TaskRuntime, cfg: ViperX300SExpertConfig, seed: int, pictures: bool
) -> dict:
    obs = env.reset(seed=seed)
    policy = ViperX300SPickPlaceExpert(env.kin, env, cfg)
    policy.reset(obs)
    retries = 0
    original = policy._retry_grasp

    def counted():
        nonlocal retries
        retries += 1
        original()

    policy._retry_grasp = counted
    if pictures:
        save(f"seed{seed}_reset_front", obs.images["front"].data)
        save(f"seed{seed}_reset_wrist", obs.images["wrist"].data)
    phases = []
    success = False
    shot = False
    steps = 0
    for steps in range(1, env.cfg.max_steps + 1):
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        if not phases or phases[-1] != policy.phase.name:
            phases.append(policy.phase.name)
        if pictures and not shot and policy.phase.name == "LIFT":
            save(f"seed{seed}_lift_front", obs.images["front"].data)
            save(f"seed{seed}_lift_wrist", obs.images["wrist"].data)
            shot = True
        if info.get("success"):
            success = True
            break
        if terminated or truncated:
            break
    if pictures:
        save(f"seed{seed}_end_front", obs.images["front"].data)
        save(f"seed{seed}_end_wrist", obs.images["wrist"].data)
    cube, target = env.cube_pos, env.target_pos
    return {
        "seed": seed,
        "success": success,
        "steps": steps,
        "retries": retries,
        "last_phase": policy.phase.name,
        "phases": phases,
        "cube_target_xy_m": round(float(np.linalg.norm(cube[:2] - target[:2])), 4),
        "cube_z": round(float(cube[2]), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--tag", default="", help="file-name suffix for this run")
    for name, value in vars(ViperX300SExpertConfig()).items():
        if isinstance(value, float):
            ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = ViperX300SExpertConfig(
        **{
            name: getattr(args, name)
            for name in vars(ViperX300SExpertConfig())
            if isinstance(getattr(ViperX300SExpertConfig(), name), float)
        }
    )
    pictures = not args.no_images
    env = make_env(render=pictures)
    try:
        report = reach(env, cfg)
        (OUT / f"reach{args.tag}.json").write_text(json.dumps(report, indent=2))
        print("reach:", json.dumps(report))
        rows = []
        started = time.time()
        for seed in range(args.start, args.start + args.seeds):
            row = run_episode(env, cfg, seed, pictures and seed == args.start)
            rows.append(row)
            print(
                f"seed {seed}: {'OK ' if row['success'] else 'FAIL'} steps={row['steps']} "
                f"retries={row['retries']} end={row['last_phase']} "
                f"cube-target={row['cube_target_xy_m']} cube_z={row['cube_z']}",
                flush=True,
            )
        wins = sum(r["success"] for r in rows)
        summary = {
            "config": {k: v for k, v in vars(cfg).items() if k != "approach_dir"},
            "success": f"{wins}/{len(rows)}",
            "rate": wins / max(len(rows), 1),
            "seconds": round(time.time() - started, 1),
        }
        (OUT / f"episodes{args.tag}.json").write_text(
            json.dumps({"summary": summary, "episodes": rows}, indent=2)
        )
        print("SUMMARY:", json.dumps(summary))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
