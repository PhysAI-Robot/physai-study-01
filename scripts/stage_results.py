"""Stage the per-seed result files of study 1 for Hugging Face, one folder per checkpoint (the weights are staged by stage_checkpoints*.py).

    python research/scripts/stage_results.py --out ~/.physai/hf_staging_results --v2-results ~/.physai/hf_staging_v2/results_v2

`--out` gets `results/<checkpoint>/<setting>.json` and `results/INDEX.tsv` (one row per file: checkpoint, file, successes, episodes, seeds,
what the setting is). Each file is `{"<seed>": true|false}`, nothing else (no paths, no logs), counted from the per-episode result files.
The second part's own files (`--v2-results`, from stage_checkpoints_v2.py) are copied as they are. For the first part's 17 checkpoints the
MuJoCo and Isaac Sim files are picked by name below and each pick is checked against the score in the published `checkpoints.json`; the
Newton file is the rescoring with the fixed renderer (`newton_rescore/`), and `newton_old_renderer.json` is the first part's own Newton
score where a file with that count and that name exists. Nothing is uploaded.

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2] / "outputs" / "study01"
V1, V2 = STUDY / "v1", STUDY / "v2"
REPO = "rizgiak/physai-study-01-checkpoints"

# checkpoint -> (MuJoCo seeds 1000-1299, Isaac Sim seeds 1000-1099, Newton before the renderer fix or None), under outputs/study01/v1
FIRST_PART = {
    "act_nominal_unseeded": (
        "evals/act_200_c100_60k_mujoco_1000-1299",
        "isaac/act_200_c100_60k_isaac_1000-1099",
        None,
    ),
    "act_nominal_seed1": (
        "experiments/e5/seed1_mujoco",
        "experiments/e5/seed1_isaac",
        None,
    ),
    "act_nominal_seed2": (
        "experiments/e5/seed2_mujoco",
        "experiments/e5/seed2_isaac",
        None,
    ),
    "act_nominal_seed3": (
        "evals/act_200_seed3_mujoco",
        "isaac/act_200_seed3_isaac",
        "evals/act_200_seed3_newton",
    ),
    "act_randomized_look_seed0": (
        "experiments/e3/mujoco_1000-1299",
        "evals/e3_isaac_merged_1000-1099",
        None,
    ),
    "act_randomized_look_seed1": (
        "experiments/e3s1/mujoco",
        "experiments/e3s1/isaac",
        None,
    ),
    "act_randomized_look_seed2": (
        "evals/act_visual_dr_seed2_mujoco",
        "isaac/act_visual_dr_seed2_isaac",
        "evals/act_visual_dr_seed2_newton",
    ),
    "act_both_renderers_unseeded": (
        "evals/act_mixed_400_mujoco_1000-1299",
        "evals/act_mixed_400_isaac_1000-1099_merged",
        None,
    ),
    "act_both_renderers_seed1": (
        "evals/act_mixed_400_seed1-eval/mujoco",
        "evals/act_mixed_400_seed1-eval/isaac",
        None,
    ),
    "act_isaac_only_100_unseeded": (
        "evals/act_isaac_100_mujoco_1000-1299",
        "isaac/act_isaac_100_isaac_1000-1099",
        "experiments/newton_wide/isaac100_1000-1299",
    ),
    "act_isaac_only_200_unseeded": (
        "evals/act_isaac_200_mujoco_1000-1299",
        "isaac/act_isaac_200_isaac_1000-1099",
        None,
    ),
    "act_isaac_only_200_seed1": (
        "evals/act_isaac_200_seed1-eval/mujoco",
        "evals/act_isaac_200_seed1-eval/isaac",
        None,
    ),
    "act_isaac_only_400_unseeded": (
        "evals/act_isaac_400_mujoco_1000-1299",
        "isaac/act_isaac_400_isaac_1000-1099",
        None,
    ),
    "act_chunk30_100demos_unseeded": (
        "evals/act_100_mujoco_1000-1299",
        "isaac/act_100_isaac_1000-1099",
        "experiments/newton_wide/act100_c30_30k_1000-1299",
    ),
    "act_chunk100_100demos_unseeded": (
        "evals/act_100_c100_60k_mujoco_1000-1299",
        "isaac/act_100_c100_60k_isaac_1000-1099",
        "experiments/newton_wide/act100_c100_60k_1000-1299",
    ),
    "diffusion_policy_v2_nominal_seed0": (
        "evals/dp2_200_mujoco",
        "isaac/dp2_200_isaac",
        "evals/dp2_200_newton",
    ),
    "diffusion_policy_v2_randomized_look_seed0": (
        "evals/dp2_visual_dr_mujoco",
        "isaac/dp2_visual_dr_isaac",
        "evals/dp2_visual_dr_newton",
    ),
}
# the 13 edits and lighting levels scored on the final nominal ACT on the seeds that settings were chosen on (900-999), and the state edits
PROBES = {
    "mujoco_seeds_900_999_background_x0.67": "evals/probe_base_bg_dim",
    "mujoco_seeds_900_999_cube_x1.16": "evals/probe_base_cube_gain",
    "mujoco_seeds_900_999_arm_tinted_orange": "evals/probe_base_arm_tint",
    "mujoco_seeds_900_999_all_four_gains": "evals/probe_base_isaac_like",
    "mujoco_seeds_900_999_blur_both_cameras": "evals/probe_base_blur",
    "mujoco_seeds_900_999_blur_front_only": "evals/probe_base_blur_front",
    "mujoco_seeds_900_999_blur_wrist_only": "evals/probe_base_blur_wrist",
    "mujoco_seeds_900_999_background_one_flat_colour": "evals/probe_base_bg_flat",
    "mujoco_seeds_900_999_background_grey": "evals/probe_base_bg_gray",
    "mujoco_seeds_900_999_front_camera_blanked": "evals/probe_base_blank_front",
    "mujoco_seeds_900_999_wrist_camera_blanked": "evals/probe_base_blank_wrist",
    "mujoco_seeds_900_999_table_x0.93": "evals/probe_base_table_dim",
    "mujoco_seeds_900_999_gripper_reading_follows_command": "evals/probe_state_gripper_track",
    "mujoco_seeds_900_999_gripper_reading_ramps": "evals/probe_state_gripper_lag",
    "mujoco_seeds_900_999_light_x0.5": "evals/val_act200_mujoco_light0.5",
    "mujoco_seeds_900_999_light_x0.7": "evals/val_act200_mujoco_light0.7",
    "mujoco_seeds_900_999_light_x1.0": "evals/val_act200_mujoco_light1.0",
    "mujoco_seeds_900_999_light_x1.4": "evals/val_act200_mujoco_light1.4",
}
N12 = {  # the file suffix of the pilot -> the file name; every one is MuJoCo, seeds 1000-1099
    "none": "mujoco_seeds_1000_1099_unedited",
    "blur": "mujoco_seeds_1000_1099_blur_both_cameras",
    "light05": "mujoco_seeds_1000_1099_light_x0.5",
    "light14": "mujoco_seeds_1000_1099_light_x1.4",
    "cam10": "mujoco_seeds_1000_1099_camera_shifted_10mm",
    "arm_tint": "mujoco_seeds_1000_1099_arm_tinted_orange",
    "bg_dim": "mujoco_seeds_1000_1099_background_x0.67",
    "plain": "mujoco_seeds_1000_1099_plain_floor",
}
LOOKS = {  # the Newton looks outside the randomized ranges (N11) -> the file name
    "all_unseen": "newton_look_outside_every_range",
    "floor_bw": "newton_look_black_white_floor",
    "floor_fine": "newton_look_fine_tile_floor",
    "light_low": "newton_look_low_light",
    "sky_dark": "newton_look_dark_sky",
    "heldout_v2": "newton_heldout_look",
}
V2_NAMES = {
    "nominal_seed1": "viperx_act_nominal_seed1",
    "nominal_seed2": "viperx_act_nominal_seed2",
}
WHAT = {
    "mujoco": "MuJoCo, seeds 1000-1299",
    "newton": "MuJoCo physics, Newton pictures (fixed renderer)",
    "newton_old_renderer": "MuJoCo physics, Newton pictures, renderer before the fix (first part)",
    "isaac": "Isaac Sim",
    "newton_heldout_look": "Newton (fixed renderer), the wood-coloured held-out look",
    "mujoco_pad_visible": "MuJoCo, scene that shows the contact pad",
    "newton_pad_visible": "Newton, scene that shows the contact pad",
    "mujoco_physics_isaac_pictures": "MuJoCo physics, Isaac Sim pictures (cell MI)",
    "isaac_physics_mujoco_pictures": "Isaac Sim physics, MuJoCo pictures (cell IM)",
}


def per_seed(source: Path) -> dict[int, bool]:
    """Outcomes of a per-episode result file, or of a folder of Isaac Sim shards."""
    files = [source] if source.is_file() else sorted(source.rglob("*.json"))
    found: dict[int, bool] = {}
    for f in files:
        if any(
            part in str(f) for part in ("videos", "attempts", "failed_after_reboot")
        ):
            continue
        try:
            rows = [
                (int(e["seed"]), bool(e["success"]))
                for e in json.loads(f.read_text(encoding="utf-8"))["results"]
            ]
        except (ValueError, KeyError, TypeError):
            continue
        for seed, ok in rows:
            if found.setdefault(seed, ok) != ok:
                raise SystemExit(f"seed {seed} has two outcomes under {source}")
    if not found:
        raise SystemExit(f"no result in {source}")
    return found


def find(base: Path, stem: str) -> Path:
    return base / f"{stem}.json" if (base / f"{stem}.json").exists() else base / stem


class Stage:
    def __init__(self, out: Path) -> None:
        self.out, self.rows = out / "results", []

    def put(
        self,
        checkpoint: str,
        name: str,
        seeds: dict[int, bool],
        what: str | None = None,
    ) -> None:
        path = self.out / checkpoint / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({str(k): seeds[k] for k in sorted(seeds)}), encoding="utf-8"
        )
        lo, hi = min(seeds), max(seeds)
        self.rows.append(
            (
                checkpoint,
                f"{name}.json",
                sum(seeds.values()),
                len(seeds),
                f"{lo}-{hi}",
                what or WHAT.get(name, name.replace("_", " ")),
            )
        )

    def counts(self, checkpoint: str, name: str) -> str:
        row = next(
            r for r in self.rows if r[0] == checkpoint and r[1] == f"{name}.json"
        )
        return f"{row[2]}/{row[3]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--v2-results",
        type=Path,
        required=True,
        help="results_v2/ of stage_checkpoints_v2.py",
    )
    ap.add_argument(
        "--card",
        type=Path,
        required=True,
        help="the published checkpoints.json of the first part",
    )
    args = ap.parse_args()
    stage = Stage(args.out)
    card = {
        c["name"]: c["scores"]
        for c in json.loads(args.card.read_text(encoding="utf-8"))
    }
    problems = []
    for name, (mujoco, isaac, newton_old) in FIRST_PART.items():
        stage.put(name, "mujoco", per_seed(find(V1, mujoco)))
        stage.put(
            name, "isaac", per_seed(find(V1, isaac)), "Isaac Sim, seeds 1000-1099"
        )
        stage.put(name, "newton", per_seed(V2 / "newton_rescore" / f"{name}.json"))
        if newton_old:
            stage.put(name, "newton_old_renderer", per_seed(find(V1, newton_old)))
            if (
                stage.counts(name, "newton_old_renderer")
                != card[name]["newton_seeds_1000_1299"]
            ):
                problems.append(
                    f"{name} newton_old_renderer {stage.counts(name, 'newton_old_renderer')} != {card[name]['newton_seeds_1000_1299']}"
                )
        for engine, key in (
            ("mujoco", "mujoco_seeds_1000_1299"),
            ("isaac", "isaac_sim_seeds_1000_1099"),
        ):
            if stage.counts(name, engine) != card[name][key]:
                problems.append(
                    f"{name} {engine} {stage.counts(name, engine)} != {card[name][key]}"
                )
    if problems:
        raise SystemExit("\n".join(problems))
    for model in sorted(args.v2_results.iterdir()):
        for f in sorted(model.glob("*.json")):
            stage.put(
                model.name,
                f.stem,
                {
                    int(k): v
                    for k, v in json.loads(f.read_text(encoding="utf-8")).items()
                },
            )
    for probe, source in PROBES.items():
        stage.put("act_nominal_unseeded", probe, per_seed(find(V1, source)))
    for folder, name in (
        ("n9", "mujoco_physics_isaac_pictures"),
        ("n13", "isaac_physics_mujoco_pictures"),
    ):
        for model in sorted((V2 / folder).iterdir()):
            if model.is_dir():
                stage.put(model.name, name, per_seed(model))
    for cell, name in (
        ("mi", "mujoco_physics_isaac_pictures"),
        ("im", "isaac_physics_mujoco_pictures"),
    ):
        for run, checkpoint in V2_NAMES.items():
            stage.put(
                checkpoint,
                name,
                per_seed(V2 / "n14" / cell / run),
                WHAT[name] + ", seeds 1000-1299",
            )
    for f in sorted((V2 / "n11").glob("*.json")) + sorted(
        (V2 / "look_rescore").glob("*.json")
    ):
        for look, name in LOOKS.items():
            if f.stem.endswith(f"_{look}"):
                model = (
                    f.stem[: -len(look) - 1]
                    .replace("randlook_bw", "act_randomized_look_bw")
                    .replace("randlook_", "act_randomized_look_")
                )
                stage.put(
                    model,
                    name,
                    per_seed(f),
                    None
                    if name in WHAT
                    else "Newton (fixed renderer), "
                    + name.replace("newton_look_", "look: ").replace("_", " "),
                )
    for f in sorted((V2 / "n12").glob("*.json")):
        model, edit = f.stem.rsplit(".", 1)
        stage.put(
            model,
            N12[edit],
            per_seed(f),
            "MuJoCo, "
            + N12[edit].replace("mujoco_seeds_1000_1099_", "").replace("_", " ")
            + " (sensitivity pilot)",
        )
    rows = sorted(stage.rows)
    (stage.out / "INDEX.tsv").write_text(
        "checkpoint\tfile\tsuccesses\tepisodes\tseeds\tsetting\n"
        + "\n".join("\t".join(map(str, r)) for r in rows)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{len(rows)} files for {len({r[0] for r in rows})} checkpoints -> {stage.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
