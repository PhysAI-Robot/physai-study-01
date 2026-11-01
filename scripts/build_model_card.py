"""Write the Hugging Face model card of rizgiak/physai-study-01-checkpoints (both parts of study 1).

    python research/scripts/build_model_card.py --v1-card OLD_README.md --staging ~/.physai/hf_staging_v2 --out README.md

The first part's table is taken from the card that is on the Hub (`--v1-card`; its scores were typed by
`stage_checkpoints.py` from FINDINGS) and gets two more columns counted from the per-seed result files of
`outputs/study01/v2/`: the Newton score after the renderer fix and the Isaac Sim score on the second part's stack, where
the model was scored again. The second part's table is counted by `stage_checkpoints_v2.py` (`checkpoints_v2.json`).
Nothing is uploaded.

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

V2 = Path(__file__).resolve().parents[2] / "outputs" / "study01" / "v2"
REPO = "rizgiak/physai-study-01-checkpoints"

# models of the first part scored again in the second part: Isaac Sim on the second part's stack
ISAAC_AGAIN = {
    "act_nominal_unseeded": "isaac_n2/bridge_act_nominal_unseeded",
    "act_nominal_seed1": "isaac_n2/bridge_act_nominal_seed1",
    "act_nominal_seed2": "isaac_n2/bridge_act_nominal_seed2",
    "act_nominal_seed3": "isaac_n2/bridge_act_nominal_seed3",
    "act_randomized_look_seed0": "isaac_n1/randlook_seed0",
    "act_randomized_look_seed1": "isaac_n1/randlook_seed1",
    "act_randomized_look_seed2": "isaac_n1/randlook_seed2",
}


def count(path: Path) -> str | None:
    """successes/episodes over the seeds 1000-1299 found in a per-seed result file or a folder of shard files."""
    files = (
        [path]
        if path.is_file()
        else sorted([*path.glob("s*/shard-*.json"), *path.glob("s*/merged.json")])
    )
    seen: dict[int, bool] = {}
    for f in files:
        for e in json.loads(f.read_text(encoding="utf-8"))["results"]:
            seen[int(e["seed"])] = bool(e["success"])
    seen = (
        {k: v for k, v in seen.items() if 1000 <= k < 1100} if path.is_dir() else seen
    )
    return f"{sum(seen.values())}/{len(seen)}" if seen else None


def v1_rows(card: str) -> list[list[str]]:
    rows = []
    for line in card.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if (
            len(cells) == 5
            and cells[0].startswith("`")
            and re.fullmatch(r"\d+/\d+", cells[2])
        ):
            rows.append([cells[0].strip("`"), *cells[1:]])
    return rows


HEAD = """---
license: apache-2.0
pipeline_tag: robotics
tags:
  - robotics
  - imitation-learning
  - act
  - diffusion-policy
  - lerobot
  - sim-to-sim
  - so101
  - viperx
---

# Study 1 checkpoints: randomized pick-and-place on the SO-101 and the ViperX 300 S

Checkpoints behind the numbers of study 1 (how much a learned policy loses when only the renderer changes; ACT and
Diffusion Policy; MuJoCo, Isaac Sim and Newton as the renderer). One subfolder per checkpoint, 34 in all: the first part
of the study (17 folders, the SO-101) and the second part, study-01-v2 (17 folders: the SO-101, the ViperX 300 S, and
the controls and mechanism tests of the second part).

- Task: the robot puts a red cube on a green pad, cube and pad placed at random over the workspace it can reach top-down
  (SO-101: `research/configs/so101_randomized_pick_place.yaml`; ViperX 300 S:
  `research/configs/viperx300s_single_cube_randomized_place.yaml`, symmetric in y).
- Demonstrations come from a privileged scripted expert on MuJoCo (seeds 0-199, or 0-399). The "Isaac" demonstrations are
  the same MuJoCo actions replayed on Isaac Sim with Isaac's frames, not an expert on Isaac.
- Models of the same recipe differ only in the network seed; the spread between them is part of the result. A folder name
  ends in `_seedN` when the network seed was set to N and in `_unseeded` when it was not; N is a label, not a run number.

## How to read the scores

Scores are successes over held-out episodes (seeds 1000-1299; seeds 900-999 were used to choose settings and are never
reported, and no demonstration seed overlaps them).

- **MuJoCo**: the training renderer, 300 episodes.
- **Newton**: a render-only renderer (MuJoCo keeps the physics, Newton draws the camera frames), 300 episodes. **The Newton
  renderer was fixed during the second part:** it drew two contact pads (collision-only geoms) and a wrongly coloured
  material that MuJoCo does not draw, and a 760-pixel difference moved the scores of some models by up to 26 of 300
  episodes. Every Newton score of the second part, and the column "Newton (fixed)" below, uses the fixed renderer. The
  column "Newton (v1)" is what the first part reported with the renderer as it was; it is kept as a sensitivity analysis,
  and 400-demonstration and randomized-look models move by 5 episodes or less.
- **Isaac Sim**: 100 episodes (seeds 1000-1099) for most models, 300 (1000-1299) where a test needed more (marked). The
  first part was scored on Isaac Sim 6.1.0.0 (driver 610.47); the second part on Isaac Sim 6.2.0.0 (driver 617.42, Windows
  build 26300). The seven first-part models that were scored again on the new stack are in the column "Isaac (new stack)":
  the mean shift of the four nominal runs is -0.75 points, inside the bridge rule of 3 points. PhysX is not
  deterministic, so an Isaac Sim score reproduces only within sampling noise.

## First part (17 folders, SO-101)

| Folder | Recipe | MuJoCo | Newton (fixed) | Newton (v1) | Isaac Sim | Isaac (new stack) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
"""

PART2 = """
## Second part, study-01-v2 (17 folders)

Counted from the per-seed files in `results_v2/<folder>/<engine>.json` (a JSON object `seed -> success`), so the paired tests
of the study (exact McNemar, rank tests) can be recomputed. `checkpoints_v2.json` lists the recipe, the seeds, the scores
and the sha256 of every weights file of this part. The Isaac Sim column says how many seeds were scored.

| Folder | Recipe | MuJoCo | Newton | Isaac Sim | Other |
| --- | --- | ---: | ---: | ---: | --- |
"""

TAIL = """
Notes on the second part: `act_nominal400_*` is the nominal look with the same 400 demonstrations and 120k steps as the
randomized look (the control of the study); `act_randomized_look_bw_*` adds black and white floor tiles to 30% of the
episodes and is also scored on a look it never saw (wood-coloured tiles, dark blue sky, mirrored key light; "Other");
`act_pad_visible_*` was trained on images that show the SO-101's two contact pads, and its "Other" column gives its scores
with the pad drawn (MuJoCo, Newton before the fix) and without it; the `viperx_*` models are for the ViperX 300 S manifest.

Each folder holds `model.safetensors`, `config.json`, `dataset_stats.json`, `training_meta.json` and `checkpoint_meta.json`
(and `training_log.json`). `checkpoints.json` (first part) and `checkpoints_v2.json` (second part) list the sha256 of every
weights file.

## Use

With the physai framework (https://github.com/PhysAI-Robot/physai-robot-starter), `--checkpoint` takes a reference to a
folder of this repository and downloads it into `checkpoints/` (ignored by git):

```bash
python scripts/eval_policy.py --manifest research/configs/so101_randomized_pick_place.yaml --policy lerobot --checkpoint hf:rizgiak/physai-study-01-checkpoints/act_randomized_look_seed0@<commit>
python scripts/eval_policy.py --manifest research/configs/viperx300s_single_cube_randomized_place.yaml --policy lerobot --checkpoint hf:rizgiak/physai-study-01-checkpoints/viperx_act_nominal_seed1@<commit>
python scripts/fetch_checkpoints.py hf:rizgiak/physai-study-01-checkpoints/act_randomized_look_seed0   # download only, to use offline
```

Add `--sim isaac` for Isaac Sim, or `--sim mujoco --render newton` for Newton. Name a commit (`@<commit>`) for a result you
report; without one `main` is used (the second part is commit `ece89d5` and later). For the Diffusion Policy checkpoints add
`--policy-arg n_action_steps=16`. To use a folder without the framework's downloader, fetch it with
`huggingface_hub.snapshot_download("rizgiak/physai-study-01-checkpoints", allow_patterns=["<folder>/*"])` and pass its path
to `--checkpoint`. `act_pad_visible_seed1` is scored with the pads drawn through `research/scripts/with_pad_visible.py`.

Three kinds of seeds appear and they are unrelated:

- **Episode seeds of the demonstrations** (`demonstration_episode_seeds`): which scene layouts the demonstrations were
  recorded on (0-199 for 200 demos, 0-399 for 400). All runs of a recipe train on the same demonstrations.
- **Network seed** (`network_seed`): seeds only the weight initialization and the order of the training batches. Runs of one
  recipe that differ only in it measure how much a single training run varies. `null` means the seed was not set (the first
  run of a recipe, trained before it was recorded).
- **Evaluation seeds**: 900-999 to choose settings, 1000-1299 for the scores above. They never overlap the demonstration seeds.

## Dataset names in `training_meta.json`

`training_meta.json` records the local folder names of the demonstrations the model was trained on:

| Name | What it is |
| --- | --- |
| `randomized_v1`, `mujoco_nominal_200_seeds0-199` | 200 expert demonstrations on MuJoCo, seeds 0-199, the SO-101's randomized pick-and-place task |
| `mujoco_nominal_200_seeds200-399` | 200 further expert demonstrations (seeds 200-399); with the previous one, the 400 of `act_nominal400_*` |
| `m5_visual_dr`, `mujoco_randomized-look_400_seeds0-399` | 400 expert demonstrations on MuJoCo, seeds 0-399, with the look (colours, floor tiles, lighting) randomized in every episode |
| `mujoco_randomized-look-bw_400_seeds0-399` | the same with black and white floor tiles in 30% of the episodes |
| `mujoco_nominal_padvisible_200_seeds0-199` | the 200 nominal demonstrations with the contact pads drawn opaque |
| `viperx_nominal_400_seeds0-399`, `viperx_randomized-look_400_seeds0-399` | 400 expert demonstrations on the ViperX 300 S, nominal and randomized look |
| `randomized_isaac_v1`, `randomized_isaac_v1b` | the actions of `randomized_v1` replayed on Isaac Sim with Isaac's frames (seeds 0-99 and 100-199) |
| `randomized_isaac_v2a`, `randomized_isaac_v2b` | the same for 200 further MuJoCo demonstrations (seeds 200-399), used by `act_isaac_only_400` |

## What is not here

- The demonstrations. Those from MuJoCo are regenerated from their seeds with the study's collection script. The Isaac Sim
  replays should not be expected to regenerate identically (Isaac runs were not deterministic in a test), which matters for
  the `act_both_renderers_*` and `act_isaac_only_*` models.
- Checkpoints of runs that are not in a table of the study (a first Diffusion Policy version that was replaced, and early
  ablations).

Reproducing the scores:

- ACT on MuJoCo: the same episodes again, step for step, given the same versions of MuJoCo, lerobot and the study's code
  (checked on one downloaded checkpoint: 10 of 10 seeds identical). Newton with the fixed renderer is deterministic too: one
  model scored twice on one day gives the same outcome on all 300 seeds.
- Diffusion Policy: the policy samples random noise at every prediction and the evaluation does not fix the torch seed, so
  two runs of the same checkpoint on the same seeds differ in the details of an episode. Four runs of
  `diffusion_policy_v2_nominal_seed0` on the MuJoCo seeds 1000-1299 scored 290, 291, 292 and 293 of 300 (standard deviation
  1.3 episodes), and two scorings on Newton (fixed) gave 11 and 10 of 300.
- Isaac Sim, for every model: PhysX is not deterministic, so success rates reproduce within sampling noise. On Isaac Sim the
  cube settles 3 mm (SO-101) or 8.6 mm (ViperX) from its drawn position in the first control step whenever it is placed at a
  new place, a small difference of the Isaac Sim task that the camera images show.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--v1-card",
        type=Path,
        required=True,
        help="the README.md of the first part, as on the Hub",
    )
    ap.add_argument(
        "--staging", type=Path, required=True, help="output of stage_checkpoints_v2.py"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for name, recipe, mujoco, newton_v1, isaac in v1_rows(
        args.v1_card.read_text(encoding="utf-8")
    ):
        fixed = count(V2 / "newton_rescore" / f"{name}.json")
        again = count(V2 / ISAAC_AGAIN[name]) if name in ISAAC_AGAIN else None
        rows.append(
            f"| `{name}` | {recipe} | {mujoco} | {fixed or '-'} | {newton_v1} | {isaac} | {again or '-'} |"
        )
    assert len(rows) == 17, len(rows)

    v2 = []
    for m in json.loads(
        (args.staging / "checkpoints_v2.json").read_text(encoding="utf-8")
    ):
        s = {k: v.split(" (")[0] for k, v in m["scores"].items()}
        extra = [
            f"{k} {v}" for k, v in s.items() if k not in ("mujoco", "newton", "isaac")
        ]
        isaac = s.get("isaac", "-")
        seeds = m["scores"].get("isaac", "")
        if "1000-1299" in seeds:
            per_seed = json.loads(
                (args.staging / "results_v2" / m["name"] / "isaac.json").read_text(
                    encoding="utf-8"
                )
            )
            first = sum(ok for seed, ok in per_seed.items() if int(seed) < 1100)
            isaac += f" (300 seeds; first 100: {first})"
        v2.append(
            f"| `{m['name']}` | {m['recipe']} | {s.get('mujoco', '-')} | {s.get('newton', '-')} | {isaac} | {'; '.join(extra) or '-'} |"
        )
    args.out.write_text(
        HEAD + "\n".join(rows) + "\n" + PART2 + "\n".join(v2) + "\n" + TAIL,
        encoding="utf-8",
    )
    print(f"{len(rows)} + {len(v2)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
