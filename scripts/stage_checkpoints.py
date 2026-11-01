"""Copy study 1's checkpoints into one upload folder with a model card and checksums.

    python research/scripts/stage_checkpoints.py --out ~/hf_staging
    python research/scripts/stage_checkpoints.py --out DIR --tier 1      # headline models only

Each checkpoint becomes one subfolder of `--out` with the files the policy loader needs. The folder also gets
`README.md` (model card: what each model is, how it was trained, how it scored) and `checkpoints.json` (per
model: algorithm, recipe, training steps and seed, scores, sha256 of the weights). Nothing in the published files
names this repository's internal run folders, which will not exist for a reader. Nothing is uploaded: check the folder, then upload it yourself, for example
`hf upload rizgiak/physai-study-01-checkpoints DIR .` after `hf auth login`.

Scores are the ones recorded in `research/imitation_learning/FINDINGS.md` (MuJoCo and Newton on seeds
1000-1299, Isaac Sim on 1000-1099); they are typed here once and not recomputed.

Note (after the study-01-v1 reorganisation): the local checkpoint folders named below (`m5/act_200_seed1`, ...) were removed
once the weights were uploaded. The published copy at `rizgiak/physai-study-01-checkpoints` is the reference; this script
stays as the record of how it was staged and will not run without those folders.

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "study01"
REPO = "rizgiak/physai-study-01-checkpoints"
REQUIRED = (
    "model.safetensors",
    "config.json",
    "dataset_stats.json",
    "training_meta.json",
    "checkpoint_meta.json",
)
OPTIONAL = ("training_log.json",)


@dataclass(frozen=True)
class Model:
    name: str
    source: str  # under outputs/study01
    tier: int
    recipe: str
    mujoco: str
    newton: str
    isaac: str


MODELS = (
    Model(
        "act_nominal_unseeded",
        "act_200_c100_60k",
        1,
        "ACT, 200 MuJoCo demos (episode seeds 0-199), chunk 100, 60k steps, network seed not set",
        "294/300",
        "268/300",
        "58/100",
    ),
    Model(
        "act_nominal_seed1",
        "m5/act_200_seed1",
        1,
        "ACT, 200 MuJoCo demos (episode seeds 0-199), chunk 100, 60k steps, network seed 1",
        "290/300",
        "228/300",
        "62/100",
    ),
    Model(
        "act_nominal_seed2",
        "m5/act_200_seed2",
        1,
        "ACT, 200 MuJoCo demos (episode seeds 0-199), chunk 100, 60k steps, network seed 2",
        "289/300",
        "242/300",
        "77/100",
    ),
    Model(
        "act_nominal_seed3",
        "m5c/act_200_seed3",
        1,
        "ACT, 200 MuJoCo demos (episode seeds 0-199), chunk 100, 60k steps, network seed 3",
        "290/300",
        "258/300",
        "65/100",
    ),
    Model(
        "act_randomized_look_seed0",
        "m5/act_visual_dr",
        1,
        "ACT, 400 randomized-look MuJoCo demos (episode seeds 0-399), chunk 100, 120k steps, network seed 0",
        "294/300",
        "295/300",
        "90/100",
    ),
    Model(
        "act_randomized_look_seed1",
        "m5/act_visual_dr_seed1",
        1,
        "ACT, 400 randomized-look MuJoCo demos (episode seeds 0-399), chunk 100, 120k steps, network seed 1",
        "292/300",
        "289/300",
        "85/100",
    ),
    Model(
        "act_randomized_look_seed2",
        "m5c/act_visual_dr_seed2",
        1,
        "ACT, 400 randomized-look MuJoCo demos (episode seeds 0-399), chunk 100, 120k steps, network seed 2",
        "292/300",
        "289/300",
        "89/100",
    ),
    Model(
        "act_both_renderers_unseeded",
        "m4/act_mixed_400",
        1,
        "ACT, 200 MuJoCo demos (episode seeds 0-199) + the same 200 replayed on Isaac Sim, 120k steps, network seed not set",
        "294/300",
        "281/300",
        "94/100",
    ),
    Model(
        "act_both_renderers_seed1",
        "m5b/act_mixed_400_seed1",
        1,
        "ACT, 200 MuJoCo demos (episode seeds 0-199) + the same 200 replayed on Isaac Sim, 120k steps, network seed 1",
        "294/300",
        "286/300",
        "91/100",
    ),
    Model(
        "diffusion_policy_v2_nominal_seed0",
        "m5c/dp2_200",
        1,
        "Diffusion Policy v2, 200 MuJoCo demos (episode seeds 0-199), 100k steps, network seed 0, plays 16 of 64 actions",
        "290/300",
        "14/300",
        "21/100",
    ),
    Model(
        "diffusion_policy_v2_randomized_look_seed0",
        "m5c/dp2_visual_dr",
        1,
        "Diffusion Policy v2, 400 randomized-look demos (episode seeds 0-399), 120k steps, network seed 0, plays 16 of 64 actions",
        "298/300",
        "18/300",
        "49/100",
    ),
    Model(
        "act_isaac_only_100_unseeded",
        "m4/act_isaac_100",
        2,
        "ACT, 100 MuJoCo demos (episode seeds 0-99) replayed on Isaac Sim (Isaac images only), chunk 100, 60k steps, network seed not set",
        "229/300",
        "109/300",
        "85/100",
    ),
    Model(
        "act_isaac_only_200_unseeded",
        "m4/act_isaac_200",
        2,
        "ACT, 200 demos (episode seeds 0-199) replayed on Isaac Sim, 60k steps, network seed not set",
        "229/300",
        "105/300",
        "91/100",
    ),
    Model(
        "act_isaac_only_200_seed1",
        "m5b/act_isaac_200_seed1",
        2,
        "ACT, 200 demos (episode seeds 0-199) replayed on Isaac Sim, 60k steps, network seed 1",
        "193/300",
        "105/300",
        "90/100",
    ),
    Model(
        "act_isaac_only_400_unseeded",
        "m4/act_isaac_400",
        2,
        "ACT, 400 demos (episode seeds 0-399) replayed on Isaac Sim, 120k steps, network seed not set",
        "253/300",
        "154/300",
        "97/100",
    ),
    Model(
        "act_chunk30_100demos_unseeded",
        "act_100",
        2,
        "ACT, 100 MuJoCo demos (episode seeds 0-99), chunk 30, 30k steps, network seed not set (rejected on validation seeds)",
        "220/300",
        "16/300",
        "26/100",
    ),
    Model(
        "act_chunk100_100demos_unseeded",
        "act_100_c100_60k",
        2,
        "ACT, 100 MuJoCo demos (episode seeds 0-99), chunk 100, 60k steps, network seed not set",
        "275/300",
        "238/300",
        "55/100",
    ),
)

CARD = """---
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
---

# Study 1 checkpoints: randomized SO-101 pick-and-place

Checkpoints behind the numbers of study 1 (how much a learned policy loses when only the renderer changes;
ACT and Diffusion Policy; MuJoCo, Isaac Sim and Newton as the renderer). One subfolder per checkpoint.

- Task: SO-101 puts a red cube on a green pad, cube and pad placed at random over the reachable workspace.
- Scores: success on held-out seeds, MuJoCo and Newton on 1000-1299 (300 episodes), Isaac Sim on 1000-1099
  (100 episodes). Newton here is a render-only renderer (MuJoCo physics, Newton draws the camera frames).
- Demonstrations come from a privileged scripted expert on MuJoCo (seeds 0-199, or 0-399 for the randomized
  look). The "Isaac" demonstrations are the same MuJoCo actions replayed on Isaac Sim with Isaac's frames,
  not an expert on Isaac.
- Models of the same recipe differ only in the network seed; the spread between them is part of the result. A
  folder name ends in `_seedN` when the network seed was set to N and in `_unseeded` when it was not; N is just a
  label for the seed, not a run number.

| Folder | Recipe | MuJoCo | Newton | Isaac Sim |
| --- | --- | --- | --- | --- |
{rows}

Each folder holds `model.safetensors`, `config.json`, `dataset_stats.json`, `training_meta.json`,
`checkpoint_meta.json` (and `training_log.json`). `checkpoints.json` lists the sha256 of every weights file.

## Use

With the physai framework (https://github.com/PhysAI-Robot/physai-robot-starter), `--checkpoint` takes a reference
to a folder of this repository and downloads it into `checkpoints/` (ignored by git):

```bash
python scripts/eval_policy.py --manifest research/configs/so101_randomized_pick_place.yaml --policy lerobot --checkpoint hf:{repo}/act_randomized_look_seed0@<commit>
python scripts/fetch_checkpoints.py hf:{repo}/act_randomized_look_seed0   # download only, to use offline
```

Name a commit (`@<commit>`) for a result you report; without one `main` is used. For the Diffusion Policy
checkpoints add `--policy-arg n_action_steps=16`. To use a folder without the framework's downloader, fetch it with
`huggingface_hub.snapshot_download("{repo}", allow_patterns=["<folder>/*"])` and pass its path to `--checkpoint`.

Three kinds of seeds appear and they are unrelated:

- **Episode seeds of the demonstrations** (`demonstration_episode_seeds`): which scene layouts the demonstrations
  were recorded on (0-199 for 200 demos, 0-399 for 400). All runs of a recipe train on the same demonstrations.
- **Network seed** (`network_seed`): seeds only the weight initialization and the order of the training batches.
  Runs of one recipe that differ only in it measure how much a single training run varies. `null` means the seed
  was not set (the first run of a recipe, trained before it was recorded).
- **Evaluation seeds**: 900-999 to choose settings, 1000-1299 for the scores above (1000-1099 on Isaac Sim). They
  never overlap the demonstration seeds.

## Dataset names in `training_meta.json`

`training_meta.json` records the local folder names of the demonstrations the model was trained on:

| Name | What it is |
| --- | --- |
| `randomized_v1` | 200 expert demonstrations on MuJoCo, seeds 0-199, the study's randomized pick-and-place task |
| `m5_visual_dr` | 400 expert demonstrations on MuJoCo, seeds 0-399, with the look (colours, floor tiles, lighting) randomized in every episode |
| `randomized_isaac_v1`, `randomized_isaac_v1b` | the actions of `randomized_v1` replayed on Isaac Sim with Isaac's frames (seeds 0-99 and 100-199) |
| `randomized_isaac_v2a`, `randomized_isaac_v2b` | the same for 200 further MuJoCo demonstrations (seeds 200-399), used by `act_isaac_only_400` |

## What is not here

- The demonstrations. Those from MuJoCo are regenerated from their seeds with the study's collection script. The
  Isaac Sim replays should not be expected to regenerate identically (Isaac runs were not deterministic in a test), which matters for the
  `act_both_renderers_*` and `act_isaac_only_*` models.
- Checkpoints of runs that are not in a table of the study (a first Diffusion Policy version that was replaced, and
  early ablations).

Reproducing the scores:

- ACT on MuJoCo: the same episodes again, step for step, given the same versions of MuJoCo, lerobot and the study's
  code (checked on one downloaded checkpoint: 10 of 10 seeds identical). Newton (MuJoCo physics, Newton frames) was not
  rechecked.
- Diffusion Policy: the policy samples random noise at every prediction and the evaluation does not fix the
  torch seed, so two runs of the same checkpoint on the same seeds differ in the details of an episode. Four runs of
  `diffusion_policy_v2_nominal_seed0` on the MuJoCo seeds 1000-1299 scored 290, 291, 292 and 293 of 300 (standard
  deviation 1.3 episodes); the Newton and Isaac Sim scores were not rerun.
- Isaac Sim, for every model: PhysX is not deterministic, so success rates reproduce within sampling noise.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def private_strings(folder: Path) -> list[str]:
    """Local paths or user names left in the small JSON files of a checkpoint."""
    hits = []
    for name in REQUIRED + OPTIONAL:
        path = folder / name
        if path.suffix == ".json" and path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(
                rf"[A-Za-z]:[\\/]+Users|/home/|{re.escape(getpass.getuser())}", text
            ):
                hits.append(name)
    return hits


def recipe_problems(model: Model, folder: Path) -> list[str]:
    """The recipe text must state the demonstration seeds and the network seed the metadata recorded."""
    path = folder / "training_meta.json"
    if not path.exists():
        return []
    meta = json.loads(path.read_text(encoding="utf-8"))
    seeds = meta["train_seeds"]
    found = []
    claimed = re.search(r"episode seeds (\d+)-(\d+)", model.recipe)
    if not claimed or (int(claimed[1]), int(claimed[2])) != (min(seeds), max(seeds)):
        found.append(
            f"{model.name}: recipe seeds differ from the metadata {min(seeds)}-{max(seeds)}"
        )
    claimed_seed = re.search(r"network seed (\d+|not set)", model.recipe)
    recorded = meta.get("training_seed")
    expected = "not set" if recorded is None else str(recorded)
    suffix = "_unseeded" if recorded is None else f"_seed{recorded}"
    if not model.name.endswith(suffix):
        found.append(f"{model.name}: the folder name should end in {suffix}")
    if not claimed_seed or claimed_seed[1] != expected:
        found.append(
            f"{model.name}: recipe network seed differs from the metadata ({expected})"
        )
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--tier", type=int, choices=(1, 2), default=2, help="2 = tiers 1 and 2"
    )
    ap.add_argument(
        "--card-only",
        action="store_true",
        help="write only README.md (the checkpoints were staged earlier and their run folders are gone)",
    )
    args = ap.parse_args()
    chosen = [m for m in MODELS if m.tier <= args.tier]
    if args.card_only:
        args.out.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(
            f"| `{m.name}` | {m.recipe} | {m.mujoco} | {m.newton} | {m.isaac} |"
            for m in chosen
        )
        card = CARD.format(rows=rows, repo=REPO)
        (args.out / "README.md").write_text(card, encoding="utf-8")
        print(f"README.md for {len(chosen)} checkpoints -> {args.out}")
        return 0
    problems = []
    for model in chosen:
        folder = RUNS / model.source
        missing = [n for n in REQUIRED if not (folder / n).exists()]
        if missing:
            problems.append(f"{model.name}: missing {missing} in {folder}")
        problems += recipe_problems(model, folder)
        leaks = private_strings(folder)
        if leaks:
            problems.append(f"{model.name}: local path or user name in {leaks}")
    if problems:
        print("\n".join(problems))
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    listing = []
    for model in chosen:
        source, target = RUNS / model.source, args.out / model.name
        target.mkdir(exist_ok=True)
        for name in REQUIRED + OPTIONAL:
            if (source / name).exists():
                shutil.copy2(source / name, target / name)
        weights = target / "model.safetensors"
        meta = json.loads((target / "training_meta.json").read_text(encoding="utf-8"))
        listing.append(
            {
                "name": model.name,
                "algorithm": meta.get("algorithm", "act"),
                "recipe": model.recipe,
                "demonstration_episode_seeds": f"{min(meta['train_seeds'])}-{max(meta['train_seeds'])}",
                "demonstration_seeds_count": len(meta["train_seeds"]),
                "training_steps": meta["steps"],
                "network_seed": meta.get("training_seed"),
                "batch_size": meta["batch_size"],
                "learning_rate": meta["lr"],
                "scores": {
                    "mujoco_seeds_1000_1299": model.mujoco,
                    "newton_seeds_1000_1299": model.newton,
                    "isaac_sim_seeds_1000_1099": model.isaac,
                },
                "megabytes": round(weights.stat().st_size / 1e6, 1),
                "sha256": sha256(weights),
            }
        )
        print(f"{model.name:40s} {listing[-1]['megabytes']:7.1f} MB")
    (args.out / "checkpoints.json").write_text(
        json.dumps(listing, indent=2), encoding="utf-8"
    )
    rows = "\n".join(
        f"| `{m.name}` | {m.recipe} | {m.mujoco} | {m.newton} | {m.isaac} |"
        for m in chosen
    )
    (args.out / "README.md").write_text(
        CARD.format(rows=rows, repo=REPO), encoding="utf-8"
    )
    total = sum(item["megabytes"] for item in listing)
    print(f"{len(listing)} checkpoints, {total / 1000:.2f} GB of weights -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
