"""Stage the study-01-v2 checkpoints for Hugging Face (the V1 ones are already in the repo; see stage_checkpoints.py).

    python research/scripts/stage_checkpoints_v2.py --out ~/hf_staging_v2

`--out` gets one subfolder per V2 checkpoint (the files the policy loader needs), `checkpoints_v2.json` (recipe, seeds,
scores, sha256 of the weights), `README_v2.md` (a section for the model card, to be merged by hand: the repo's README is
the user's) and `results_v2/<model>/<engine>.json` (the per-seed success lists the paired tests are recomputed from).
Scores are counted from those per-seed files, not typed. Nothing is uploaded.

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from stage_checkpoints import OPTIONAL, REQUIRED, private_strings, sha256

V2 = Path(__file__).resolve().parents[2] / "outputs" / "study01" / "v2"
ACT = "ACT, {data}, chunk 100, {steps}k steps, network seed {seed}"
NOMINAL = "{n} MuJoCo demos (episode seeds 0-{last}), nominal look"
RANDOM = "{n} MuJoCo demos (episode seeds 0-{last}), look randomized in every episode"


@dataclass(frozen=True)
class Model:
    name: str
    source: str  # under outputs/study01/v2
    recipe: str
    seed: int
    # engine -> files or folders (Isaac shard folders) under V2 holding per-seed results
    scores: dict[str, tuple[str, ...]] = field(default_factory=dict)


def act(data: str, steps: int, seed: int) -> str:
    return ACT.format(data=data, steps=steps, seed=seed)


def n1(seed: int) -> Model:
    return Model(
        f"act_nominal400_seed{seed}",
        f"act_nominal_400_120k_seed{seed}",
        act("400 MuJoCo demos (episode seeds 0-399), nominal look", 120, seed),
        seed,
        {
            "mujoco": (f"act_nominal_400_120k_seed{seed}_mujoco_1000-1299.json",),
            "newton": (f"newton_rescore/nominal400_seed{seed}.json",),
            "isaac": (
                f"isaac_n1/nominal400_seed{seed}",
                f"isaac_n2lr/nominal400_seed{seed}",
            ),
        },
    )


def n2(kind: str, seed: int) -> Model:
    if kind == "nominal":
        return Model(
            f"act_nominal_seed{seed}",
            f"act_nominal_200_60k_seed{seed}",
            act(NOMINAL.format(n=200, last=199), 60, seed),
            seed,
            {
                "mujoco": (f"n2/nominal200_seed{seed}_mujoco.json",),
                "newton": (f"n2/nominal200_seed{seed}_newton.json",),
                "isaac": (
                    f"isaac_n2/nominal200_seed{seed}",
                    f"isaac_n2lr/nominal200_seed{seed}",
                ),
            },
        )
    return Model(
        f"act_randomized_look_seed{seed}",
        f"act_randomized_look_120k_seed{seed}",
        act(RANDOM.format(n=400, last=399), 120, seed),
        seed,
        {
            "mujoco": (f"n2/randlook_seed{seed}_mujoco.json",),
            "newton": (f"n2/randlook_seed{seed}_newton.json",),
            "isaac": (f"isaac_n2/randlook_seed{seed}",),
        },
    )


def n4(kind: str, seed: int) -> Model:
    nominal = kind == "nominal"
    short = "nominal" if nominal else "randlook"
    return Model(
        f"viperx_act_{'nominal' if nominal else 'randomized_look'}_seed{seed}",
        f"n4/act_{'nominal' if nominal else 'randlook'}_seed{seed}",
        "ViperX 300 S, "
        + act((NOMINAL if nominal else RANDOM).format(n=400, last=399), 120, seed),
        seed,
        {
            "mujoco": (f"n4/{short}_seed{seed}_mujoco.json",),
            "newton": (f"n4/{short}_seed{seed}_newton.json",),
            "newton_heldout_look": (f"n5_viperx/{short}_seed{seed}_heldout.json",),
            "isaac": (f"isaac_n4/{short}_seed{seed}",),
        },
    )


def n5(seed: int) -> Model:
    return Model(
        f"act_randomized_look_bw_seed{seed}",
        f"act_randlook_bw_120k_seed{seed}",
        act(
            "400 MuJoCo demos (episode seeds 0-399), look randomized in every episode and the floor tiles black and white in 30% of them",
            120,
            seed,
        ),
        seed,
        {
            "mujoco": (f"n5/randlook_bw_seed{seed}_mujoco.json",),
            "newton": (f"n5/randlook_bw_seed{seed}_newton.json",),
            "newton_heldout_look": (f"n5/randlook_bw_seed{seed}_heldout.json",),
            "isaac": (f"isaac_n5/randlook_bw_seed{seed}",),
        },
    )


MODELS = (
    *(n1(s) for s in (1, 2, 3)),
    n2("nominal", 4),
    n2("nominal", 5),
    n2("randlook", 3),
    n2("randlook", 4),
    n4("nominal", 1),
    n4("nominal", 2),
    n4("randlook", 1),
    n4("randlook", 2),
    *(n5(s) for s in (1, 2, 3)),
    *(
        Model(
            f"diffusion_policy_v2_nominal_seed{s}",
            f"n6/dp_nominal_100k_seed{s}",
            f"Diffusion Policy, {NOMINAL.format(n=200, last=199)}, chunk 64, 100k steps, network seed {s} (add --policy-arg n_action_steps=16)",
            s,
            {
                "mujoco": (f"n6/dp_seed{s}_mujoco.json",),
                "newton": (f"n6/dp_seed{s}_newton.json",),
                "isaac": (f"isaac_n6/dp_seed{s}",),
            },
        )
        for s in (2, 3)
    ),
    Model(
        "act_pad_visible_seed1",
        "n7/act_padvisible_200_60k_seed1",
        act(
            "200 MuJoCo demos (episode seeds 0-199), nominal look, the two contact pads drawn opaque",
            60,
            1,
        ),
        1,
        {
            "mujoco_pad_visible": ("n7/C2_mujoco_pad.json",),
            "mujoco": ("n7/C1_mujoco_nopad.json",),
            "newton_pad_visible": ("n7/C3_newton_pad.json",),
            "newton": ("n7/C4_newton_nopad.json",),
        },
    ),
)


def per_seed(paths: tuple[str, ...]) -> dict[int, bool]:
    found: dict[int, bool] = {}
    for rel in paths:
        path = V2 / rel
        # a shard folder holds shard-N.json, or only merged.json where the shard file was folded into it
        files = (
            [path]
            if path.is_file()
            else sorted([*path.glob("s*/shard-*.json"), *path.glob("s*/merged.json")])
        )
        for f in files:
            for episode in json.loads(f.read_text(encoding="utf-8"))["results"]:
                seed, ok = int(episode["seed"]), bool(episode["success"])
                if found.setdefault(seed, ok) != ok:
                    raise SystemExit(
                        f"seed {seed} has two different outcomes under {rel}"
                    )
    return found


def describe(results: dict[int, bool]) -> str:
    return (
        f"{sum(results.values())}/{len(results)} (seeds {min(results)}-{max(results)})"
        if results
        else "none"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    problems = []
    for m in MODELS:
        folder = V2 / m.source
        missing = [n for n in REQUIRED if not (folder / n).exists()]
        if missing:
            problems.append(f"{m.name}: missing {missing} in {folder}")
        elif private_strings(folder):
            problems.append(
                f"{m.name}: local path or user name in {private_strings(folder)}"
            )
        else:
            meta = json.loads(
                (folder / "training_meta.json").read_text(encoding="utf-8")
            )
            if meta.get("training_seed") != m.seed:
                problems.append(
                    f"{m.name}: network seed {meta.get('training_seed')} in the metadata, {m.seed} in the name"
                )
            if not m.name.endswith(f"_seed{m.seed}"):
                problems.append(
                    f"{m.name}: the folder name should end in _seed{m.seed}"
                )
        for engine, paths in m.scores.items():
            if not per_seed(paths):
                problems.append(
                    f"{m.name}: no per-seed results for {engine} in {paths}"
                )
    if problems:
        print("\n".join(problems))
        return 1
    listing, rows = [], []
    for m in MODELS:
        source, target = V2 / m.source, args.out / m.name
        target.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED + OPTIONAL:
            if (source / name).exists():
                shutil.copy2(source / name, target / name)
        meta = json.loads((target / "training_meta.json").read_text(encoding="utf-8"))
        scores = {}
        for engine, paths in m.scores.items():
            results = per_seed(paths)
            scores[engine] = describe(results)
            out = args.out / "results_v2" / m.name
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{engine}.json").write_text(
                json.dumps({str(k): v for k, v in sorted(results.items())}),
                encoding="utf-8",
            )
        weights = target / "model.safetensors"
        listing.append(
            {
                "name": m.name,
                "algorithm": meta.get("algorithm", "act"),
                "recipe": m.recipe,
                "demonstration_episode_seeds": f"{min(meta['train_seeds'])}-{max(meta['train_seeds'])}",
                "training_steps": meta["steps"],
                "network_seed": meta["training_seed"],
                "scores": scores,
                "megabytes": round(weights.stat().st_size / 1e6, 1),
                "sha256": sha256(weights),
            }
        )
        rows.append(
            f"| `{m.name}` | {m.recipe} | "
            + "; ".join(f"{e} {s}" for e, s in scores.items())
            + " |"
        )
        print(f"{m.name:42s} {listing[-1]['megabytes']:7.1f} MB  {scores}")
    (args.out / "checkpoints_v2.json").write_text(
        json.dumps(listing, indent=2), encoding="utf-8"
    )
    (args.out / "README_v2.md").write_text(
        "## study-01-v2 checkpoints\n\nScores count the per-seed files in `results_v2/`. Engines: MuJoCo and Newton on seeds "
        "1000-1299 (Newton renders MuJoCo physics), Isaac Sim on the seeds shown.\n\n| Folder | Recipe | Scores (successes/episodes) |\n"
        "| --- | --- | --- |\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    print(
        f"{len(listing)} checkpoints, {sum(i['megabytes'] for i in listing) / 1000:.2f} GB -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
