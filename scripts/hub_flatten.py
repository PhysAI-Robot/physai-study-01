"""Write the Hugging Face card and `checkpoints.json` of rizgiak/physai-study-01-checkpoints as one release, without the history of how the study grew.

    python research/scripts/hub_flatten.py --staging ~/.physai/hf_staging_v3 --v2-json ~/.physai/hub_checkpoints_v2.json --out ~/.physai/hf_staging_flat

`--staging` has the card, `checkpoints.json` (first group), `checkpoints_v3.json` and `results/INDEX.tsv` as they are on the Hub; `--v2-json` is the
Hub's `checkpoints_v2.json`. `--out` gets `README.md` and `checkpoints.json`: one list of the 45 folders with one set of keys, the scores counted from
`results/INDEX.tsv` (so they equal the per-seed files), and a card that groups the folders by what the model is (ACT on the SO-101, Diffusion Policy,
ACT on the ViperX 300 S, SmolVLA, runs without weights) instead of by when they were made. The upload then deletes `checkpoints_v2.json` and
`checkpoints_v3.json`. Nothing is uploaded here.

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CODE = "https://github.com/PhysAI-Robot/physai-study-01"
MAIN = {
    "mujoco",
    "newton",
    "isaac",
    "newton_old_renderer",
    "mujoco_physics_isaac_pictures",
    "isaac_physics_mujoco_pictures",
    "validation_mujoco_seeds_900_999",
}
HIDDEN = (
    "mujoco_seeds_",
    "newton_look_",
)  # the sensitivity pilot and the looks: in the result files, not in a table cell
ACT_ROWS = (
    "act_nominal_unseeded act_nominal_seed1 act_nominal_seed2 act_nominal_seed3 act_nominal_seed4 act_nominal_seed5 "
    "act_nominal400_seed1 act_nominal400_seed2 act_nominal400_seed3 "
    "act_randomized_look_seed0 act_randomized_look_seed1 act_randomized_look_seed2 act_randomized_look_seed3 act_randomized_look_seed4 "
    "act_randomized_look_bw_seed1 act_randomized_look_bw_seed2 act_randomized_look_bw_seed3 "
    "act_both_renderers_unseeded act_both_renderers_seed1 "
    "act_isaac_only_100_unseeded act_isaac_only_200_unseeded act_isaac_only_200_seed1 act_isaac_only_400_unseeded "
    "act_chunk30_100demos_unseeded act_chunk100_100demos_unseeded act_pad_visible_seed1 act_nominal_color_jitter_unseeded"
).split()
DP_ROWS = "diffusion_policy_v2_nominal_seed0 diffusion_policy_v2_nominal_seed2 diffusion_policy_v2_nominal_seed3 diffusion_policy_v2_randomized_look_seed0".split()
VIPERX_ROWS = "viperx_act_nominal_seed1 viperx_act_nominal_seed2 viperx_act_randomized_look_seed1 viperx_act_randomized_look_seed2".split()


def section(card: str, title: str) -> str:
    """The text of the `## title...` section of the old card, without its heading line."""
    match = re.search(
        r"^## " + re.escape(title) + r"[^\n]*\n(.*?)(?=^## |\Z)", card, re.S | re.M
    )
    if not match:
        raise SystemExit(f"no section {title!r} in the card")
    return match.group(1).strip("\n")


def edit(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{text.count(old)} copies of: {old[:70]!r}")
    return text.replace(old, new)


def scores_from_index(
    rows: list[list[str]],
) -> dict[str, dict[str, tuple[int, int, str]]]:
    out: dict[str, dict[str, tuple[int, int, str]]] = {}
    for name, file, successes, episodes, seeds, _ in rows:
        out.setdefault(name, {})[file.removesuffix(".json")] = (
            int(successes),
            int(episodes),
            seeds,
        )
    return out


def text_of(counts: tuple[int, int, str]) -> str:
    return f"{counts[0]}/{counts[1]} (seeds {counts[2]})"


def cell(counts: dict[str, tuple[int, int, str]], stem: str) -> str:
    return f"{counts[stem][0]}/{counts[stem][1]}" if stem in counts else "-"


def other(counts: dict[str, tuple[int, int, str]], skip: set[str]) -> str:
    items = [
        f"{stem} {c[0]}/{c[1]}"
        for stem, c in sorted(counts.items())
        if stem not in skip and not stem.startswith(HIDDEN)
    ]
    return "; ".join(items) or "-"


def table(header: str, rows: list[list[str]]) -> str:
    aligns = (
        ["---"]
        + ["---"]
        + [
            "---:" if re.match(r"^(MuJoCo|Newton|Isaac|MI|IM|Validation)", h) else "---"
            for h in header.split("|")[2:]
        ]
    )
    return "\n".join(
        [
            f"| {header.replace('|', ' | ')} |",
            "| " + " | ".join(a.strip() for a in aligns) + " |",
            *("| " + " | ".join(r) + " |" for r in rows),
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--v2-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    st = args.staging
    first = json.loads((st / "checkpoints.json").read_text(encoding="utf-8"))
    second = json.loads(args.v2_json.read_text(encoding="utf-8"))
    third = json.loads((st / "checkpoints_v3.json").read_text(encoding="utf-8"))
    index = [
        line.split("\t")
        for line in (st / "results" / "INDEX.tsv")
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    ]
    counts = scores_from_index(index)
    old_card = (st / "README.md").read_text(encoding="utf-8")

    # 1  checkpoints.json: one list, one set of keys
    entries = []
    for e in [*first, *second, *third["checkpoints"]]:
        name = e["name"]
        c = counts[name]
        scores = {
            stem: text_of(v)
            for stem, v in c.items()
            if stem not in ("newton_old_renderer",) and not stem.startswith(HIDDEN)
        }
        before = e["scores"].get("newton_seeds_1000_1299") if e in first else None
        if before:
            scores["newton_before_fix"] = before + " (seeds 1000-1299)"
        keep = {k: v for k, v in e.items() if k not in ("scores", "results")}
        keep["scores"] = scores
        entries.append(keep)
    if len(entries) != 45 or len({e["name"] for e in entries}) != 45:
        raise SystemExit("expected 45 distinct folders")
    by_name = {e["name"]: e for e in entries}
    covered = (
        set(ACT_ROWS)
        | set(DP_ROWS)
        | set(VIPERX_ROWS)
        | {e["name"] for e in third["checkpoints"] if e["algorithm"] == "smolvla"}
    )
    if covered != set(by_name):
        raise SystemExit(
            f"table rows differ from the folders: {sorted(covered ^ set(by_name))}"
        )

    # 2  the card
    def recipe(name: str) -> str:
        return re.sub(
            r"Diffusion Policy v2,", "Diffusion Policy,", by_name[name]["recipe"]
        )

    def standard(names: list[str], with_before: bool) -> str:
        head = (
            "Folder|Recipe|MuJoCo|Newton (after the fix)|"
            + ("Newton (before the fix)|" if with_before else "")
            + "Isaac Sim|Other"
        )
        rows = []
        for n in names:
            c, s = counts[n], by_name[n]["scores"]
            before = s.get("newton_before_fix", "-").split(" (")[0]
            row = [f"`{n}`", recipe(n), cell(c, "mujoco"), cell(c, "newton")]
            row += [before] if with_before else []
            row += [
                cell(c, "isaac"),
                other(c, {"mujoco", "newton", "isaac", "newton_old_renderer"}),
            ]
            rows.append(row)
        return table(head, rows)

    smol = [e["name"] for e in third["checkpoints"] if e["algorithm"] == "smolvla"]
    smol_rows = []
    for n in smol:
        c = counts[n]
        smol_rows.append(
            [
                f"`{n}`",
                by_name[n]["recipe"],
                cell(c, "mujoco"),
                cell(c, "newton"),
                cell(c, "isaac"),
                cell(c, "mujoco_physics_isaac_pictures"),
                cell(c, "isaac_physics_mujoco_pictures"),
                cell(c, "validation_mujoco_seeds_900_999"),
                other(c, MAIN),
            ]
        )
    smol_table = table(
        "Folder|Recipe|MuJoCo|Newton|Isaac Sim|MI|IM|Validation|Other", smol_rows
    )
    no_weights = []
    for v in third["validation_only_runs"]:
        c = counts[v["name"]]["validation_mujoco_seeds_900_999"]
        no_weights.append([f"`{v['name']}`", v["recipe"], f"{c[0]}/{c[1]}"])
    no_weights_table = table("Name in `results/`|Recipe|Validation", no_weights)

    head = old_card.split("\n# ", 1)[0]  # the front matter
    how = section(old_card, "How to read the scores")
    how = edit(
        how,
        "the third part publishes\nper-seed validation files, see its section)",
        'per-seed validation files of the runs\nwithout weights are published, see "Runs without weights")',
    )
    how = edit(
        how,
        "**The Newton\n  renderer was fixed during the second part:** it drew",
        "**The Newton\n  renderer had a fault that was fixed:** it drew",
    )
    how = edit(
        how,
        'Every Newton score of the second part, and the column "Newton (after the fix)" below, uses the fixed renderer. The\n  column "Newton (before the fix)" is what the first part reported with the renderer as it was; it is kept as a sensitivity analysis,\n  and 400-demonstration and randomized-look models move by 5 episodes or less.',
        'Every Newton score here, and the column "Newton (after the fix)" below, uses the fixed renderer. The\n  column "Newton (before the fix)" is what the earliest runs reported with the renderer as it was; it is kept as a sensitivity analysis,\n  and 400-demonstration and randomized-look models move by 5 episodes or less.',
    )
    how = edit(
        how,
        "The\n  first part, like every other Isaac Sim score here, was taken on Isaac Sim 6.2.0.0\n  (driver 617.42, Windows build 26300), the one stack of the paper. PhysX is not\n  deterministic, so an Isaac Sim score reproduces only within sampling noise.",
        "Every Isaac Sim score here was taken on Isaac Sim 6.2.0.0\n  (driver 617.42, Windows 11 build 26300, RTX 3060 Laptop 6 GB), the one stack of the paper (MuJoCo 3.12.0; Newton 1.6.1 with mujoco-warp 3.12.0 and\n  warp-lang 1.18.0; torch 2.11.0+cu128; lerobot 0.6.1; numpy 2.2.6; transformers 5.5.4 for SmolVLA). PhysX is not\n  deterministic, so an Isaac Sim score reproduces only within sampling noise.",
    )

    third_text = section(old_card, "Third part")
    recipe_start = third_text.index("**SmolVLA recipe**")
    recipe_end = third_text.index(
        "| Folder | Recipe | MuJoCo | Newton | Isaac Sim | MI | IM | Validation | Other |"
    )
    smol_intro = third_text[recipe_start:recipe_end].rstrip("\n")
    notes_start = third_text.index("Notes on the third part:")
    notes = third_text[notes_start:]
    notes = edit(
        notes,
        "Notes on the third part:",
        "Notes on the SmolVLA folders and the runs without weights:",
    )
    notes = edit(
        notes,
        "The validation seeds are published for this part because",
        "The validation seeds are published for these runs because",
    )
    wv_start = third_text.index("**Validation-only runs.**")
    wv_end = third_text.index("| Name in `results/` |")
    no_weights_intro = (
        third_text[wv_start:wv_end]
        .rstrip("\n")
        .replace("**Validation-only runs.** The runs below", "The runs below")
    )

    second_text = section(old_card, "Second part")
    second_notes = second_text[second_text.index("Notes on the second part:") :]
    second_notes = edit(
        second_notes,
        "Notes on the second part:",
        "Notes on the ACT and ViperX folders:",
    )
    second_notes = second_notes[: second_notes.index("Each folder holds")].rstrip()

    result_files = section(old_card, "Result files")
    result_files = edit(
        result_files,
        "for **all 34 folders of the first and second part** (the third part's files are described in its section):",
        "for **all 45 folders**:",
    )
    result_files = edit(
        result_files,
        " The files were counted from the per-episode result files of the evaluations, and for the first part's 17 folders every\nMuJoCo and Isaac Sim file was checked against the score in `checkpoints.json`.",
        " The files were counted from the per-episode result files of the evaluations.",
    )
    result_files = edit(
        result_files,
        "only for the first part's folders where that score has a file (7 of 17); the other first-part Newton scores in `checkpoints.json` have no per-seed file that matches their count",
        "only for the 7 folders where that score has a per-seed file; for the other earliest runs the count is `newton_before_fix` in `checkpoints.json`",
    )
    result_files = edit(
        result_files,
        "- The first-part MuJoCo files and the unedited rerun",
        "- The MuJoCo files and the unedited rerun",
    )
    result_files = edit(
        result_files,
        "- Other first-part experiments (sweeps",
        "- Other experiments (sweeps",
    )

    use = section(old_card, "Use")
    use = edit(use, " (the second part is commit `ece89d5` and later)", "")
    dataset_names = section(old_card, "Dataset names in `training_meta.json`")
    not_here = section(old_card, "What is not here")
    not_here = edit(
        not_here,
        "(the `diffusion_policy_v2_*` folders are its replacement, not the second part)",
        "(the `diffusion_policy_v2_*` folders are its replacement)",
    )

    old_title_block = old_card.split("\n# ", 1)[1].split("\n## ", 1)[0]
    bullets = old_title_block[old_title_block.index("- Task:") :].strip("\n")
    intro = (
        "Checkpoints behind the numbers of study 1 (how much a learned policy loses when only the renderer changes; ACT, Diffusion Policy and\n"
        "SmolVLA; MuJoCo, Isaac Sim and Newton as the renderer). One subfolder per checkpoint, 45 in all: ACT on the SO-101 (27), Diffusion Policy on the SO-101 (4),\n"
        "ACT on the ViperX 300 S (4) and SmolVLA on the SO-101 (10). The code that produced them is at "
        f"{CODE} (tag `study-01`), checked out as the folder `research/` of the physai framework, so the `research/...` paths below are relative to the framework's root."
    )
    card = f"""{head}
# Study 1 checkpoints: randomized pick-and-place on the SO-101 and the ViperX 300 S

{intro}

{bullets}

## How to read the scores

{how}

## Checkpoints

Each folder holds `model.safetensors`, `config.json`, `dataset_stats.json`, `training_meta.json` and `checkpoint_meta.json` (and `training_log.json`).
`checkpoints.json` lists the recipe, the seeds, the scores and the sha256 of every weights file. The scores are counted from the per-seed files in
`results/<folder>/` (see "Result files" below), so the paired tests of the study (exact McNemar, rank tests) can be recomputed. A score is successes over
episodes; the number of episodes says which seeds were scored.

### ACT on the SO-101 (27 folders)

{standard(ACT_ROWS, True)}

{second_notes}

### Diffusion Policy on the SO-101 (4 folders)

Every Diffusion Policy folder plays 16 of its 64 predicted actions: add `--policy-arg n_action_steps=16`.

{standard(DP_ROWS, True)}

### ACT on the ViperX 300 S (4 folders)

{standard(VIPERX_ROWS, False)}

### SmolVLA on the SO-101 (10 folders)

{smol_intro}

{smol_table}

### Runs without weights (9 names in `results/`)

{no_weights_intro}

{no_weights_table}

{notes}

## Result files

{result_files}

## Use

{use}

## Dataset names in `training_meta.json`

{dataset_names}

## What is not here

{not_here}
"""
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "README.md").write_text(card, encoding="utf-8")
    (args.out / "checkpoints.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(entries)} folders; card {len(card.splitlines())} lines -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
