"""V3 P1: the cube start of Isaac Sim's own scoring runs, as `cube_start` files for the MuJoCo envs.

The idle measurement of cube_start_table.py depends on the order (decide exits 2), so the start is taken from the runs that already scored each model with
Isaac Sim's physics: N13 (SO-101, seeds 1000-1099, `im/`) and N14 (ViperX, seeds 1000-1299). Each episode records `cube_start` (xy after reset, 0.1 mm) in that run's
own order. One file per model; the shift against the MuJoCo run of the same seeds is printed.

    python research/experiments/p1_tables.py
"""

import json
from pathlib import Path

import numpy as np

STUDY = Path("outputs/study01")
OUT = Path("outputs/study01")

# name: (Isaac-physics shards, MuJoCo file of the same seeds, seeds)
MODELS = {
    "so101_nominal_seed1": ("n13/act_nominal_seed1", "n12/act_nominal_seed1.none.json"),
    "so101_nominal_seed2": ("n13/act_nominal_seed2", "n12/act_nominal_seed2.none.json"),
    "so101_randlook_seed0": (
        "n13/act_randomized_look_seed0",
        "n12/act_randomized_look_seed0.none.json",
    ),
    "viperx_nominal_seed1": ("n14/im/nominal_seed1", "n4/nominal_seed1_mujoco.json"),
    "viperx_nominal_seed2": ("n14/im/nominal_seed2", "n4/nominal_seed2_mujoco.json"),
}


def episodes(shards: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(shards.glob("s*/merged.json")):
        for e in json.loads(f.read_text(encoding="utf-8"))["results"]:
            out[e["seed"]] = e
    return out


for name, (isaac_dir, mujoco_file) in MODELS.items():
    isaac = episodes(STUDY / isaac_dir)
    mj = {
        e["seed"]: e
        for e in json.loads((STUDY / mujoco_file).read_text(encoding="utf-8"))[
            "results"
        ]
    }
    seeds = sorted(set(isaac) & set(mj))
    shift = (
        np.array(
            [np.subtract(isaac[s]["cube_start"], mj[s]["cube_start"]) for s in seeds]
        )
        * 1000
    )
    # the same pad: the target is the same in both runs, only the cube is moved
    same_target = all(isaac[s]["target_pos"] == mj[s]["target_pos"] for s in seeds)
    print(
        f"{name}: {len(isaac)} Isaac, {len(mj)} MuJoCo, {len(seeds)} paired seeds; Isaac-MuJoCo cube xy mm: mean {shift.mean(0).round(2)}, "
        f"abs median {np.median(np.abs(shift), axis=0).round(2)}, max {np.abs(shift).max(0).round(1)}, "
        f"{int((np.abs(shift).max(1) > 1).sum())} seeds off by more than 1 mm; target equal: {same_target}"
    )
    table = {
        "source": f"Isaac Sim physics scoring run {isaac_dir} (cube_start after reset, 0.1 mm), seeds {seeds[0]}-{seeds[-1]}, in that run's own order",
        "xy_by_seed": {str(s): isaac[s]["cube_start"] for s in seeds},
    }
    (Path("research/experiments") / f"cube_start_{name}.json").write_text(
        json.dumps(table, indent=1), encoding="utf-8"
    )
