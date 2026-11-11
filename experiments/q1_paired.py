"""Paired MuJoCo-vs-Isaac Sim table of the paper (seeds 1000-1099) on the old (6.1.0.0) and the new (6.2.0.0) per-seed files.
Old files are read only to check that this script reproduces the V2 paper's table; the paper takes the new column."""

import glob, json, os
from math import comb

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)


def load(files, lo=1000, hi=1100):
    out = {}
    for f in files:
        d = json.load(open(f))
        for e in d["results"]:
            if lo <= e["seed"] < hi:
                out[e["seed"]] = bool(e["success"])
    return out


def g(p):
    return sorted(glob.glob(p))


def mcn(b, c):
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


ROWS = [  # label, mujoco files, old isaac files, new isaac files
    (
        "chunk30 100 demos",
        [f"{STUDY}/evals/act_100_mujoco_1000-1299.json"],
        [f"{STUDY}/isaac/act_100_isaac_1000-1099/merged.json"],
        g(f"{STUDY}/q1/B_chunk/act_chunk30_100demos_unseeded/s*/shard-*.json"),
    ),
    (
        "chunk100 100 demos 60k",
        [f"{STUDY}/evals/act_100_c100_60k_mujoco_1000-1299.json"],
        [f"{STUDY}/isaac/act_100_c100_60k_isaac_1000-1099/merged.json"],
        g(f"{STUDY}/q1/B_chunk/act_chunk100_100demos_unseeded/s*/shard-*.json"),
    ),
    (
        "final ACT run 1",
        [f"{STUDY}/evals/act_200_c100_60k_mujoco_1000-1299.json"],
        [f"{STUDY}/isaac/act_200_c100_60k_isaac_1000-1099/merged.json"],
        g(f"{STUDY}/isaac_n2/bridge_act_nominal_unseeded/s*/shard-*.json"),
    ),
    (
        "final ACT seed 1",
        [f"{STUDY}/experiments/e5/seed1_mujoco.json"],
        [],
        g(f"{STUDY}/isaac_n2/bridge_act_nominal_seed1/s*/shard-*.json"),
    ),
    (
        "final ACT seed 2",
        [f"{STUDY}/experiments/e5/seed2_mujoco.json"],
        [],
        g(f"{STUDY}/isaac_n2/bridge_act_nominal_seed2/s*/shard-*.json"),
    ),
    (
        "final ACT seed 3",
        [f"{STUDY}/evals/act_200_seed3_mujoco.json"],
        [f"{STUDY}/isaac/act_200_seed3_isaac/merged.json"],
        g(f"{STUDY}/isaac_n2/bridge_act_nominal_seed3/s*/shard-*.json"),
    ),
    (
        "Isaac replays 100",
        [f"{STUDY}/evals/act_isaac_100_mujoco_1000-1299.json"],
        [f"{STUDY}/isaac/act_isaac_100_isaac_1000-1099/merged.json"],
        g(f"{STUDY}/q1/C_isaac_only/act_isaac_only_100_unseeded/s*/shard-*.json"),
    ),
    (
        "Isaac replays 200",
        [f"{STUDY}/evals/act_isaac_200_mujoco_1000-1299.json"],
        [f"{STUDY}/isaac/act_isaac_200_isaac_1000-1099/merged.json"],
        g(f"{STUDY}/q1/C_isaac_only/act_isaac_only_200_unseeded/s*/shard-*.json"),
    ),
    (
        "Isaac replays 200 seed 1",
        [f"{STUDY}/evals/act_isaac_200_seed1-eval/mujoco.json"],
        [f"{STUDY}/evals/act_isaac_200_seed1-eval/isaac/merged.json"],
        g(f"{STUDY}/q1/C_isaac_only/act_isaac_only_200_seed1/s*/shard-*.json"),
    ),
    (
        "Isaac replays 400",
        [f"{STUDY}/evals/act_isaac_400_mujoco_1000-1299.json"],
        [f"{STUDY}/isaac/act_isaac_400_isaac_1000-1099/merged.json"],
        g(f"{STUDY}/q1/C_isaac_only/act_isaac_only_400_unseeded/s*/shard-*.json"),
    ),
    (
        "both renderers run 1",
        [f"{STUDY}/evals/act_mixed_400_mujoco_1000-1299.json"],
        [f"{STUDY}/evals/act_mixed_400_isaac_1000-1099_merged.json"],
        g(f"{STUDY}/q1/D_both_renderers/act_both_renderers_unseeded/s*/shard-*.json"),
    ),
    (
        "both renderers seed 1",
        [f"{STUDY}/evals/act_mixed_400_seed1-eval/mujoco.json"],
        [f"{STUDY}/evals/act_mixed_400_seed1-eval/isaac/merged.json"],
        g(f"{STUDY}/q1/D_both_renderers/act_both_renderers_seed1/s*/shard-*.json"),
    ),
    (
        "randomized look seed 0",
        [f"{STUDY}/experiments/e3/mujoco_1000-1299.json"],
        [f"{STUDY}/evals/e3_isaac_merged_1000-1099.json"],
        g(f"{STUDY}/isaac_n1/randlook_seed0/s*/shard-*.json"),
    ),
    (
        "randomized look seed 1",
        [f"{STUDY}/experiments/e3s1/mujoco.json"],
        g(f"{STUDY}/experiments/e3s1/isaac/merged.json"),
        g(f"{STUDY}/isaac_n1/randlook_seed1/s*/shard-*.json"),
    ),
    (
        "randomized look seed 2",
        [f"{STUDY}/evals/act_visual_dr_seed2_mujoco.json"],
        [f"{STUDY}/isaac/act_visual_dr_seed2_isaac/merged.json"],
        g(f"{STUDY}/isaac_n1/randlook_seed2/s*/shard-*.json"),
    ),
]
print(
    "| row | MuJoCo | old Isaac | only MJ | only Isaac | p | NEW Isaac | only MJ | only Isaac | p |"
)
for lab, mj, old, new in ROWS:
    m = load(mj)
    cells = [lab, sum(m.values()) if len(m) == 100 else f"n={len(m)}"]
    for files in (old, new):
        if not files:
            cells += ["-"] * 4 if files is old else ["-"] * 4
            continue
        i = load(files)
        if len(i) != 100 or len(m) != 100:
            cells += [f"n={len(i)}", "", "", ""]
            continue
        b = sum(m[s] and not i[s] for s in m)
        c = sum(i[s] and not m[s] for s in m)
        cells += [sum(i.values()), b, c, f"{mcn(b, c):.3g}"]
    print("| " + " | ".join(str(c) for c in cells) + " |")
