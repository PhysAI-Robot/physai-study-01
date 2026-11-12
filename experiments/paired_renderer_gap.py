"""Paired MuJoCo-vs-Isaac Sim table of the paper (the renderer gap on the same 100 test seeds, 1000-1099): per row the two scores, the seeds that pass on one engine only, and the exact McNemar p.
Reads the per-episode result files that the experiments write under outputs/study01/."""

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


ROWS = [  # label, mujoco files, isaac files
    (
        "chunk30 100 demos",
        [f"{STUDY}/evals/act_100_mujoco_1000-1299.json"],
        g(
            f"{STUDY}/isaac_controls/B_chunk/act_chunk30_100demos_unseeded/s*/shard-*.json"
        ),
    ),
    (
        "chunk100 100 demos 60k",
        [f"{STUDY}/evals/act_100_c100_60k_mujoco_1000-1299.json"],
        g(
            f"{STUDY}/isaac_controls/B_chunk/act_chunk100_100demos_unseeded/s*/shard-*.json"
        ),
    ),
    (
        "final ACT run 1",
        [f"{STUDY}/evals/act_200_c100_60k_mujoco_1000-1299.json"],
        g(f"{STUDY}/isaac_nominal/act_nominal_unseeded/s*/shard-*.json"),
    ),
    (
        "final ACT seed 1",
        [f"{STUDY}/experiments/e5/seed1_mujoco.json"],
        g(f"{STUDY}/isaac_nominal/act_nominal_seed1/s*/shard-*.json"),
    ),
    (
        "final ACT seed 2",
        [f"{STUDY}/experiments/e5/seed2_mujoco.json"],
        g(f"{STUDY}/isaac_nominal/act_nominal_seed2/s*/shard-*.json"),
    ),
    (
        "final ACT seed 3",
        [f"{STUDY}/evals/act_200_seed3_mujoco.json"],
        g(f"{STUDY}/isaac_nominal/act_nominal_seed3/s*/shard-*.json"),
    ),
    (
        "Isaac replays 100",
        [f"{STUDY}/evals/act_isaac_100_mujoco_1000-1299.json"],
        g(
            f"{STUDY}/isaac_controls/C_isaac_only/act_isaac_only_100_unseeded/s*/shard-*.json"
        ),
    ),
    (
        "Isaac replays 200",
        [f"{STUDY}/evals/act_isaac_200_mujoco_1000-1299.json"],
        g(
            f"{STUDY}/isaac_controls/C_isaac_only/act_isaac_only_200_unseeded/s*/shard-*.json"
        ),
    ),
    (
        "Isaac replays 200 seed 1",
        [f"{STUDY}/evals/act_isaac_200_seed1-eval/mujoco.json"],
        g(
            f"{STUDY}/isaac_controls/C_isaac_only/act_isaac_only_200_seed1/s*/shard-*.json"
        ),
    ),
    (
        "Isaac replays 400",
        [f"{STUDY}/evals/act_isaac_400_mujoco_1000-1299.json"],
        g(
            f"{STUDY}/isaac_controls/C_isaac_only/act_isaac_only_400_unseeded/s*/shard-*.json"
        ),
    ),
    (
        "both renderers run 1",
        [f"{STUDY}/evals/act_mixed_400_mujoco_1000-1299.json"],
        g(
            f"{STUDY}/isaac_controls/D_both_renderers/act_both_renderers_unseeded/s*/shard-*.json"
        ),
    ),
    (
        "both renderers seed 1",
        [f"{STUDY}/evals/act_mixed_400_seed1-eval/mujoco.json"],
        g(
            f"{STUDY}/isaac_controls/D_both_renderers/act_both_renderers_seed1/s*/shard-*.json"
        ),
    ),
    (
        "randomized look seed 0",
        [f"{STUDY}/experiments/e3/mujoco_1000-1299.json"],
        g(f"{STUDY}/isaac_randomized_look/randlook_seed0/s*/shard-*.json"),
    ),
    (
        "randomized look seed 1",
        [f"{STUDY}/experiments/e3s1/mujoco.json"],
        g(f"{STUDY}/isaac_randomized_look/randlook_seed1/s*/shard-*.json"),
    ),
    (
        "randomized look seed 2",
        [f"{STUDY}/evals/act_visual_dr_seed2_mujoco.json"],
        g(f"{STUDY}/isaac_randomized_look/randlook_seed2/s*/shard-*.json"),
    ),
]
print("| row | MuJoCo | Isaac | only MJ | only Isaac | p |")
for lab, mj, isaac in ROWS:
    m = load(mj)
    cells = [lab, sum(m.values()) if len(m) == 100 else f"n={len(m)}"]
    i = load(isaac)
    if len(i) != 100 or len(m) != 100:
        cells += [f"n={len(i)}", "", "", ""]
    else:
        b = sum(m[s] and not i[s] for s in m)
        c = sum(i[s] and not m[s] for s in m)
        cells += [sum(i.values()), b, c, f"{mcn(b, c):.3g}"]
    print("| " + " | ".join(str(c) for c in cells) + " |")
