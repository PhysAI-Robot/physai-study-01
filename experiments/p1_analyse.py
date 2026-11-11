"""P1: MM-start (MuJoCo with Isaac Sim's cube start) against MuJoCo and Isaac Sim, pooled over the two nominal models of a robot (python research/experiments/p1_analyse.py).

S = MuJoCo minus MM-start; G = MuJoCo minus Isaac Sim (as in N9 / N14). Bars of the study file: the start carries a part of the loss if S >= 0.1 G and the pooled McNemar
test of MM-start against MuJoCo gives p < 0.05; it carries nothing if S <= 0.03 G; between, partial. The randomized-look model is shown apart (no bar), as in N9.
"""

import glob, json, os
from math import comb


STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)


def episodes(files, hi):
    out = {}
    for f in files:
        for e in json.load(open(f))["results"]:
            out[e["seed"]] = bool(e["success"])
    return {k: v for k, v in out.items() if 1000 <= k < hi}


def shards(d):
    return glob.glob(os.path.join(STUDY, d, "s*", "shard-*.json"))


def mcnemar(b, c):  # exact, two-sided
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


# name: (robot, MuJoCo file, Isaac Sim own run, seeds end)
MODELS = {
    "so101_nominal_seed1": (
        "so101",
        "experiments/e5/seed1_mujoco.json",
        "isaac_n2/bridge_act_nominal_seed1",
        1100,
    ),
    "so101_nominal_seed2": (
        "so101",
        "experiments/e5/seed2_mujoco.json",
        "isaac_n2/bridge_act_nominal_seed2",
        1100,
    ),
    "so101_randlook_seed0": (
        "so101",
        "n9/randlook_seed0_mujoco_1000-1099.json",
        "isaac_n1/randlook_seed0",
        1100,
    ),
    "viperx_nominal_seed1": (
        "viperx",
        "n4/nominal_seed1_mujoco.json",
        "isaac_n4/nominal_seed1",
        1300,
    ),
    "viperx_nominal_seed2": (
        "viperx",
        "n4/nominal_seed2_mujoco.json",
        "isaac_n4/nominal_seed2",
        1300,
    ),
}
pool = {}
for name, (robot, mj_file, isaac_dir, hi) in MODELS.items():
    f = os.path.join(STUDY, "mm_start", name + ".json")
    if not os.path.exists(f):
        print(f"{name}: MM-start not finished")
        continue
    mj = episodes([os.path.join(STUDY, mj_file)], hi)
    isa = episodes(shards(isaac_dir), hi)
    mm = episodes([f], hi)
    assert len(mj) == len(isa) == len(mm) == hi - 1000, (
        name,
        len(mj),
        len(isa),
        len(mm),
    )
    g, s = sum(mj.values()) - sum(isa.values()), sum(mj.values()) - sum(mm.values())
    lost = sum(mj[k] and not mm[k] for k in mj)
    gained = sum(mm[k] and not mj[k] for k in mj)
    print(
        f"{name}: MuJoCo {sum(mj.values())}, MM-start {sum(mm.values())}, Isaac Sim {sum(isa.values())}; G {g}, S {s}"
        + (f" ({s / g:.2f} G)" if g else "")
        + f"; MM-start vs MuJoCo lost {lost} gained {gained} McNemar p={mcnemar(lost, gained):.4f}"
    )
    if "randlook" not in name:
        p = pool.setdefault(robot, [0, 0, 0, 0])
        p[0] += g
        p[1] += s
        p[2] += lost
        p[3] += gained
for robot, (g, s, lost, gained) in pool.items():
    p = mcnemar(lost, gained)
    verdict = (
        "carries a part"
        if (s >= 0.1 * g and p < 0.05)
        else "nothing"
        if s <= 0.03 * g
        else "partial / not separable"
    )
    print(
        f"pooled {robot}: G {g}, S {s}, S/G {s / g:.2f}; lost {lost} gained {gained} McNemar p={p:.4f} -> {verdict}"
    )
