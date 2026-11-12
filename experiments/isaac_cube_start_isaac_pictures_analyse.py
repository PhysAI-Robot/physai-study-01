"""Isaac Sim's cube start, second reading: the MI-start cell (MuJoCo physics, Isaac Sim's pictures and Isaac Sim's cube start) against MuJoCo, the MI cell and Isaac Sim, SO-101, seeds 1000-1099.
L' = MuJoCo minus MI-start, pooled over the two nominal models; bars: L'/G >= 0.8 the pictures and the start together account for the gap; <= 0.6 the rest is physics or timing.
(python research/experiments/isaac_cube_start_isaac_pictures_analyse.py)"""

import glob, json, os
from math import comb

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)


def eps(files):
    out = {}
    for f in files:
        for e in json.load(open(f))["results"]:
            out[e["seed"]] = bool(e["success"])
    return {k: v for k, v in out.items() if 1000 <= k < 1100}


def sh(d):
    return glob.glob(os.path.join(d, "s*", "shard-*.json"))


def mcnemar(b, c):
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


MODELS = {  # name: (MuJoCo file, Isaac own, MI cell dir, MI-start dir)
    "nominal_seed1": (
        "experiments/e5/seed1_mujoco.json",
        "isaac_nominal/act_nominal_seed1",
        "mi_cell/act_nominal_seed1",
    ),
    "nominal_seed2": (
        "experiments/e5/seed2_mujoco.json",
        "isaac_nominal/act_nominal_seed2",
        "mi_cell/act_nominal_seed2",
    ),
    "randlook_seed0": (
        "mi_cell/randlook_seed0_mujoco_1000-1099.json",
        "isaac_randomized_look/randlook_seed0",
        "mi_cell/act_randomized_look_seed0",
    ),
}
T = dict(G=0, L=0, Lp=0, lost=0, gained=0, vs_mi_cell=[0, 0])
for m, (mjf, isd, mi_dir) in MODELS.items():
    mj, isa = eps([os.path.join(STUDY, mjf)]), eps(sh(os.path.join(STUDY, isd)))
    mi_cell, mi = (
        eps(sh(os.path.join(STUDY, mi_dir))),
        eps(sh(os.path.join(STUDY, "cube_start_mi", "so101_" + m))),
    )
    assert len(mj) == len(isa) == len(mi_cell) == len(mi) == 100, (
        m,
        len(mj),
        len(isa),
        len(mi_cell),
        len(mi),
    )
    g, l, lp = (sum(mj.values()) - sum(x.values()) for x in (isa, mi_cell, mi))
    lost, gained = (
        sum(mj[k] and not mi[k] for k in mj),
        sum(mi[k] and not mj[k] for k in mj),
    )
    a, b = (
        sum(mi_cell[k] and not mi[k] for k in mj),
        sum(mi[k] and not mi_cell[k] for k in mj),
    )
    print(
        f"so101_{m}: MuJoCo {sum(mj.values())}, MI {sum(mi_cell.values())}, MI-start {sum(mi.values())}, Isaac Sim {sum(isa.values())}; G {g}, L {l}, L' {lp}; "
        f"MI-start vs MuJoCo lost {lost} gained {gained} p={mcnemar(lost, gained):.4f}; MI-start vs MI cell start-only-worse {a} better {b} p={mcnemar(a, b):.4f}"
    )
    if "randlook" not in m:
        T["G"] += g
        T["L"] += l
        T["Lp"] += lp
        T["lost"] += lost
        T["gained"] += gained
        T["vs_mi_cell"][0] += a
        T["vs_mi_cell"][1] += b
print(
    f"pooled nominal: G {T['G']}, L {T['L']} ({T['L'] / T['G']:.2f} G), L' {T['Lp']} ({T['Lp'] / T['G']:.2f} G); MI-start vs MuJoCo lost {T['lost']} gained {T['gained']} p={mcnemar(T['lost'], T['gained']):.4f}; "
    f"MI-start vs MI cell: worse {T['vs_mi_cell'][0]}, better {T['vs_mi_cell'][1]} p={mcnemar(*T['vs_mi_cell']):.4f}"
)
