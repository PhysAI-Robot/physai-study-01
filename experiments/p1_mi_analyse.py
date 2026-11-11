"""P1 second reading: MI-start (MuJoCo physics, Isaac Sim's pictures and Isaac Sim's cube start) against MuJoCo, N9's MI and Isaac Sim, SO-101, seeds 1000-1099.
L' = MuJoCo minus MI-start, pooled over the two nominal models; bars: L'/G >= 0.8 the pictures and the start together account for the gap; <= 0.6 the rest is physics or timing.
(python research/experiments/p1_mi_analyse.py)"""

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


MODELS = {  # name: (MuJoCo file, Isaac own, N9 cell dir, MI-start dir)
    "nominal_seed1": (
        "experiments/e5/seed1_mujoco.json",
        "isaac_n2/bridge_act_nominal_seed1",
        "n9/act_nominal_seed1",
    ),
    "nominal_seed2": (
        "experiments/e5/seed2_mujoco.json",
        "isaac_n2/bridge_act_nominal_seed2",
        "n9/act_nominal_seed2",
    ),
    "randlook_seed0": (
        "n9/randlook_seed0_mujoco_1000-1099.json",
        "isaac_n1/randlook_seed0",
        "n9/act_randomized_look_seed0",
    ),
}
T = dict(G=0, L=0, Lp=0, lost=0, gained=0, vs_n9=[0, 0])
for m, (mjf, isd, n9d) in MODELS.items():
    mj, isa = eps([os.path.join(STUDY, mjf)]), eps(sh(os.path.join(STUDY, isd)))
    n9, mi = (
        eps(sh(os.path.join(STUDY, n9d))),
        eps(sh(os.path.join(STUDY, "mi_start", "so101_" + m))),
    )
    assert len(mj) == len(isa) == len(n9) == len(mi) == 100, (
        m,
        len(mj),
        len(isa),
        len(n9),
        len(mi),
    )
    g, l, lp = (sum(mj.values()) - sum(x.values()) for x in (isa, n9, mi))
    lost, gained = (
        sum(mj[k] and not mi[k] for k in mj),
        sum(mi[k] and not mj[k] for k in mj),
    )
    a, b = sum(n9[k] and not mi[k] for k in mj), sum(mi[k] and not n9[k] for k in mj)
    print(
        f"so101_{m}: MuJoCo {sum(mj.values())}, N9 MI {sum(n9.values())}, MI-start {sum(mi.values())}, Isaac Sim {sum(isa.values())}; G {g}, L {l}, L' {lp}; "
        f"MI-start vs MuJoCo lost {lost} gained {gained} p={mcnemar(lost, gained):.4f}; MI-start vs N9 MI start-only-worse {a} better {b} p={mcnemar(a, b):.4f}"
    )
    if "randlook" not in m:
        T["G"] += g
        T["L"] += l
        T["Lp"] += lp
        T["lost"] += lost
        T["gained"] += gained
        T["vs_n9"][0] += a
        T["vs_n9"][1] += b
print(
    f"pooled nominal: G {T['G']}, L {T['L']} ({T['L'] / T['G']:.2f} G), L' {T['Lp']} ({T['Lp'] / T['G']:.2f} G); MI-start vs MuJoCo lost {T['lost']} gained {T['gained']} p={mcnemar(T['lost'], T['gained']):.4f}; "
    f"MI-start vs N9 MI: worse {T['vs_n9'][0]}, better {T['vs_n9'][1]} p={mcnemar(*T['vs_n9']):.4f}"
)
