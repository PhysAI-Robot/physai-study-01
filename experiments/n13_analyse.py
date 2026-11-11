"""N13: MuJoCo minus IM-cell difference (P) against MuJoCo minus Isaac Sim (G), pooled over the two nominal models, bars of the study file
(python research/experiments/n13_analyse.py)."""

import glob, json, os
from math import comb

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)


def episodes(files):
    out = {}
    for f in files:
        for e in json.load(open(f))["results"]:
            out[e["seed"]] = bool(e["success"])
    return {k: v for k, v in out.items() if 1000 <= k < 1100}


def shards(d):
    return glob.glob(os.path.join(d, "s*", "shard-*.json"))


def mcnemar(b, c):  # exact, two-sided
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


MODELS = {  # name: (MuJoCo per-seed file, Isaac Sim own run)
    "act_nominal_seed1": (
        "experiments/e5/seed1_mujoco.json",
        "isaac_n2/bridge_act_nominal_seed1",
    ),
    "act_nominal_seed2": (
        "experiments/e5/seed2_mujoco.json",
        "isaac_n2/bridge_act_nominal_seed2",
    ),
}
G = L = Lmi = 0
for name, (mj_file, isaac_dir) in MODELS.items():
    mj = episodes([os.path.join(STUDY, mj_file)])
    isa = episodes(shards(os.path.join(STUDY, isaac_dir)))
    cell = episodes(shards(os.path.join(STUDY, "n13", name)))
    mi = episodes(shards(os.path.join(STUDY, "n9", name)))
    if len(cell) < 100:
        print(f"{name}: cell not finished ({len(cell)} seeds)")
        continue
    g, l = sum(mj.values()) - sum(isa.values()), sum(mj.values()) - sum(cell.values())
    G, L = G + g, L + l
    Lmi += sum(mj.values()) - sum(mi.values())
    lost = sum(mj[k] and not cell[k] for k in mj)
    gained = sum(cell[k] and not mj[k] for k in mj)
    ci = sum(cell[k] and not isa[k] for k in mj)
    ic = sum(isa[k] and not cell[k] for k in mj)
    print(
        f"{name}: MuJoCo {sum(mj.values())}, Isaac Sim {sum(isa.values())}, IM cell {sum(cell.values())}; G {g}, P {l}, P/G {l / g:.2f}; "
        f"cell vs MuJoCo lost {lost} gained {gained} McNemar p={mcnemar(lost, gained):.4f}; cell vs Isaac Sim cell-only {ci} Isaac-only {ic} p={mcnemar(ci, ic):.4f}"
    )
if G:
    print(
        f"pooled: G {G}, P {L}, P/G {L / G:.2f}, (MI cell of N9: L {Lmi}, L/G {Lmi / G:.2f}) (bars: P >= 0.25 G Isaac Sim's side carries a part alone; P <= 0.1 G nothing alone; else partial)"
    )
# consistency check, no bar: the randomized-look model (training seed 0)
mj = episodes([os.path.join(STUDY, "n9/randlook_seed0_mujoco_1000-1099.json")])
isa = episodes(shards(os.path.join(STUDY, "isaac_n1/randlook_seed0")))
cell = episodes(shards(os.path.join(STUDY, "n13/act_randomized_look_seed0")))
lost = sum(mj[k] and not cell[k] for k in mj)
gained = sum(cell[k] and not mj[k] for k in mj)
ci = sum(cell[k] and not isa[k] for k in mj)
ic = sum(isa[k] and not cell[k] for k in mj)
print(
    f"act_randomized_look_seed0 (no bar): MuJoCo {sum(mj.values())}, Isaac Sim {sum(isa.values())}, cell {sum(cell.values())}; cell vs MuJoCo lost {lost} gained {gained} p={mcnemar(lost, gained):.3f}; cell vs Isaac Sim cell-only {ci} Isaac-only {ic} p={mcnemar(ci, ic):.4f}"
)
