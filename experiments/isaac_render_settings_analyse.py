"""The eight rendering settings: recovery of the MuJoCo-minus-Isaac difference and the paired exact McNemar of each Isaac setting against the unchanged
Isaac scores (python research/experiments/isaac_render_settings_analyse.py)."""

import glob, json, os
from math import comb

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)
MUJOCO = {1: "experiments/e5/seed1_mujoco.json", 2: "experiments/e5/seed2_mujoco.json"}
SETTINGS = "floor_raw cumulative hard_shadows gi_off ao_off reflections_off spot_light sky_dome".split()


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


for n in (1, 2):
    mj = episodes([os.path.join(STUDY, MUJOCO[n])])
    base = episodes(shards(os.path.join(STUDY, f"isaac_nominal/act_nominal_seed{n}")))
    gap = sum(mj.values()) - sum(base.values())
    print(
        f"act_nominal_seed{n}: MuJoCo {sum(mj.values())}, Isaac as it is {sum(base.values())}, gap {gap} (half = {gap / 2:g})"
    )
    for s in SETTINGS:
        run = episodes(
            shards(
                os.path.join(STUDY, f"isaac_render_settings/{s}/act_nominal_seed{n}")
            )
        )
        if len(run) < 100:
            print(f"  {s:16s} {sum(run.values()):3d} of {len(run)} (not finished)")
            continue
        gained = sum(run[k] and not base[k] for k in base)
        lost = sum(base[k] and not run[k] for k in base)
        got = sum(run.values()) - sum(base.values())
        print(
            f"  {s:16s} {sum(run.values()):3d}  change {got:+3d}  recovers {got / gap:6.1%}  gained {gained} lost {lost}  McNemar p={mcnemar(gained, lost):.3f}"
        )
