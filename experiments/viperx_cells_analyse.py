"""The ViperX repeat of the MI and IM cells, seeds 1000-1299, pooled over the two nominal models (python research/experiments/viperx_cells_analyse.py)."""

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
    return {k: v for k, v in out.items() if 1000 <= k < 1300}


def shards(d):
    return glob.glob(os.path.join(d, "s*", "shard-*.json"))


def mcnemar(b, c):  # exact, two-sided
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


tot = {"G": 0, "mi": 0, "im": 0}
have = {"mi": True, "im": True}
for m in ("nominal_seed1", "nominal_seed2"):
    mj = episodes([os.path.join(STUDY, f"viperx/{m}_mujoco.json")])
    isa = episodes(shards(os.path.join(STUDY, f"isaac_viperx/{m}")))
    g = sum(mj.values()) - sum(isa.values())
    tot["G"] += g
    line = f"{m}: MuJoCo {sum(mj.values())}, Isaac Sim {sum(isa.values())}, G {g}"
    for cell in ("mi", "im"):
        c = episodes(shards(os.path.join(STUDY, f"viperx_cells/{cell}/{m}")))
        if len(c) < 300:
            have[cell] = False
            line += f"; {cell.upper()} not finished ({len(c)} seeds)"
            continue
        d = sum(mj.values()) - sum(c.values())
        tot[cell] += d
        lost = sum(mj[k] and not c[k] for k in mj)
        gained = sum(c[k] and not mj[k] for k in mj)
        line += f"; {cell.upper()} {sum(c.values())}, {'L' if cell == 'mi' else 'P'} {d} ({d / g:.2f} G), vs MuJoCo lost {lost} gained {gained} McNemar p={mcnemar(lost, gained):.4f}"
    print(line)
G = tot["G"]
if have["mi"]:
    print(
        f"pooled MI: G {G}, L {tot['mi']}, L/G {tot['mi'] / G:.2f} (bars: >= 0.8 pictures carry it; <= 0.5 interaction; else partial)"
    )
if have["im"]:
    print(
        f"pooled IM: G {G}, P {tot['im']}, P/G {tot['im'] / G:.2f} (bars: >= 0.25 alone; <= 0.1 nothing alone; else partial)"
    )
