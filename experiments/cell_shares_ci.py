"""Shares of the loss in the physics-by-renderer factorial with 95% bootstrap intervals (percentile; episodes resampled with replacement
from the pooled (model, seed) cases, the four cells of one case kept together; 20,000 draws, seed 0).
G = MM - II, L = MM - MI (pictures of Isaac Sim, physics of MuJoCo), P = MM - IM (physics of Isaac Sim, pictures of MuJoCo), in episodes.
SO-101 and SmolVLA: seeds 1000-1099; ViperX: 1000-1299. Gives the shares of the cells table for ACT and SmolVLA (no bar was written
for the SmolVLA cells)."""

import glob, json, os, random

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)


def ep(files, lo, hi):
    out = {}
    for f in files:
        for e in json.load(open(f))["results"]:
            if lo <= e["seed"] < hi:
                out[e["seed"]] = int(bool(e["success"]))
    assert len(out) == hi - lo, (files[:1], len(out))
    return out


sh = lambda d: sorted(glob.glob(os.path.join(d, "s*", "shard-*.json")))
GROUPS = {}
GROUPS["ACT SO-101 nominal (2)"] = [
    (
        f"{STUDY}/experiments/e5/seed{k}_mujoco.json",
        f"{STUDY}/isaac_nominal/act_nominal_seed{k}",
        f"{STUDY}/mi_cell/act_nominal_seed{k}",
        f"{STUDY}/im_cell/act_nominal_seed{k}",
        1100,
    )
    for k in (1, 2)
]
GROUPS["ACT ViperX nominal (2)"] = [
    (
        f"{STUDY}/viperx/nominal_seed{k}_mujoco.json",
        f"{STUDY}/isaac_viperx/nominal_seed{k}",
        f"{STUDY}/viperx_cells/mi/nominal_seed{k}",
        f"{STUDY}/viperx_cells/im/nominal_seed{k}",
        1300,
    )
    for k in (1, 2)
]
for look in ("nominal", "randlook"):
    GROUPS[f"SmolVLA {look} (3)"] = [
        (
            f"{STUDY}/smolvla/{look}_seed{k}/mujoco.json",
            f"{STUDY}/smolvla/{look}_seed{k}/isaac",
            f"{STUDY}/smolvla/{look}_seed{k}/mi",
            f"{STUDY}/smolvla/{look}_seed{k}/im",
            1100,
        )
        for k in (1, 2, 3)
    ]
rnd = random.Random(0)
for name, models in GROUPS.items():
    cases = []
    cnt = [0, 0, 0, 0]
    for mj, ii, mi, im, hi in models:
        a = ep([mj], 1000, hi)
        b = ep(sh(ii), 1000, hi)
        c = ep(sh(mi) if os.path.isdir(mi) else [mi], 1000, hi)
        d = ep(sh(im), 1000, hi)
        for s in a:
            cases.append((a[s], b[s], c[s], d[s]))
    n = len(cases)

    def stat(cs):
        MM, II, MI, IM = (sum(x[i] for x in cs) for i in range(4))
        G = MM - II
        return MM, II, MI, IM, G, MM - MI, MM - IM

    MM, II, MI, IM, G, L, P = stat(cases)
    rs = []
    for _ in range(20000):
        cs = [cases[rnd.randrange(n)] for _ in range(n)]
        m, i, mi_, im_, g, l, p = stat(cs)
        if g > 0:
            rs.append((l / g, p / g, (g - l - p) / g))
    q = lambda k, lo: sorted(r[k] for r in rs)[int(len(rs) * lo)]
    print(
        f"{name:26s} n={n} MM {MM} II {II} MI {MI} IM {IM} | G {G} L {L} P {P} resid {G - L - P} | L/G {L / G:.2f} [{q(0, 0.025):.2f}, {q(0, 0.975):.2f}] P/G {P / G:.2f} [{q(1, 0.025):.2f}, {q(1, 0.975):.2f}] resid/G {(G - L - P) / G:.2f} [{q(2, 0.025):.2f}, {q(2, 0.975):.2f}]"
    )
