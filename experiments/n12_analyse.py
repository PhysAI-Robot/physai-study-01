"""N12 pilot: does the sensitivity S of a model to edits on MuJoCo rank its MuJoCo-minus-Isaac Sim loss? (python research/experiments/n12_analyse.py)

S (primary) = mean fall in points of the success under blur, lighting 0.5, lighting 1.4 and a 10 mm camera shift (four edits both robots take), unedited minus edited on the
same 100 seeds. S7 = the same over the seven edits (SO-101 models only). Loss = MuJoCo (unedited, same seeds) minus Isaac Sim on seeds 1000-1099, in points.
Bars (study file): Spearman >= 0.6 with a one-sided permutation p < 0.05, and >= 0.5 still with the randomized-look group removed."""

import json, os
import numpy as np
from scipy.stats import spearmanr

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)
OUT = os.path.join(STUDY, "n12")
EDITS4 = ["blur", "light05", "light14", "cam10"]
EDITS7 = EDITS4 + ["arm_tint", "bg_dim", "plain"]


def score(name, edit):
    f = os.path.join(OUT, f"{name}.{edit}.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))["results"]
    return sum(bool(e["success"]) for e in r) * 100 / len(r)


def perm_p(x, y, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearmanr(x, y)[0]
    hits = sum(spearmanr(x, rng.permutation(y))[0] >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


rows = []
for m in json.load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "n12_models.json"))
):
    base = score(m["name"], "none")
    falls = (
        {
            e: None if score(m["name"], e) is None else base - score(m["name"], e)
            for e in EDITS7
        }
        if base is not None
        else {}
    )
    if base is None or any(falls.get(e) is None for e in EDITS4):
        print("missing:", m["name"])
        continue
    s7 = (
        np.mean([falls[e] for e in EDITS7])
        if all(falls.get(e) is not None for e in EDITS7)
        else None
    )
    rows.append(
        dict(
            m,
            mujoco=base,
            loss=base - m["isaac100"],
            s4=float(np.mean([falls[e] for e in EDITS4])),
            s7=s7,
            falls=falls,
        )
    )
print(f"{'model':36s} {'MuJoCo':>6s} {'Isaac':>6s} {'loss':>6s} {'S4':>6s} {'S7':>6s}")
for r in sorted(rows, key=lambda r: r["loss"]):
    print(
        f"{r['name']:36s} {r['mujoco']:6.0f} {r['isaac100']:6d} {r['loss']:6.1f} {r['s4']:6.1f} "
        + (f"{r['s7']:6.1f}" if r["s7"] is not None else "     -")
    )


def report(label, sel, key="s4"):
    x = [r[key] for r in sel]
    y = [r["loss"] for r in sel]
    rho = spearmanr(x, y)[0]
    print(
        f"{label}: n={len(sel)} Spearman {rho:.2f}, one-sided permutation p={perm_p(x, y):.4f}"
    )
    return rho


rho = report("all ACT models, S4", rows)
rho_wo = report(
    "without the randomized-look group (S4)",
    [r for r in rows if not r["randomized_look"]],
)
report(
    "SO-101 models, S7",
    [r for r in rows if r["robot"] == "so101" and r["s7"] is not None],
    "s7",
)
report("SO-101 models, S4", [r for r in rows if r["robot"] == "so101"])
print(
    f"bar: rho >= 0.6 with p < 0.05 and >= 0.5 without the randomized-look group -> {'GO' if rho >= 0.6 and rho_wo >= 0.5 else 'NO-GO'} (see p above)"
)
