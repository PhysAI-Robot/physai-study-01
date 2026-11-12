"""The ladder of the six SmolVLA runs (SmolVLA table), against the bars written before the runs, with the shares of episodes in which the arm started (python research/experiments/smolvla_ladder_analyse.py)."""

import glob, json, os, statistics as st
from math import comb

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)
RUNS = [f"{k}_seed{s}" for k in ("nominal", "randlook") for s in (1, 2, 3)]


def shards(n, c):
    r = {}
    for f in glob.glob(os.path.join(STUDY, "smolvla", n, c, "s*", "shard-*.json")):
        for e in json.load(open(f))["results"]:
            r[e["seed"]] = e
    return r


def whole(n, c):
    p = os.path.join(STUDY, "smolvla", n, c + ".json")
    return (
        {e["seed"]: e for e in json.load(open(p))["results"]}
        if os.path.exists(p)
        else {}
    )


def cell(n, c):
    return whole(n, c) if c in ("mujoco", "newton") else shards(n, c)


def k100(d):
    return {s: e for s, e in d.items() if 1000 <= s < 1100}


def mcnemar(b, c):
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


def succ(d):
    return sum(bool(e["success"]) for e in d.values())


def started(d):
    fm = [e["first_move_step"] for e in d.values() if e.get("first_move_step", -1) >= 0]
    ok = [e for e in d.values() if e.get("first_move_step", -1) >= 0 and e["success"]]
    return len(fm), (st.median(fm) if fm else None), len(ok)


print(
    "== ladder (MuJoCo 300 / Newton 300 on 1000-1299; the rest 100 on 1000-1099; MJ100 = MuJoCo on the same 100 seeds)"
)
print(
    f"{'run':16}{'MJ300':>6}{'MJ100':>6}{'NW300':>6}{'Isaac':>6}{'MI':>5}{'IM':>5}  G=MJ100-Isaac  L(MI)  P(IM)"
)
loss = {}
for n in RUNS:
    mj = cell(n, "mujoco")
    mj100 = succ(k100(mj))
    row = {c: succ(cell(n, c)) for c in ("isaac", "mi", "im") if cell(n, c)}
    loss[n] = dict(
        G=mj100 - row["isaac"],
        L=mj100 - row["mi"],
        P=mj100 - row["im"],
        MJ100=mj100,
        **row,
    )
    print(
        f"{n:16}{succ(mj):6}{mj100:6}{succ(cell(n, 'newton')):6}{row['isaac']:6}{row['mi']:5}{row['im']:5}  {loss[n]['G']:>8} {loss[n]['L']:>9} {loss[n]['P']:>6}"
    )

print(
    "\n== started (arm moved 1 cm): count of 100 (300 for MuJoCo/Newton), median first-move step, successes among the started"
)
for n in RUNS:
    parts = []
    for c in ("mujoco", "newton", "isaac", "mi", "im"):
        d = cell(n, c)
        if d:
            s, med, ok = started(d)
            parts.append(f"{c} {s}/{len(d)} med {med} ok {ok}")
    print(f"{n:16}", "; ".join(parts))
print(
    "repeat cells of nominal_seed1:",
    {c: started(shards("nominal_seed1", c)) for c in ("isaac_r2", "im_r2")},
)

print("\n== bars")
nom = [loss[f"nominal_seed{s}"] for s in (1, 2, 3)]
rl = [loss[f"randlook_seed{s}"] for s in (1, 2, 3)]
print(
    "1. every nominal run loses >= 15 points on Isaac Sim:",
    [x["G"] for x in nom],
    all(x["G"] >= 15 for x in nom),
)
print(
    "2. every nominal run loses < 9 points (smallest ACT loss):",
    [x["G"] for x in nom],
    all(x["G"] < 9 for x in nom),
)
mn, mr = sum(x["L"] for x in nom) / 3, sum(x["L"] for x in rl) / 3
print(
    f"3. MI loss: nominal {[x['L'] for x in nom]} mean {mn:.1f}; randomized {[x['L'] for x in rl]} mean {mr:.1f}; ratio {mr / mn:.2f} (bar <= 0.33)"
)
print(
    "   Isaac loss of randomized runs:",
    [x["G"] for x in rl],
    "mean",
    sum(x["G"] for x in rl) / 3,
)
print(
    "   IM loss (P) nominal", [x["P"] for x in nom], "randomized", [x["P"] for x in rl]
)

print("\n== repeat of nominal_seed1 (first score against the repeat, per seed)")
for a, b in (("isaac", "isaac_r2"), ("im", "im_r2")):
    A, B = shards("nominal_seed1", a), shards("nominal_seed1", b)
    seeds = sorted(set(A) & set(B))
    lost = sum(A[s]["success"] and not B[s]["success"] for s in seeds)
    gained = sum(B[s]["success"] and not A[s]["success"] for s in seeds)
    print(
        f"{a} {succ(A)} -> {b} {succ(B)}: {len(seeds)} seeds, first-only {lost}, repeat-only {gained}, same outcome {len(seeds) - lost - gained}, McNemar p={mcnemar(lost, gained):.3f}"
    )

print("\n== MuJoCo vs Isaac per seed (McNemar, pooled over the three runs of a recipe)")
for kind in ("nominal", "randlook"):
    b = c = 0
    for s in (1, 2, 3):
        n = f"{kind}_seed{s}"
        mj, isa = k100(cell(n, "mujoco")), cell(n, "isaac")
        b += sum(mj[k]["success"] and not isa[k]["success"] for k in mj)
        c += sum(isa[k]["success"] and not mj[k]["success"] for k in mj)
    print(kind, "MuJoCo-only", b, "Isaac-only", c, f"p={mcnemar(b, c):.2g}")

print("\n== training")
for n in RUNS:
    log = json.load(open(os.path.join(STUDY, f"smolvla_{n}", "training_log.json")))
    print(
        n,
        "final loss",
        round(log[-1]["loss"], 4),
        "mean of last 10 logged",
        round(sum(x["loss"] for x in log[-10:]) / 10, 4),
        "time h",
        round(log[-1]["elapsed_s"] / 3600, 2),
    )
