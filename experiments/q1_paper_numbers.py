"""Numbers of the paper that depend on the Q1 re-scores and are not in the FINDINGS tables: failure locations of the visual servo and of
ACT, the y-band table, the servo-ACT overlap, the actions-per-chunk pairs, the replay-control 4 cm counts. Old (6.1.0.0) values are
computed beside the new ones only to check the script against the V2 paper; the paper takes the new column. The near-base and y-band
classification uses the cube start drawn by the sampler (MuJoCo's record of the same seed), so that it does not move with Isaac Sim's
3 mm settling."""

import glob, json, math, os, random, collections, statistics as st
from math import comb

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)


def load(files, lo=1000, hi=1300):
    out = {}
    for f in files:
        for e in json.load(open(f))["results"]:
            if lo <= e["seed"] < hi:
                out[e["seed"]] = e
    return out


g = lambda p: sorted(glob.glob(p))


def mcn(b, c):
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


mj_servo = load([f"{STUDY}/evals/visual_servo_mujoco_1000-1299.json"])
start = {
    s: e["cube_start"] for s, e in mj_servo.items()
}  # sampler start, same for every engine
rad = lambda s: math.hypot(*start[s])
ok = lambda d: {s: bool(e["success"]) for s, e in d.items()}

print("== A. visual servo, Isaac Sim 1000-1299")
for nm, files in (
    ("old", [f"{STUDY}/evals/visual_servo_isaac_1000-1299.json"]),
    ("new", g(f"{STUDY}/q1/A_visual_servo/visual_servo/s*/shard-*.json")),
):
    d = load(files)
    f = [s for s, e in d.items() if not e["success"]]
    near = [s for s in d if rad(s) < 0.19]
    nf = [s for s in f if rad(s) < 0.19]
    print(
        nm,
        "succ",
        sum(e["success"] for e in d.values()),
        "fail",
        len(f),
        "| near-base",
        len(near),
        "near-fail",
        len(nf),
        "= %.0f%%" % (100 * len(nf) / len(near)),
        "| other",
        len(d) - len(near),
        "other-fail",
        len(f) - len(nf),
        "| reasons",
        dict(collections.Counter(d[s]["failure_reason"] for s in f)),
        "| phases",
        dict(collections.Counter(d[s]["phase"] for s in f)),
        "| first100",
        sum(d[s]["success"] for s in range(1000, 1100)),
    )
print(
    "MuJoCo servo failures",
    sum(not e["success"] for e in mj_servo.values()),
    "of 300, all near-base:",
    all(rad(s) < 0.19 for s, e in mj_servo.items() if not e["success"]),
    "x of those:",
    sorted(round(start[s][0], 3) for s, e in mj_servo.items() if not e["success"]),
)

act_mj = load([f"{STUDY}/evals/act_200_c100_60k_mujoco_1000-1299.json"])
act_old = load(
    [f"{STUDY}/isaac/act_200_c100_60k_isaac_1000-1099/merged.json"], 1000, 1100
)
act_new = load(
    g(f"{STUDY}/isaac_n2/bridge_act_nominal_unseeded/s*/shard-*.json"), 1000, 1100
)
print("\n== B. ACT run 1 (act_nominal_unseeded), Isaac Sim 1000-1099")
for nm, d in (("old", act_old), ("new", act_new)):
    f = [s for s, e in d.items() if not e["success"]]
    near = [s for s in d if rad(s) < 0.19]
    nf = [s for s in f if rad(s) < 0.19]
    dist = [d[s]["dist_cube_target"] for s in f]
    print(
        nm,
        "succ",
        100 - len(f),
        "fail",
        len(f),
        "| near",
        len(near),
        "near-fail",
        len(nf),
        "| far",
        100 - len(near),
        "far-fail",
        len(f) - len(nf),
        "| timeouts",
        sum(d[s]["timeout"] for s in f),
        "| end>=10cm",
        sum(x >= 0.10 for x in dist),
        "median %.3f" % st.median(dist),
        "min %.3f" % min(dist),
    )
print(
    "MuJoCo run 1 failures (1000-1299):",
    sum(not e["success"] for e in act_mj.values()),
    "timeouts",
    sum(e["timeout"] for e in act_mj.values() if not e["success"]),
)
s100 = {
    s: v
    for s, v in ok(
        load(g(f"{STUDY}/q1/A_visual_servo/visual_servo/s*/shard-*.json"), 1000, 1100)
    ).items()
}
a_new = ok(act_new)
a_old = ok(act_old)
so = ok(load([f"{STUDY}/evals/visual_servo_isaac_1000-1299.json"], 1000, 1100))
print(
    "servo vs ACT run1, first 100 seeds: old (servo 6.1, ACT 6.1): servo-only",
    sum(so[s] and not a_old[s] for s in so),
    "ACT-only",
    sum(a_old[s] and not so[s] for s in so),
    "| new: servo-only",
    sum(s100[s] and not a_new[s] for s in s100),
    "ACT-only",
    sum(a_new[s] and not s100[s] for s in s100),
    "servo",
    sum(s100.values()),
    "ACT",
    sum(a_new.values()),
)

print(
    "\n== C. actions per chunk (act_nominal_unseeded), seeds 1000-1099, p against 100 actions"
)
mj100 = ok(load([f"{STUDY}/evals/act_200_c100_60k_mujoco_1000-1299.json"], 1000, 1100))
for n, mjf, old_is, new_is in (
    (
        "50",
        "final_n50_mujoco.json",
        f"{STUDY}/isaac/final_n50_isaac/merged.json",
        g(f"{STUDY}/q1/G_chunk/act_nominal_unseeded_n50/s*/shard-*.json"),
    ),
    (
        "25",
        "final_n25_mujoco.json",
        f"{STUDY}/isaac/final_n25_isaac/merged.json",
        g(f"{STUDY}/q1/G_chunk/act_nominal_unseeded_n25/s*/shard-*.json"),
    ),
):
    m = ok(load([f"{STUDY}/evals/{mjf}"], 1000, 1100))
    for nm, base, f in (("old", a_old, [old_is]), ("new", a_new, new_is)):
        i = ok(load(f, 1000, 1100))
        pm = mcn(
            sum(mj100[s] and not m[s] for s in m), sum(m[s] and not mj100[s] for s in m)
        )
        pi = mcn(
            sum(base[s] and not i[s] for s in i), sum(i[s] and not base[s] for s in i)
        )
        print(
            f"{n} actions {nm}: MuJoCo {sum(m.values())} (100 actions {sum(mj100.values())}) p={pm:.3g} | Isaac {sum(i.values())} (100 actions {sum(base.values())}) p={pi:.3g} | gap {sum(m.values()) - sum(i.values())} vs {sum(mj100.values()) - sum(base.values())}"
        )

print("\n== D. replay controls: success and 'within 4 cm' on Isaac Sim")
for nm, old, new, lo, hi in (
    (
        "expert replay 0-49",
        f"{STUDY}/isaac/replay_isaac_0-49/merged.json",
        g(f"{STUDY}/q1/E_replay/expert_replay/s*/shard-*.json"),
        0,
        50,
    ),
    (
        "ACT MuJoCo actions 900-999",
        f"{STUDY}/isaac/l5_act_replay_isaac/merged.json",
        g(f"{STUDY}/q1/E_replay/act_nominal_unseeded_replay/s*/shard-*.json"),
        900,
        1000,
    ),
):
    for lab, f in (("old", [old]), ("new", new)):
        d = load(f, lo, hi)
        print(
            nm,
            lab,
            "success",
            sum(e["success"] for e in d.values()),
            "within 4 cm",
            sum(e["dist_cube_target"] <= 0.04 for e in d.values()),
            "n",
            len(d),
        )

print(
    "\n== E. y bands of ACT failures on Isaac Sim, seeds 1000-1099 (sampler y), pooled"
)


def ybands(runs, label):
    bands = [(-9, -0.07), (-0.07, 0), (0, 0.07), (0.07, 9)]
    fail = [0] * 4
    n = [0] * 4
    for d in runs:
        for s, e in d.items():
            y = start[s][1]
            b = next(i for i, (lo, hi) in enumerate(bands) if lo <= y < hi)
            n[b] += 1
            fail[b] += not e["success"]
    cnt = {s: sum(not d[s]["success"] for d in runs) for s in runs[0]}
    ys = [start[s][1] for s in cnt]
    v = [cnt[s] for s in cnt]

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        sa = math.sqrt(sum((x - ma) ** 2 for x in a))
        sb = math.sqrt(sum((x - mb) ** 2 for x in b))
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)

    r = pear(ys, v)
    rnd = random.Random(0)
    big = 0
    N = 100000
    vv = v[:]
    for _ in range(N):
        rnd.shuffle(vv)
        big += abs(pear(ys, vv)) >= abs(r) - 1e-12
    print(
        label,
        " | ".join(f"{f}/{m} ({100 * f / m:.0f}%)" for f, m in zip(fail, n)),
        f"| r={r:.3f} p={(big + 1) / (N + 1):.4f}",
    )


nom_new = [
    load(g(f"{STUDY}/isaac_n2/bridge_act_nominal_{k}/s*/shard-*.json"), 1000, 1100)
    for k in ("unseeded", "seed1", "seed2", "seed3")
]
rl_new = [
    load(g(f"{STUDY}/isaac_n1/randlook_seed{k}/s*/shard-*.json"), 1000, 1100)
    for k in (0, 1, 2)
]
ybands(nom_new, "nominal 4 runs, new")
ybands(rl_new, "randomized 3 runs, new")
print(
    "randomized look seed 0 (new): fail",
    sum(not e["success"] for e in rl_new[0].values()),
    "near-base fails",
    sum((not e["success"]) and rad(s) < 0.19 for s, e in rl_new[0].items()),
    "of near",
    sum(rad(s) < 0.19 for s in rl_new[0]),
)
