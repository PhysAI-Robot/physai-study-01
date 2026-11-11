"""T1 Part 2 numbers (python research/experiments/t1p2_analyse.py): per checkpoint the MuJoCo, Isaac Sim, MI and IM scores with 95% Wilson intervals,
G = MJ100 - Isaac, L = MJ100 - MI, P = MJ100 - IM with the shares L/G, P/G and the rest with 95% bootstrap intervals (as shares_ci.py: percentile,
episodes resampled with replacement, the four cells of one episode kept together, 20,000 draws, seed 0), started counts and median first-move step,
and the same numbers of the three O1 runs of the same look. Numbers only; no reading."""

import glob, json, os, random, statistics as st
from math import sqrt

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)
P2 = os.path.join(STUDY, "t1", "p2")
CK = ["randlook_30k", "randlook_40k", "nominal_30k", "nominal_40k"]


def shards(d):
    r = {}
    for f in glob.glob(os.path.join(d, "s*", "shard-*.json")):
        for e in json.load(open(f))["results"]:
            r[e["seed"]] = e
    return r


def whole(p):
    return (
        {e["seed"]: e for e in json.load(open(p))["results"]}
        if os.path.exists(p)
        else {}
    )


def cells(base):
    return {
        "mujoco": whole(base + "/mujoco.json"),
        "isaac": shards(base + "/isaac"),
        "mi": shards(base + "/mi"),
        "im": shards(base + "/im"),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def succ(d, lo=0, hi=10**9):
    return sum(bool(e["success"]) for s, e in d.items() if lo <= s < hi)


def n_of(d, lo=0, hi=10**9):
    return sum(1 for s in d if lo <= s < hi)


def started(d):
    fm = [e["first_move_step"] for e in d.values() if e.get("first_move_step", -1) >= 0]
    return len(fm), len(d), (st.median(fm) if fm else None)


def shares(c):
    seeds = sorted(set(c["mujoco"]) & set(c["isaac"]) & set(c["mi"]) & set(c["im"]))
    seeds = [s for s in seeds if 1000 <= s < 1100]
    cases = [
        (
            int(bool(c["mujoco"][s]["success"])),
            int(bool(c["isaac"][s]["success"])),
            int(bool(c["mi"][s]["success"])),
            int(bool(c["im"][s]["success"])),
        )
        for s in seeds
    ]
    n = len(cases)

    def stat(cs):
        mm, ii, mi, im = (sum(x[i] for x in cs) for i in range(4))
        return mm - ii, mm - mi, mm - im

    G, L, Pp = stat(cases)
    rnd = random.Random(0)
    rs = []
    for _ in range(20000):
        g, l, p = stat([cases[rnd.randrange(n)] for _ in range(n)])
        if g > 0:
            rs.append((l / g, p / g, (g - l - p) / g))
    q = lambda k, f: sorted(r[k] for r in rs)[int(len(rs) * f)] if rs else float("nan")
    out = f"G {G} L {L} P {Pp} rest {G - L - Pp}"
    if G > 0:
        out += (
            f" | L/G {L / G:.2f} [{q(0, 0.025):.2f}, {q(0, 0.975):.2f}] P/G {Pp / G:.2f} [{q(1, 0.025):.2f}, {q(1, 0.975):.2f}]"
            f" rest/G {(G - L - Pp) / G:.2f} [{q(2, 0.025):.2f}, {q(2, 0.975):.2f}]"
        )
    return out, (G, L, Pp)


def report(name, base):
    c = cells(base)
    mj = c["mujoco"]
    if not mj:
        print(name, "no MuJoCo result yet")
        return
    row = [
        f"MuJoCo 300: {succ(mj)} {tuple(round(x, 1) for x in wilson(succ(mj), n_of(mj)))}",
        f"MJ100: {succ(mj, 1000, 1100)}",
    ]
    for k in ("isaac", "mi", "im"):
        d = c[k]
        if d:
            row.append(
                f"{k}: {succ(d)}/{len(d)} {tuple(round(x, 1) for x in wilson(succ(d), len(d)))}"
            )
    print(f"{name:18}", " | ".join(row))
    for k in ("mujoco", "isaac", "mi", "im"):
        if c[k]:
            s, n, med = started(c[k])
            print(f"{'':18}started {k}: {s} of {n}, median first-move step {med}")
    if all(c[k] for k in ("isaac", "mi", "im")):
        print(f"{'':18}", shares(c)[0])


print(
    "== T1 Part 2 (MuJoCo seeds 1000-1299; Isaac Sim, MI, IM seeds 1000-1099; 95% intervals in brackets, in %)"
)
for n in CK:
    report("T1P2 " + n, os.path.join(P2, n))
print("\n== the three O1 runs of the same look (20k steps)")
for look in ("randlook", "nominal"):
    for s in (1, 2, 3):
        report(f"O1 {look}_seed{s}", os.path.join(STUDY, "o1", f"{look}_seed{s}"))
print(
    "\n== validation (MuJoCo 900-999, from T1a / T1b): see FINDINGS 'T1 (study-01-v3)'"
)
print("\n== scoring time per checkpoint (from logs/t1p2_queue.log)")
for line in open(os.path.join(STUDY, "logs", "t1p2_queue.log")):
    if " start (" in line or line.split(" ")[1:2] == ["done"]:
        print(line.rstrip())
