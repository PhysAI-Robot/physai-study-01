"""Q1: new Isaac Sim 6.2.0.0 score (q1/<group>/<name>/s*/shard-*.json) next to the old 6.1.0.0 per-seed file; Wilson 95%, per-seed old-only / new-only, exact McNemar (python research/experiments/q1_analyse.py [group ...])."""

import glob, json, os, sys
from math import comb, sqrt

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)
# group, name (Hugging Face folder), old per-seed file(s) on 6.1.0.0, seeds
ITEMS = [
    (
        "A_visual_servo",
        "visual_servo",
        ["evals/visual_servo_isaac_1000-1299.json"],
        (1000, 1300),
    ),
    (
        "B_chunk",
        "act_chunk30_100demos_unseeded",
        ["isaac/act_100_isaac_1000-1099/merged.json"],
        (1000, 1100),
    ),
    (
        "B_chunk",
        "act_chunk100_100demos_unseeded",
        ["isaac/act_100_c100_60k_isaac_1000-1099/merged.json"],
        (1000, 1100),
    ),
    (
        "C_isaac_only",
        "act_isaac_only_100_unseeded",
        ["isaac/act_isaac_100_isaac_1000-1099/merged.json"],
        (1000, 1100),
    ),
    (
        "C_isaac_only",
        "act_isaac_only_200_unseeded",
        ["isaac/act_isaac_200_isaac_1000-1099/merged.json"],
        (1000, 1100),
    ),
    (
        "C_isaac_only",
        "act_isaac_only_200_seed1",
        ["evals/act_isaac_200_seed1-eval/isaac/merged.json"],
        (1000, 1100),
    ),
    (
        "C_isaac_only",
        "act_isaac_only_400_unseeded",
        ["isaac/act_isaac_400_isaac_1000-1099/merged.json"],
        (1000, 1100),
    ),
    (
        "D_both_renderers",
        "act_both_renderers_unseeded",
        ["evals/act_mixed_400_isaac_1000-1099_merged.json"],
        (1000, 1100),
    ),
    (
        "D_both_renderers",
        "act_both_renderers_seed1",
        ["evals/act_mixed_400_seed1-eval/isaac/merged.json"],
        (1000, 1100),
    ),
    ("E_replay", "expert_replay", ["isaac/replay_isaac_0-49/merged.json"], (0, 50)),
    (
        "E_replay",
        "act_nominal_unseeded_replay",
        ["isaac/l5_act_replay_isaac/merged.json"],
        (900, 1000),
    ),
    (
        "F_validation",
        "act_nominal_unseeded",
        ["isaac/val_base_isaac/merged.json"],
        (900, 1000),
    ),
    (
        "F_validation",
        "act_200_c100_60k_jitter",
        ["isaac/val_jitter_isaac/merged.json"],
        (900, 1000),
    ),
    (
        "G_chunk",
        "act_nominal_unseeded_n50",
        ["isaac/final_n50_isaac/merged.json"],
        (1000, 1100),
    ),
    (
        "G_chunk",
        "act_nominal_unseeded_n25",
        ["isaac/final_n25_isaac/merged.json"],
        (1000, 1100),
    ),
    (
        "H_diffusion",
        "diffusion_policy_v2_randomized_look_seed0",
        ["isaac/dp2_visual_dr_isaac/merged.json"],
        (1000, 1100),
    ),
]


def load(files, lo, hi):
    out = {}
    for f in files:
        for e in json.load(open(f))["results"]:
            if lo <= e["seed"] < hi:
                out[e["seed"]] = bool(e["success"])
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(b, c):
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


want = set(sys.argv[1:])
print(
    "| Model | Seeds | New (6.2.0.0) | Wilson 95% | Old (6.1.0.0) | Wilson 95% | Old only | New only | McNemar p |"
)
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
tot = [0, 0, 0, 0]
for g, nm, old, (lo, hi) in ITEMS:
    if want and g not in want:
        continue
    new = load(
        glob.glob(os.path.join(STUDY, "q1", g, nm, "s*", "shard-*.json")), lo, hi
    )
    o = load([os.path.join(STUDY, f) for f in old], lo, hi)
    if len(new) != hi - lo or len(o) != hi - lo:
        print(f"| {nm} | {lo}-{hi - 1} | incomplete: new {len(new)}, old {len(o)} |")
        continue
    kn, ko = sum(new.values()), sum(o.values())
    b = sum(o[s] and not new[s] for s in o)
    c = sum(new[s] and not o[s] for s in o)
    wn, wo = wilson(kn, hi - lo), wilson(ko, hi - lo)
    print(
        f"| `{nm}` | {lo}-{hi - 1} | {kn}/{hi - lo} | {wn[0]:.0f}-{wn[1]:.0f}% | {ko}/{hi - lo} | {wo[0]:.0f}-{wo[1]:.0f}% | {b} | {c} | {mcnemar(b, c):.3f} |"
    )
    tot = [tot[0] + kn, tot[1] + ko, tot[2] + b, tot[3] + c]
print(
    f"\nsum over the rows: new {tot[0]}, old {tot[1]}, old-only {tot[2]}, new-only {tot[3]}"
)
