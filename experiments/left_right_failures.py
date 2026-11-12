"""Left and right (Section "failures"): failure rate with the cube at y >= 0.07 m against y < -0.07 m on Isaac Sim, the nine nominal-look runs
pooled (failure of a seed = the share of the nine runs that fail it), one-sided permutation test over seeds (higher at y >= 0.07).

    python research/experiments/left_right_failures.py

Seeds 1100-1299 are the pre-registered test; seeds 1000-1099 are the old ones (the observation was first made on them).
"""

import glob
import json
import os

import numpy as np

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)
rng = np.random.default_rng(2026)
# run -> (folders holding seeds 1000-1099, folder holding seeds 1100-1299)
RUNS = {
    "act_nominal_unseeded": (
        "isaac_nominal/act_nominal_unseeded",
        "isaac_left_right/act_nominal_unseeded",
    ),
    "act_nominal_seed1": (
        "isaac_nominal/act_nominal_seed1",
        "isaac_left_right/act_nominal_seed1",
    ),
    "act_nominal_seed2": (
        "isaac_nominal/act_nominal_seed2",
        "isaac_left_right/act_nominal_seed2",
    ),
    "act_nominal_seed3": (
        "isaac_nominal/act_nominal_seed3",
        "isaac_left_right/act_nominal_seed3",
    ),
    "nominal200_seed4": (
        "isaac_nominal/nominal200_seed4",
        "isaac_left_right/nominal200_seed4",
    ),
    "nominal200_seed5": (
        "isaac_nominal/nominal200_seed5",
        "isaac_left_right/nominal200_seed5",
    ),
    "nominal400_seed1": (
        "isaac_randomized_look/nominal400_seed1",
        "isaac_left_right/nominal400_seed1",
    ),
    "nominal400_seed2": (
        "isaac_randomized_look/nominal400_seed2",
        "isaac_left_right/nominal400_seed2",
    ),
    "nominal400_seed3": (
        "isaac_randomized_look/nominal400_seed3",
        "isaac_left_right/nominal400_seed3",
    ),
}


def episodes(folder: str) -> dict[int, dict]:
    out = {}
    for f in sorted(
        glob.glob(os.path.join(STUDY, folder, "s*", "shard-*.json"))
        + glob.glob(os.path.join(STUDY, folder, "s*", "merged.json"))
    ):
        for e in json.load(open(f))["results"]:
            out[int(e["seed"])] = e
    return out


data = {}
for name, folders in RUNS.items():
    data[name] = {}
    for folder in folders:
        data[name].update(episodes(folder))
    print(
        f"{name:22s} seeds {min(data[name])}-{max(data[name])}, {len(data[name])} episodes"
    )


def test(lo: int, hi: int, reps: int = 100000):
    seeds = [s for s in sorted(data[next(iter(data))]) if lo <= s < hi]
    assert all(s in d for d in data.values() for s in seeds), "a run lacks seeds"
    y = {s: data[next(iter(data))][s]["cube_start"][1] for s in seeds}
    fail = {s: np.mean([1 - data[m][s]["success"] for m in data]) for s in seeds}
    left = [s for s in seeds if y[s] < -0.07]
    right = [s for s in seeds if y[s] >= 0.07]
    fl, fr = np.mean([fail[s] for s in left]), np.mean([fail[s] for s in right])
    pool = np.array([fail[s] for s in left + right])
    k = len(right)
    hit = sum(
        pool[(p := rng.permutation(len(pool)))[:k]].mean() - pool[p[k:]].mean()
        >= fr - fl
        for _ in range(reps)
    )
    print(
        f"seeds {lo}-{hi - 1}: y<-0.07 {100 * fl:.1f}% ({len(left)} seeds), y>=0.07 {100 * fr:.1f}% ({len(right)} seeds), "
        f"difference {100 * (fr - fl):+.1f} points, one-sided p {(hit + 1) / (reps + 1):.5f}"
    )


test(1100, 1300)
test(1000, 1100)
