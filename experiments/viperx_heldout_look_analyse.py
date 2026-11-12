"""The check on the second robot: the four ViperX models on the held-out look (Newton, seeds 1000-1299), against
their own standard Newton scores. Bar written beforehand: a model within 10 points of its standard Newton score means the look holds beyond its ranges.

    python research/experiments/viperx_heldout_look_analyse.py
"""

import json
import os
from math import comb

STUDY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "study01"
)


def outcomes(path):
    return {
        int(e["seed"]): bool(e["success"]) for e in json.load(open(path))["results"]
    }


def mcnemar(b, c):
    n = b + c
    return (
        1.0
        if n == 0
        else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    )


for name in ("randlook_seed1", "randlook_seed2", "nominal_seed1", "nominal_seed2"):
    standard = outcomes(os.path.join(STUDY, "viperx", f"{name}_newton.json"))
    held = os.path.join(STUDY, "viperx_heldout_look", f"{name}_heldout.json")
    if not os.path.exists(held):
        print(f"{name}: not scored yet")
        continue
    held = outcomes(held)
    k0, k1 = sum(standard.values()), sum(held.values())
    lost = sum(standard[s] and not held[s] for s in standard)
    gained = sum(held[s] and not standard[s] for s in standard)
    print(
        f"{name:15s} standard {k0}/300 ({k0 / 3:.1f}%), held-out look {k1}/300 ({k1 / 3:.1f}%), drop {(k0 - k1) / 3:+.1f} points, "
        f"lost {lost} gained {gained}, McNemar p {mcnemar(lost, gained):.4f}, within 10 points: {(k0 - k1) / 3 <= 10}"
    )
