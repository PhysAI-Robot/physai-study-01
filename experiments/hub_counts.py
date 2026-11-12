"""Recompute a paired row of the paper from the per-seed files on Hugging Face, without running a queue.

    python research/experiments/hub_counts.py act_nominal_unseeded/mujoco.json act_nominal_unseeded/isaac.json
    python research/experiments/hub_counts.py results/A/mujoco.json results/B/isaac.json      # local copies

Each argument is `<folder>/<file>` of `results/` in https://huggingface.co/rizgiak/physai-study-01-checkpoints (fetched), or a path to a file
`{"<seed>": true|false}`. Prints the successes of each file on the seeds both have, the seeds that pass in one file only, and the exact two-sided
McNemar p (the paired test of the tables). `results/INDEX.tsv` lists every file. Standard library only.
"""

import json
import sys
import urllib.request
from math import comb
from pathlib import Path

HUB = "https://huggingface.co/rizgiak/physai-study-01-checkpoints/resolve/main/results/"


def load(ref: str) -> dict[int, bool]:
    if Path(ref).is_file():
        text = Path(ref).read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(HUB + ref) as response:
            text = response.read().decode("utf-8")
    return {int(seed): bool(ok) for seed, ok in json.loads(text).items()}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    seeds = sorted(set(a) & set(b))
    only_a = sum(a[s] and not b[s] for s in seeds)
    only_b = sum(b[s] and not a[s] for s in seeds)
    print(
        f"{len(seeds)} common seeds ({seeds[0]}-{seeds[-1]}): "
        f"first {sum(a[s] for s in seeds)}, second {sum(b[s] for s in seeds)}; "
        f"pass only in the first {only_a}, only in the second {only_b}; exact McNemar p = {mcnemar(only_a, only_b):.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
