"""Where does the cube start on Isaac Sim? Makes the file the `cube_start` option of the MuJoCo envs reads (study 01).

Two steps. Measure the same seeds twice in two different orders (Isaac Sim only, no policy):

    OMNI_KIT_ACCEPT_EULA=YES python research/scripts/cube_start_table.py measure \
        --manifest configs/manifests/so101_single_cube_randomized_place.yaml \
        --seeds 1000-1099 --order forward --out outputs/study01/v3/cube_so101_forward.json
    ... --order reverse --out outputs/study01/v3/cube_so101_reverse.json

then decide from the two files (no Isaac Sim needed):

    python research/scripts/cube_start_table.py decide \
        outputs/study01/v3/cube_so101_forward.json outputs/study01/v3/cube_so101_reverse.json \
        --out outputs/study01/v3/cube_start_so101.json

Why two orders: the shift of the cube after it is placed (about 3 mm for the SO-101, 8.6 mm for the
ViperX in an earlier probe) may depend on where the cube sat in the episode before. `decide` writes
  - offset_xy, if the shift agrees between the two orders and is the same for every seed,
  - xy_by_seed, if it agrees between the orders but differs from seed to seed,
  - nothing, and exits with 2, if it depends on the order. Then a table is only valid for the order
    of the scoring run; log the cube pose at the first step in the Isaac Sim scoring runs instead.
The thresholds are below and were written before any measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

AGREE_MM = 0.3  # the same seed's shift in the two orders must agree to this (per axis)
CONSTANT_MM = 0.5  # the shift counts as one offset if its spread over seeds is at most this (per axis)


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += range(int(lo), int(hi or lo) + 1)
    return seeds


def measure(args: argparse.Namespace) -> None:
    sys.path[:0] = [
        str(ROOT / "src"),
        str(ROOT),
        str(ROOT / "scripts"),
        str(ROOT / "research" / "scripts"),
    ]
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.robots.so101.isaac_env import (
        SO101IsaacEnv,
    )  # the ViperX class reuses its _reset_cube
    from physai.runtime import create_session

    seeds = parse_seeds(args.seeds)
    if args.order == "reverse":
        seeds = seeds[::-1]
    elif args.order == "shuffle":
        seeds = list(np.random.default_rng(args.shuffle_seed).permutation(seeds))
    manifest = with_overrides(
        load_manifest(args.manifest), simulator="isaac", policy="idle", max_steps=10
    )
    env = create_session(manifest, render=True).runtime.robot

    placed: dict = {}
    original = SO101IsaacEnv._reset_cube

    def spy(self):
        out = original(self)
        placed["xy"] = [float(out[0]), float(out[1])]
        placed["z"] = float(np.asarray(self._cube_body.get_world_poses()[0])[0, 2])
        return out

    SO101IsaacEnv._reset_cube = spy
    records = {}
    for seed in seeds:
        env.reset(seed=int(seed))
        after = [float(v) for v in np.asarray(env.cube_pos)]
        records[str(int(seed))] = {
            "placed_xy": placed["xy"],
            "placed_z": placed["z"],
            "after": after,
        }
    env.close()
    out = {
        "manifest": str(args.manifest),
        "order": args.order,
        "seed_order": [int(s) for s in seeds],
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(records)} seeds, order {args.order})")


def shifts(path: str) -> tuple[dict[int, np.ndarray], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(k): np.array(
            [v["after"][0] - v["placed_xy"][0], v["after"][1] - v["placed_xy"][1]]
        )
        for k, v in data["records"].items()
    }, data


def decide(args: argparse.Namespace) -> int:
    (first, d1), (second, d2) = (shifts(p) for p in args.files[:2])
    seeds = sorted(set(first) & set(second))
    if not seeds:
        print("no seed is in both files")
        return 1
    a = np.array([first[s] for s in seeds]) * 1000
    b = np.array([second[s] for s in seeds]) * 1000
    worst_agree = float(np.abs(a - b).max())
    spread = a.std(axis=0)
    print(
        f"{len(seeds)} seeds; shift mm mean {a.mean(axis=0).round(2)} spread over seeds {spread.round(2)}; "
        f"largest disagreement between the two orders {worst_agree:.2f} mm (orders: {d1['order']}, {d2['order']})"
    )
    if worst_agree > AGREE_MM:
        print(
            f"ORDER-DEPENDENT: the shift differs by more than {AGREE_MM} mm between orders; no file written"
        )
        return 2
    source = f"Isaac Sim, {args.files[0]} and {args.files[1]} ({d1['order']}, {d2['order']}), seeds {seeds[0]}-{seeds[-1]}"
    if float(spread.max()) <= CONSTANT_MM:
        dx, dy = (a.mean(axis=0) / 1000).tolist()
        body = {"source": source + "; one offset for every seed", "offset_xy": [dx, dy]}
        print(f"constant shift: offset_xy = [{dx:.6f}, {dy:.6f}] m")
    else:
        raw = json.loads(Path(args.files[0]).read_text(encoding="utf-8"))["records"]
        body = {
            "source": source + "; per seed",
            "xy_by_seed": {str(s): raw[str(s)]["after"][:2] for s in seeds},
        }
        print("the shift differs from seed to seed: writing a table")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(body, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser(
        "measure", help="read the cube pose on Isaac Sim after reset (needs Isaac Sim)"
    )
    m.add_argument("--manifest", required=True)
    m.add_argument(
        "--seeds", default="1000-1099", help="e.g. 1000-1099 or 1000-1010,1050"
    )
    m.add_argument(
        "--order", choices=("forward", "reverse", "shuffle"), default="forward"
    )
    m.add_argument("--shuffle-seed", type=int, default=0)
    m.add_argument("--out", required=True)
    d = sub.add_parser(
        "decide", help="write the cube_start file from two measure files"
    )
    d.add_argument("files", nargs=2)
    d.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "measure":
        measure(args)
        return 0
    return decide(args)


if __name__ == "__main__":
    raise SystemExit(main())
