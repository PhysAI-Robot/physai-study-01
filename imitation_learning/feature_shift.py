"""Does the encoder's view of the first frame predict which seeds fail on Isaac Sim?

For each seed with a MuJoCo and an Isaac first frame (`data/m5_first`, captured with
`eval_policy.py --policy constant --max-steps 2 --save-dataset`), take the ACT image encoder's
pooled features of the front and wrist camera in both renders and measure how far apart they
are. The distance is scored as a predictor of whether the policy fails on that seed on Isaac
(`roc_auc`, 0.5 is chance). A raw pixel distance is scored the same way, so the encoder has to
beat the pixels to count. No simulation runs.

    python research/imitation_learning/feature_shift.py \
        --checkpoint hf:rizgiak/physai-study-01-checkpoints/act_nominal_unseeded \
        --isaac-results outputs/validation_isaac/merged.json \
                        outputs/test_isaac/merged.json \
        --out outputs/feature_shift.json

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "src", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402

FRONT, WRIST = "observation.images.front", "observation.images.wrist"
STATE = "observation.state"


def roc_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = int(positive.sum())
    n_neg = len(score) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(score, positive, n=2000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    while len(aucs) < n:
        pick = rng.integers(0, len(score), len(score))
        if 0 < positive[pick].sum() < len(pick):
            aucs.append(roc_auc(score[pick], positive[pick]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _first_frames(folder: Path) -> dict[int, dict]:
    from physai.data import load_episode

    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    frames = {}
    for entry in meta["episodes"]:
        episode = load_episode(folder / entry["file"], (FRONT, WRIST, STATE))
        frames[entry["seed"]] = {
            "front": episode[FRONT][0],
            "wrist": episode[WRIST][0],
            "state": episode[STATE][0],
        }
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--first-frames", type=Path, default=Path("data/m5_first"))
    ap.add_argument("--isaac-results", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=_ROOT / "research/configs/so101_randomized_pick_place.yaml",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    import research.imitation_learning.lerobot_policy  # noqa: F401  (registers "lerobot")
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.runtime import create_session

    manifest = with_overrides(
        load_manifest(args.manifest), policy="lerobot", simulator="mujoco"
    )
    session = create_session(
        manifest, render=True, policy_kwargs={"checkpoint": args.checkpoint}
    )
    policy = session.runtime.policy

    def features(front, wrist, state) -> dict[str, np.ndarray]:
        sample = {
            FRONT: policy._resize(front),
            WRIST: policy._resize(wrist),
            STATE: torch.from_numpy(np.asarray(state)).float(),
            "task": policy.instruction,
        }
        with torch.no_grad():
            batch = policy.preprocessor(sample)
            out = {}
            for name, key in (("front", FRONT), ("wrist", WRIST)):
                feature_map = policy.policy.model.backbone(batch[key])["feature_map"]
                out[name] = feature_map.mean(dim=(2, 3)).squeeze(0).cpu().numpy()
        return out

    mujoco = _first_frames(args.first_frames / "mujoco")
    isaac = _first_frames(args.first_frames / "isaac")
    outcome: dict[int, bool] = {}
    for path in args.isaac_results:
        for row in json.loads(path.read_text(encoding="utf-8"))["results"]:
            outcome[row["seed"]] = bool(row["success"])

    seeds = sorted(set(mujoco) & set(isaac) & set(outcome))
    rows = []
    for seed in seeds:
        m, i = mujoco[seed], isaac[seed]
        fm = features(m["front"], m["wrist"], m["state"])
        fi = features(i["front"], i["wrist"], i["state"])
        row = {"seed": seed, "fails_on_isaac": not outcome[seed]}
        for cam in ("front", "wrist"):
            row[f"feature_{cam}"] = float(np.linalg.norm(fm[cam] - fi[cam]))
            row[f"pixel_{cam}"] = float(
                np.mean((m[cam].astype(float) - i[cam].astype(float)) ** 2)
            )
        row["feature_both"] = row["feature_front"] + row["feature_wrist"]
        row["pixel_both"] = row["pixel_front"] + row["pixel_wrist"]
        rows.append(row)
    session.close()

    fails = np.array([r["fails_on_isaac"] for r in rows])
    summary = {"seeds": len(rows), "failures": int(fails.sum()), "auc": {}}
    for name in (
        "feature_front",
        "feature_wrist",
        "feature_both",
        "pixel_front",
        "pixel_wrist",
        "pixel_both",
    ):
        score = np.array([r[name] for r in rows])
        low, high = bootstrap_ci(score, fails)
        summary["auc"][name] = {
            "auc": roc_auc(score, fails),
            "ci95": [low, high],
        }
        print(
            f"{name:14s} AUC {summary['auc'][name]['auc']:.3f}  CI {low:.3f}-{high:.3f}"
        )
    print(f"{summary['seeds']} seeds, {summary['failures']} fail on Isaac")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
