"""Train Diffusion Policy (LeRobot) on the same recorded demonstrations as `train_act.py`.

Study 1: a second policy class on the same data, image size, cameras and seeds, to see
whether the renderer gap is specific to ACT. It reuses `ACTEpisodeDataset` (one sample = one
timestep with the next `horizon` actions) and adds the previous frame as the second
observation step. The checkpoint folder has the same layout as `train_act.py`'s, with
`algorithm: diffusion` in `training_meta.json`, so `--policy lerobot --checkpoint DIR` loads it.

    python research/imitation_learning/train_diffusion.py --dataset data/randomized_v1 \
        --steps 100000 --lr 1e-4 --num-workers 0 --seed 0 --out outputs/dp_200

This script follows LeRobot's Diffusion Policy defaults: DDPM with 100 inference steps, state and
action normalized min-max to [-1, 1] with sample clipping, the AdamW and cosine-with-warm-up presets of the
config, horizon 64 (the original's default is 16; 64 is LeRobot's) so that the policy reads the cameras about as
rarely as ACT does, the padded end of an episode masked out of the loss, and an exponential moving average
of the weights as in the original Diffusion Policy (`diffusers.training_utils.EMAModel`, power 0.75, at most
0.9999), which is what the saved checkpoint holds. What still differs from the original: batch 16 against 64
(6 GB), 128 px images cropped to 112, two cameras, ImageNet-pretrained ResNet18 with BatchNorm, and the 200 or 400
demonstrations of the study. The U-Net widths are 256/512/1024 (`--unet-width 256`, 89M parameters at horizon 16):
LeRobot's 512/1024/2048 trains in 6 GB without the weight average (5.24 GiB peak at batch 16), but with it the
extra copy of the weights overflows the GPU memory and training falls to 0.6 steps/s (measured), so the study's
runs use 256.

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "src", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers.training_utils import EMAModel  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from physai.data import CheckpointMetadata  # noqa: E402
from research.imitation_learning.act_dataset import ACTEpisodeDataset  # noqa: E402


class WithPreviousFrame(Dataset):
    """`ACTEpisodeDataset` samples with the observation stacked over the previous step.

    Index i-1 is the previous timestep of the same episode except at an episode's first
    step, where the first frame is repeated (the policy does the same at the start of a run).
    """

    def __init__(self, base: ACTEpisodeDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        current = self.base[i]
        first_step = self.base.index[i][1] == 0
        previous = current if first_step else self.base[i - 1]
        sample = dict(current)
        for key, value in current.items():
            if key.startswith("observation."):
                sample[key] = torch.stack([previous[key], value])
        return sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, help="train on the first N episodes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--crop", type=int, default=112)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--n-obs-steps", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=48)
    ap.add_argument("--unet-width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.n_obs_steps != 2:
        raise SystemExit("the dataset wrapper builds exactly two observation steps")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion import (
        DiffusionConfig,
        DiffusionPolicy,
        make_diffusion_pre_post_processors,
    )

    metas = [
        json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        for folder in args.dataset
    ]
    meta = metas[0]
    task = meta["task"]
    base = ACTEpisodeDataset(
        args.dataset,
        chunk_size=args.horizon,
        image_size=args.image_size,
        task=meta.get("task_name", task),
        max_episodes=args.episodes,
    )
    train_set = WithPreviousFrame(base)
    print(f"{len(base.episodes)} episodes, {len(train_set)} training samples")
    stats = base.compute_stats()
    # min-max normalization of the state and the action needs their range too
    for key, attr in (("observation.state", "observation.state"), ("action", "action")):
        values = np.concatenate([e[attr] for e in base.episodes], axis=0)
        stats.per_key[key]["min"] = values.min(0).tolist()
        stats.per_key[key]["max"] = values.max(0).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    stats.to_json(args.out / "dataset_stats.json")

    checkpoint_meta = CheckpointMetadata(
        robot=meta.get("robot", meta.get("robot_type", "unknown")),
        task=meta.get("task_name", task),
        observation_schema=meta.get(
            "observation_schema",
            {"observation.state": meta["features"]["observation.state"]},
        ),
        action_schema=meta.get("action_schema", meta["features"]["action"]),
        normalization=stats.per_key,
        training_config={
            "algorithm": "diffusion",
            "horizon": args.horizon,
            "n_obs_steps": args.n_obs_steps,
            "n_action_steps": args.n_action_steps,
            "unet_width": args.unet_width,
            "version": 2,
            "image_size": args.image_size,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
        },
        scene_name=meta.get("scene_name"),
        scene_config=meta.get("scene_config", {}),
    )
    (args.out / "checkpoint_meta.json").write_text(
        json.dumps(checkpoint_meta.to_dict(), indent=2), encoding="utf-8"
    )
    train_seeds = sorted(
        {
            e["seed"]
            for m in metas
            for e in m.get("episodes", [])[: args.episodes]
            if e.get("seed") is not None
        }
    )
    (args.out / "training_meta.json").write_text(
        json.dumps(
            {
                "algorithm": "diffusion",
                "chunk_size": args.horizon,
                "image_size": args.image_size,
                "task": task,
                "task_name": meta.get("task_name", task),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": args.device,
                "dataset": [str(d) for d in args.dataset],
                "train_seeds": train_seeds,
                "training_seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    side = args.image_size
    cfg = DiffusionConfig(
        input_features={
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, side, side)
            ),
            "observation.images.wrist": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, side, side)
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        crop_shape=(args.crop, args.crop),
        down_dims=(args.unet_width, 2 * args.unet_width, 4 * args.unet_width),
        do_mask_loss_for_padding=True,
        device=args.device,
        optimizer_lr=args.lr,
    )
    preprocessor, _ = make_diffusion_pre_post_processors(
        cfg, dataset_stats=stats.per_key
    )
    policy = DiffusionPolicy(cfg).to(args.device)
    policy.train()
    print(f"{sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M parameters")

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    scheduler = cfg.get_scheduler_preset().build(optimizer, args.steps)
    ema = EMAModel(policy.parameters(), decay=0.9999, use_ema_warmup=True, power=0.75)

    history: list[dict] = []
    step, t0 = 0, time.time()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        loss, _ = policy.forward(preprocessor(batch))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        ema.step(policy.parameters())
        step += 1
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            history.append(
                {"step": step, "loss": loss.item(), "elapsed_s": round(elapsed, 1)}
            )
            print(
                f"step {step:5d}/{args.steps}  loss={loss.item():.4f}  ({elapsed:.0f}s)",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            ema.store(policy.parameters())  # the checkpoint holds the averaged weights
            ema.copy_to(policy.parameters())
            policy.save_pretrained(args.out)
            ema.restore(policy.parameters())
    (args.out / "training_log.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0.0
    total = time.time() - t0
    print(
        f"\ndone: {args.steps} steps in {total:.0f}s ({args.steps / total:.2f} steps/s), "
        f"peak GPU memory {peak:.2f} GiB\ncheckpoint -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
