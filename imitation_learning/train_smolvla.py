"""Fine-tune SmolVLA (a small pretrained vision-language-action model) on recorded demonstrations (study 01).

Same data path as `train_act.py` (`ACTEpisodeDataset` over the .npz episodes, no LeRobotDataset on disk), but the policy starts from
`lerobot/smolvla_base`, the instruction is the task sentence of the dataset, and the optimizer follows the published SO-101 recipe:
action expert trained, vision-language model frozen, cosine decay from 1e-4, chunk 50.

    python research/imitation_learning/train_smolvla.py --dataset data/study01/mujoco_nominal_200_seeds0-199 \\
        data/study01/mujoco_nominal_200_seeds200-399 --steps 20000 --batch-size 64 --out outputs/study01/smolvla_nominal_seed1 --seed 1

`--accum N` runs N micro-batches of `--batch-size / N` per optimizer step, so the batch of the recipe fits in 6 GB without changing it.
`--steps` counts optimizer steps; `--steps 0` only writes the base model with this dataset's features and statistics (the untrained floor). The peak GPU memory and the time per step are written to `training_log.json`.

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

import torch
from torch.utils.data import DataLoader

from physai.data import CheckpointMetadata
from research.imitation_learning.act_dataset import ACTEpisodeDataset, keep_snapshot

BASE = "lerobot/smolvla_base"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, help="the first N episodes of each folder")
    ap.add_argument("--color-jitter", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000, help="optimizer steps")
    ap.add_argument(
        "--batch-size", type=int, default=64, help="samples per optimizer step"
    )
    ap.add_argument(
        "--accum", type=int, default=1, help="micro-batches per optimizer step"
    )
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="size of the cached frames (as the ACT runs)",
    )
    ap.add_argument(
        "--model-image-size",
        type=int,
        default=512,
        help="the model resizes the frames to this (padding, as the base model)",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="bfloat16 autocast for the forward pass (deviation from the reference, write it down)",
    )
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument(
        "--keep-every",
        type=int,
        default=0,
        help="also copy the checkpoint to OUT/step_NNNNNN every N steps (a multiple of --save-every)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, help="seed torch for a repeatable run")
    args = ap.parse_args()
    if args.batch_size % args.accum:
        raise SystemExit("--batch-size must be a multiple of --accum")
    micro = args.batch_size // args.accum
    if args.seed is not None:
        torch.manual_seed(args.seed)

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla import (
        SmolVLAConfig,
        SmolVLAPolicy,
        make_smolvla_pre_post_processors,
    )

    metas = [
        json.loads((d / "meta.json").read_text(encoding="utf-8")) for d in args.dataset
    ]
    meta = metas[0]
    task = meta["task"]
    if len({m.get("task_name", m["task"]) for m in metas}) > 1:
        raise SystemExit("datasets are for different tasks")
    print(
        f"dataset: {[str(d) for d in args.dataset]}  task: {task!r}  device: {args.device}"
    )

    train_set = ACTEpisodeDataset(
        args.dataset,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
        task=task,
        max_episodes=args.episodes,
        color_jitter=args.color_jitter,
    )
    print(f"{len(train_set.episodes)} episodes, {len(train_set)} training samples")
    stats = train_set.compute_stats()
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
            "algorithm": "smolvla",
            "base": BASE,
            "chunk_size": args.chunk_size,
            "image_size": args.image_size,
            "model_image_size": args.model_image_size,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "accum": args.accum,
            "lr": args.lr,
            "bf16": args.bf16,
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
    # training_meta.json, not config.json: save_pretrained() owns that name (see train_act.py)
    (args.out / "training_meta.json").write_text(
        json.dumps(
            {
                "algorithm": "smolvla",
                "base": BASE,
                "chunk_size": args.chunk_size,
                "image_size": args.image_size,
                "model_image_size": args.model_image_size,
                "task": task,
                "task_name": meta.get("task_name", task),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "accum": args.accum,
                "lr": args.lr,
                "bf16": args.bf16,
                "dataset": [str(d) for d in args.dataset],
                "train_seeds": train_seeds,
                "training_seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    state_dim = meta["features"]["observation.state"]["shape"][0]
    action_dim = meta["features"]["action"]["shape"][0]
    S = args.image_size
    cfg = SmolVLAConfig.from_pretrained(BASE)
    cfg.input_features = {
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, S, S)
        ),
        "observation.images.wrist": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, S, S)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
    }
    cfg.chunk_size = cfg.n_action_steps = args.chunk_size
    cfg.resize_imgs_with_padding = (args.model_image_size, args.model_image_size)
    cfg.optimizer_lr = args.lr
    cfg.scheduler_warmup_steps = args.warmup
    cfg.scheduler_decay_steps = args.steps
    cfg.device = args.device
    preprocessor, _ = make_smolvla_pre_post_processors(cfg, dataset_stats=stats.per_key)
    policy = SmolVLAPolicy.from_pretrained(BASE, config=cfg).to(args.device)
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    print(
        f"parameters {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M, trainable {sum(p.numel() for p in params) / 1e6:.0f}M"
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.optimizer_lr,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )
    scheduler = cfg.get_scheduler_preset().build(optimizer, max(args.steps, 1))
    loader = DataLoader(
        train_set,
        batch_size=micro,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    history: list[dict] = []
    data_iter = iter(loader)
    t0 = time.time()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = preprocessor(batch)
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.bf16):
                loss, _ = policy.forward(batch)
            (loss / args.accum).backward()
            total += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(params, cfg.optimizer_grad_clip_norm)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == 1:
            peak = (
                torch.cuda.max_memory_allocated() / 2**30
                if args.device == "cuda"
                else 0.0
            )
            history.append(
                {
                    "step": step,
                    "loss": total,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(time.time() - t0, 1),
                    "peak_gb": round(peak, 2),
                }
            )
            print(
                f"step {step:6d}/{args.steps}  loss={total:.4f}  lr={history[-1]['lr']:.2e}  {history[-1]['elapsed_s']:.0f}s  peak {peak:.2f} GB",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            policy.save_pretrained(args.out)
            (args.out / "training_log.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if args.keep_every and step % args.keep_every == 0:
                keep_snapshot(args.out, step)
    if (
        args.steps == 0
    ):  # the base model as it comes, with this dataset's features and statistics (untrained floor)
        policy.save_pretrained(args.out)
    elapsed = max(time.time() - t0, 1e-9)
    print(
        f"done: {args.steps} steps in {elapsed:.0f}s ({args.steps / elapsed:.3f} steps/s); checkpoint -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
